"""
Testes da arquitetura CVA + Modelo 2 adaptativo (arquiteturas/cva_m2a.py).
Rodar na raiz:   python -m pytest -q tests/test_cva_m2a.py

Sementes: treino 1-2, calibracao 3, analise 500-599 (politica de sementes do benchmark).
"""
import copy
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from arquiteturas.cva_m2a import ArquiteturaCVAM2A
from arquiteturas.m2_adaptativo import preparar_entrada
from benchmark.sim_ext import COLUNAS_VERDADE, simular_ext


def _sim(horas, seed, **kw):
    return simular_ext(horas, seed=seed, **kw).drop(columns=list(COLUNAS_VERDADE))


@pytest.fixture(scope="module")
def arq():
    tr = [_sim(4, s) for s in (1, 2)]
    return ArquiteturaCVAM2A().treinar(tr).calibrar([_sim(8, 3)])


def test_draw_cruzado_usa_encoders_quando_o_clp_esta_parado(arq):
    df = _sim(3, 510, draw_fator=1.05)
    certo = arq.ma._draw(df)[0]                                  # CLP atualizado
    parado = df.assign(draw_pb1=df["draw_pb1"] / 1.05, draw_pb2=df["draw_pb2"] / 1.05)
    d1 = arq.ma._draw(parado)[0]
    rod = df["v_setpoint"].to_numpy() > 400
    assert np.median(np.abs(d1[rod] / certo[rod] - 1)) < 0.01   # recuperou o draw comandado
    assert np.mean(certo == df["draw_pb1"].to_numpy()) > 0.95   # CLP certo: fica com o CLP


def test_cva_segue_o_material_aceito_pelo_m2(arq):
    """EA -2 % sem aviso: o M2 adaptativo muda theta e a CVA recebe a mesma correcao de torque."""
    df = _sim(6, 511, ea_fator=0.98)
    s = arq._series(df)
    assert s["theta"].min() < 0.99                               # theta acompanhou o material
    ev = arq.analisar(df)
    assert len([e for e in ev if e["modelo"] == "cva"]) <= 1


def test_deriva_detectada_isolada_e_incidente_ligado(arq):
    df = _sim(6, 512, falha="deriva_ganho", rolo="pb2")
    ev = arq.analisar(df)
    for e in ev:
        assert {"t_s", "rolo", "causa", "severidade", "modelo", "incidente"} <= set(e)
    dep = [e for e in ev if e["t_s"] >= 3600]
    assert dep and (dep[0]["causa"], dep[0]["rolo"]) == ("encoder", "pb2")
    cva = {e["incidente"] for e in ev if e["modelo"] == "cva"}
    assert all(e["incidente"] in cva for e in ev if e["modelo"] == "cva+torque")


# ====================================================================== robustez da entrada (F7-F10)
def _falha_pb2():
    return _sim(6, 512, falha="deriva_ganho", rolo="pb2")


def _pos(ev, t0=3600.0):
    return [e for e in ev if e["t_s"] >= t0]


def test_nan_nao_cala_o_modelo(arq):
    """NaN na tensao, no torque, na velocidade, no draw e no setpoint: so os blocos afetados saem; o M2 e
    a CVA continuam alarmando depois (antes, um NaN na tensao calava o M2 pelo resto do arquivo)."""
    df = _falha_pb2()
    base = arq.analisar(df.copy())
    d2 = df.copy()
    for i0, c in ((15000, "tensao_entrada"), (20000, "tq_pb1"), (25000, "v_pb2"), (30000, "draw_pb1"),
                  (35000, "v_setpoint")):
        d2.loc[i0:i0 + 100, c] = np.nan
    with pytest.warns(UserWarning, match="NaN"):
        ev = arq.analisar(d2)
    s = arq._series(arq._preparar(d2))
    assert np.isfinite(s["Z"]).all() and np.isfinite(s["S"][s["ok"]]).all()
    m2_dep = [e for e in ev if e["modelo"] != "cva" and e["t_s"] >= df["t_s"].iloc[35100]]
    assert m2_dep, "M2 mudo depois do NaN"
    assert len(ev) >= 0.8 * len(base)
    assert (_pos(ev)[0]["causa"], _pos(ev)[0]["rolo"]) == ("encoder", "pb2")


def test_lacuna_em_t_s_vira_parada(arq):
    df = _falha_pb2()
    gap = df.drop(index=range(50000, 86000)).reset_index(drop=True)          # 1 h sem amostras
    d, info = preparar_entrada(gap, avisar=False)
    assert info["n_paradas_inseridas"] == 1 and len(d) == len(df)
    t0, t1 = df["t_s"].iloc[50000], df["t_s"].iloc[86000]
    dentro = (d["t_s"] > t0) & (d["t_s"] < t1)
    assert (d.loc[dentro, "v_setpoint"] == 0).all() and d.loc[dentro, "_fora_regime"].all()
    with pytest.warns(UserWarning, match="lacuna"):
        ev = arq.analisar(gap)
    assert not [e for e in ev if t0 < e["t_s"] < t1]
    assert _pos(ev)


def test_sem_draw_e_sem_receita(arq):
    df = _falha_pb2()
    base = _pos(arq.analisar(df.copy()))
    with pytest.warns(UserWarning, match="draw lido pelos encoders"):
        ev = _pos(arq.analisar(df.drop(columns=["draw_pb1", "draw_pb2", "receita"])))
    assert ev and (ev[0]["causa"], ev[0]["rolo"]) == ("encoder", "pb2")
    assert abs(ev[0]["t_s"] - base[0]["t_s"]) <= 600


def test_sem_tensao_modo_degradado(arq):
    """Sem tensao das pontas no treino: usar_tensao_pontas=False decidido no treino (aviso), CVA sem os
    atributos de tensao, analise consistente; modelo treinado COM tensao recusa arquivo sem tensao."""
    sem = ["tensao_entrada", "tensao_saida"]
    tr = [_sim(4, s).drop(columns=sem) for s in (1, 2)]
    with pytest.warns(UserWarning, match="MODO DEGRADADO"):
        a = ArquiteturaCVAM2A().treinar(tr)
    a.calibrar([_sim(8, 3).drop(columns=sem)])
    assert a.ma.usar_tensao is False and a.m2.usar_tensao is False
    assert not [n for n in a.nomes_atributos() if n.startswith("tensao")]
    df = _falha_pb2()
    ev = _pos(a.analisar(df.drop(columns=sem)))
    # detecta; a isolacao confirmada (2 de 3 do M2) aponta o encoder do P&B2 (o 1o alarme pode errar no
    # modo degradado: sem a tensao das pontas o P&B2 carrega o passeio das pontas)
    assert ev and any(e.get("confirmada") and e["confirmada"][:2] == ("encoder", "pb2") for e in ev)
    assert [e["t_s"] for e in a.analisar(df.copy())] == [e["t_s"] for e in a.analisar(df.drop(columns=sem))]
    with pytest.raises(ValueError, match="tensao"):
        arq.analisar(df.drop(columns=sem))


def test_amostragem_5hz_erro_e_reamostragem(arq):
    df = _sim(3, 513, falha="deriva_ganho", rolo="pb1", inicio_h=0.5)
    with pytest.raises(ValueError, match="10 Hz"):
        arq.analisar(df.iloc[::2].reset_index(drop=True))
    # 20 Hz regular (cada amostra repetida no meio do intervalo): reamostrado pela media = o arquivo de 10 Hz
    d20 = pd.concat([df, df.assign(t_s=df["t_s"] + 0.05)]).sort_values("t_s", kind="stable").reset_index(drop=True)
    with pytest.raises(ValueError, match="reamostrar=True"):
        arq.analisar(d20)
    a = copy.deepcopy(arq)
    a.reamostrar = True
    with pytest.warns(UserWarning, match="reamostrado de 20 Hz"):
        ev20 = a.analisar(d20)
    ref = arq.analisar(df.copy())
    assert [(round(e["t_s"], 1), e["modelo"], e["rolo"]) for e in ev20] == \
        [(round(e["t_s"], 1), e["modelo"], e["rolo"]) for e in ref]
    with pytest.warns(UserWarning, match="MAIS LENTA"):
        a.analisar(df.iloc[::2].reset_index(drop=True))               # 5 Hz: roda, com aviso


def test_rolo_errado_e_arquivo_curto(arq):
    df = _sim(0.5, 514)
    with pytest.raises(ValueError, match="rolos"):
        arq.analisar(df.rename(columns={"v_pb1": "v_pb_1", "tq_pb1": "tq_pb_1"}))
    for n in (0, 1, 2, 599):
        assert arq.analisar(df.iloc[:n].copy()) == []


def test_analisar_nao_muda_o_modelo(arq):
    antes = pickle.dumps(arq)
    arq.analisar(_falha_pb2())
    assert pickle.dumps(arq) == antes and not hasattr(arq.ma, "ultimo_log")


# ====================================================================== causalidade (F1, F2)
def _degrau_oculto():
    """Degrau de material sem aviso as 3 h (draw x 0,985 oculto = EA x 0,985) + deriva do festoon as 3,5 h:
    o detector de degrau do M2 adaptativo suspende e depois devolve blocos ('rampa')."""
    df = _sim(8, 505, falha="deriva_ganho", rolo="festoon", inicio_h=3.5,
              receitas=((3 * 3600.0, 0, {"draw": 0.985}),))
    df["draw_pb1"], df["draw_pb2"] = 5e-5, 1e-4                          # o CLP nao muda
    return df


def _chave(e):
    return (round(e["t_s"], 1), e["modelo"], e["rolo"], e["causa"])


def test_causal_sem_alarme_retroativo_e_t_s_na_decisao(arq):
    df = _degrau_oculto()
    s = arq._series(arq._preparar(df))
    assert s["degraus"], "cenario sem degrau detectado"
    ev = arq.analisar(df.copy())
    assert all(e["t_s"] >= e["t_bloco_s"] for e in ev)
    retidos = [e for e in ev if e["t_s"] > e["t_bloco_s"]]
    assert retidos                                                       # alarmes de blocos retidos
    atras = s["dec"] > np.arange(len(s["dec"]))
    dec_t = set(np.round(s["t"][s["dec"][atras]], 3))
    assert all(round(e["t_s"], 3) in dec_t for e in retidos)            # sairam no instante da decisao
    # sem retratacao: a analise de cada prefixo (como em streaming) da os mesmos alarmes ate ali
    for h in (3.6, 4.2, 4.6, 5.5):
        T = df["t_s"].iloc[0] + h * 3600
        pre = arq.analisar(df[df["t_s"] < T].copy())
        assert {_chave(e) for e in pre if e["t_s"] <= T - 180} == {_chave(e) for e in ev if e["t_s"] <= T - 180}, h
    # modo lote (so comparacao): alarmes datados no instante do bloco, inclusive os retidos
    lote = copy.deepcopy(arq)
    lote.causal = lote.ma.causal = False
    evl = lote.analisar(df.copy())
    assert all(e["t_s"] == e["t_bloco_s"] for e in evl)
    assert min(e["t_s"] for e in ev) >= min(e["t_s"] for e in evl)


def test_fim_de_arquivo_com_degrau_pendente_nao_emite(arq):
    """F2: arquivo que termina durante a quarentena de um degrau: no modo causal os blocos retidos nao
    alarmam; no lote (antigo) eram devolvidos."""
    a = copy.deepcopy(arq)
    R0, N, idx, rec = a.ma._blocos(a._preparar(_sim(4, 515)))
    n = len(N)
    degrau = (np.arange(n) >= n - 15)[:, None] * (-0.08 * N)            # degrau nos ultimos 15 blocos
    _, ie, _, log = a.ma._serie(None, _blocos=(R0 + degrau, N, idx, rec))
    assert log["pendentes_fim"] > 0
    assert not np.isin(idx[-log["pendentes_fim"]:], ie).any()
    a.ma.causal = False
    _, ie2, _, log2 = a.ma._serie(None, _blocos=(R0 + degrau, N, idx, rec))
    assert np.isin(idx[-log2["pendentes_fim"]:], ie2).all()


# ====================================================================== EA_nom / raio da configuracao (F4)
def test_ea_nominal_errado_absorvido_no_treino():
    """EA_nom configurado +-30 % errado: com o draw variando dentro da receita de treino, theta0 (escala de
    EA_nom x raio) e estimado no treino e a analise fica igual a da configuracao certa. Com draw constante
    (como no benchmark) theta0 nao e identificavel e fica 1."""
    from benchmark.sim_ext import perfil, tempo_troca

    def tr_draw(h, seed):
        return _sim(h, seed, receitas=((tempo_troca(perfil(h, seed), h / 2 * 3600), 0, {"draw": 1.3}),))

    tr = [tr_draw(4, s) for s in (1, 2)]
    cal = [_sim(8, 3)]
    df = _sim(6, 512, falha="deriva_ganho", rolo="pb1", draw_fator=1.05)
    res = {}
    for ea in (2.0e6, 1.4e6, 2.6e6):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = ArquiteturaCVAM2A(ea_nominal=ea).treinar(tr).calibrar(cal)
        assert a.ma.theta0_identificavel
        res[ea] = (a.ma.theta0, [(round(e["t_s"]), e["rolo"], e["causa"]) for e in a.analisar(df.copy())])
    th_ok = res[2.0e6][0]
    for ea in (1.4e6, 2.6e6):
        assert abs(res[ea][0] / th_ok - 2.0e6 / ea) < 0.03 * (2.0e6 / ea)
        assert res[ea][1] == res[2.0e6][1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = ArquiteturaCVAM2A(ea_nominal=1.4e6).treinar([_sim(4, s) for s in (1, 2)])
    assert not a.ma.theta0_identificavel and a.ma.theta0 == 1.0


def test_coluna_lacuna_do_importador_fica_fora_do_regime(arq):
    """implantacao/importar_trace.py marca lacunas ate 15 min com lacuna = 1 (sinais congelados, sem NaN):
    essas amostras (e 30 s depois) ficam fora do regime; os blocos nao entram na CVA nem no M2."""
    df = _falha_pb2()
    df["lacuna"] = 0
    df.loc[60000:65999, "lacuna"] = 1                                  # 10 min
    d, _ = preparar_entrada(df, avisar=False)
    fora = d["_fora_regime"].to_numpy()
    assert fora[60000:66000].all() and fora[66000:66290].all() and not fora[67000:70000].any()
    s = arq._series(d)
    t0, t1 = df["t_s"].iloc[60000], df["t_s"].iloc[66000]
    assert not ((s["t"] > t0) & (s["t"] <= t1)).any()                   # nenhum bloco de regime na lacuna
    with pytest.warns(UserWarning, match="lacuna=1"):
        ev = arq.analisar(df)
    assert _pos(ev) and not [e for e in ev if t0 < e["t_bloco_s"] <= t1]
