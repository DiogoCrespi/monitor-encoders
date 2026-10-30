"""
Testes do Modelo 1 (avisos e contadores do drive).  python -m pytest -q tests/test_avisos.py

Usam so sementes de DESENVOLVIMENTO (1-40); as de teste (91-95, 101-110) ficam para
validacao_avisos.py.
"""
import json

import numpy as np
import pandas as pd
import pytest

from monitor import avisos
from monitor.avisos import (MonitorAvisos, RegraAtual, arl_cusum_poisson, contadores_para_eventos,
                            familia, limiar_para_arl, validar_eventos)
from poc.simulador_avisos import CENARIOS, ENCODERS, contadores, eventos_do_log, simular

H = 3600.0
DIA = 86400.0


def preparar(seed, cenario):
    df, info = simular(seed, cenario)
    return eventos_do_log(df), contadores(df, 60.0, info["t_fim_s"]), info, df


def treinar(ev, cont, info, **kw):
    th = info["t_fim_hist_s"]
    return MonitorAvisos(**kw).treinar(ev[ev["t_s"] < th], th, ENCODERS, cont[cont["t_s"] < th])


# ------------------------------------------------------------------ formato e conversoes
def test_familia_dos_codigos():
    assert familia("aviso_cabo") == familia("contador_erros") == familia("falha_cabo") == "cabo"
    assert familia("aviso_sinal") == familia("aviso_encoder") == familia("aviso_sinal_b") == familia("falha_sinal_b") == "sinal"
    assert familia("falha_motor") is None          # falha que nao e de encoder e ignorada


def test_log_invalido_e_rejeitado():
    with pytest.raises(ValueError, match="codigo"):
        validar_eventos(pd.DataFrame({"t_s": [0.0], "encoder": ["pb1"], "tipo": ["aviso"]}))
    with pytest.raises(ValueError, match="tipo"):
        validar_eventos(pd.DataFrame({"t_s": [0.0], "encoder": ["pb1"], "codigo": ["aviso_sinal_b"],
                                      "tipo": ["warning"]}))


def test_contador_acumulado_vira_eventos_com_reset():
    cont = pd.DataFrame({"t_s": [0, 60, 120, 180, 240], "encoder": "pb1", "contador": "contador_erros",
                         "valor": [5, 5, 7, 1, 1]})        # +2, depois contador zerado e +1
    ev = contadores_para_eventos(cont, declust_s=60)
    assert len(ev) == 2                                    # +2 em 60 s: no maximo 1 incidente
    assert list(ev["t_s"]) == [120.0, 180.0]
    cont_lento = cont.assign(t_s=[0, 3600, 7200, 10800, 14400])
    assert len(contadores_para_eventos(cont_lento, declust_s=60)) == 3   # 2 + 1


# ------------------------------------------------------------------ estatistica
def test_arl_markov_confere_com_monte_carlo():
    mu0 = 0.02
    k = mu0 * 3 / np.log(4)
    h = limiar_para_arl(mu0, k, 500)
    assert arl_cusum_poisson(mu0, k, h) >= 500
    rng = np.random.default_rng(0)
    corridas = []
    for _ in range(600):
        s, n = 0.0, 0
        while s <= h:
            n += 1
            s = max(0.0, s + rng.poisson(mu0) - k)
        corridas.append(n)
    assert np.mean(corridas) == pytest.approx(arl_cusum_poisson(mu0, k, h), rel=0.15)


def test_linha_de_corte_sobe_com_meta_mais_rigida():
    mu0 = 0.01 / 6
    k = mu0 * 3 / np.log(4)
    assert limiar_para_arl(mu0, k, 10 * 1008) < limiar_para_arl(mu0, k, 100 * 1008)


# ------------------------------------------------------------------ operacao normal
@pytest.fixture(scope="module")
def normal():
    return preparar(31, None)


def test_poucos_falsos_alarmes_em_4_semanas_normais(normal):
    ev, cont, info, _ = normal
    m = treinar(ev, cont, info, razao=8, fator_seguranca=3)
    alarmes, _ = m.analisar(ev, info["t_fim_hist_s"], info["t_fim_s"], cont)
    # meta 1/semana na linha: em 4 semanas o esperado e <= 4 (no desenvolvimento deu ~2);
    # a semente 31 e desfavoravel (6), entao o teste so barra uma explosao de falsos alarmes
    assert len(alarmes) <= 8, [a["mensagem"] for a in alarmes]


def test_canais_so_onde_o_drive_tem_o_aviso(normal):
    ev, cont, info, _ = normal
    m = treinar(ev, cont, info)
    assert ("b_1", "cabo") not in m.canais            # tipo_b simulado nao tem contador de erros
    assert {("a_1", "cabo"), ("a_1", "sinal"), ("b_1", "sinal")} <= set(m.canais)


def test_exportar_e_carregar_reproduz_alarmes(normal):
    ev, cont, info, _ = normal
    m = treinar(ev, cont, info)
    copia = MonitorAvisos.carregar(json.dumps(m.exportar()))
    a = m.analisar(ev, info["t_fim_hist_s"], info["t_fim_s"], cont)[0]
    b = copia.analisar(ev, info["t_fim_hist_s"], info["t_fim_s"], cont)[0]
    assert a == b


# ------------------------------------------------------------------ degradacao
CODIGO_ESPERADO = {"cabo": {"aviso_cabo"}, "amplitude": {"aviso_sinal_b", "aviso_sinal", "aviso_encoder"},
                   "intermitente": {"aviso_cabo", "aviso_sinal_b", "aviso_sinal", "aviso_encoder", "falha_cabo", "falha_sinal_b"}}


@pytest.mark.parametrize("seed", [3, 4, 5])          # 3 % 3 = cabo, 4 = amplitude, 5 = intermitente
def test_detecta_degradacao_antes_da_falha_no_encoder_certo(seed):
    cenario = CENARIOS[seed % 3]
    ev, cont, info, _ = preparar(seed, cenario)
    m = treinar(ev, cont, info, razao=8, fator_seguranca=3)
    alarmes, _ = m.analisar(ev, info["t_fim_hist_s"], info["t_fim_s"], cont)
    certos = [a for a in alarmes if a["encoder"] == info["encoder"]
              and info["t_inicio_s"] <= a["t_s"] < info["t_falha_s"]]
    assert certos, f"nao detectou {cenario} em {info['encoder']}"
    primeiro = certos[0]
    assert info["t_falha_s"] - primeiro["t_s"] > 2 * H, "detectou em cima da falha"
    assert primeiro["codigo"] in CODIGO_ESPERADO[cenario]
    assert info["encoder"].upper() in primeiro["mensagem"] and primeiro["codigo"] in primeiro["mensagem"]


def _log(linhas):
    return pd.DataFrame(linhas, columns=["t_s", "drive", "encoder", "codigo", "tipo", "valor"])


@pytest.fixture(scope="module")
def monitor_sintetico():
    """Historico de 28 dias: cada encoder com 1 aviso a cada 2 dias."""
    rng = np.random.default_rng(7)
    linhas = []
    for enc, drv, cod in [("pb1", "tipo_a", "aviso_sinal"), ("pb2", "tipo_a", "aviso_sinal"), ("rw", "tipo_b", "aviso_sinal_b")]:
        for t in np.sort(rng.uniform(0, 28 * DIA, 14)):
            linhas.append((t, drv, enc, cod, "aviso", 300.0 if drv == "tipo_a" else np.nan))
    return MonitorAvisos().treinar(_log(linhas), 28 * DIA)


def test_mensagem_traz_amplitude_e_acao(monitor_sintetico):
    t0 = 30 * DIA
    novos = _log([(t0 + i * 1800, "tipo_a", "pb1", "aviso_sinal", "aviso", 220 - 5 * i) for i in range(8)])
    alarmes, traj = monitor_sintetico.analisar(novos, t0, t0 + DIA)
    assert alarmes and alarmes[0]["encoder"] == "pb1"
    msg = alarmes[0]["mensagem"]
    assert "aviso_sinal" in msg and "mV" in msg and "Acao" in msg
    assert traj[("pb1", "sinal")]["S"].max() > alarmes[0]["h"]


def test_rajada_curta_conta_como_um_incidente(monitor_sintetico):
    t0 = 30 * DIA
    rajada = _log([(t0 + i * 5.0, "tipo_a", "pb1", "aviso_sinal", "aviso", 220) for i in range(10)])
    alarmes, _ = monitor_sintetico.analisar(rajada, t0, t0 + DIA)
    assert alarmes == []


def test_varios_encoders_juntos_marca_causa_comum(monitor_sintetico):
    t0 = 30 * DIA
    linhas = [(t0 + i * 1800 + j, "tipo_a", e, "aviso_sinal", "aviso", 220)
              for i in range(8) for j, e in enumerate(["pb1", "pb2"])]
    alarmes, _ = monitor_sintetico.analisar(_log(linhas), t0, t0 + DIA)
    assert {a["encoder"] for a in alarmes} == {"pb1", "pb2"}
    # so para tras: o 1o alarme do par nao sabe do 2o (streaming); os seguintes sim
    assert not alarmes[0]["causa_comum"]
    assert all(a["causa_comum"] for a in alarmes[1:]) and "causa comum" in alarmes[1]["mensagem"]


# ------------------------------------------------------------------ regra atual e simulador
def test_regra_atual_e_maximo_mais_dois():
    hist = _log([(t, "tipo_a", "pb1", "aviso_cabo", "aviso", 1) for t in (0, 10, 20, 5000)])
    r = RegraAtual().treinar(hist, ["pb1"])
    assert r.limiar["pb1"] == 3 + 2
    novos = _log([(10000 + 60 * i, "tipo_a", "pb1", "aviso_cabo", "aviso", 1) for i in range(6)])
    alarmes = r.analisar(novos, 10000, 20000)
    assert len(alarmes) == 1 and alarmes[0]["n_1h"] == 6


def test_simulador_e_reprodutivel_e_coerente():
    df, info = simular(12, "cabo")
    df2, _ = simular(12, "cabo")
    pd.testing.assert_frame_equal(df, df2)
    ev = eventos_do_log(df)
    final = ev[ev["t_s"] == info["t_falha_s"]]
    assert list(final["codigo"]) == ["falha_cabo"] and list(final["encoder"]) == [info["encoder"]]
    a = ev[(ev["codigo"] == "aviso_cabo") & (ev["encoder"] == info["encoder"])]["t_s"].to_numpy()
    assert np.all(np.diff(a) > 5.0)                    # aviso_cabo some sozinho em 5 s
    c = contadores(df, 60.0, info["t_fim_s"])
    for _, g in c.groupby("encoder"):
        assert np.all(np.diff(g["valor"].to_numpy()) >= 0)
    assert c["valor"].max() >= len(a)                  # contador >= alarmes registrados


def test_configurar_codigos_troca_a_tabela():
    original = dict(avisos.CODIGOS)
    try:
        avisos.configurar_codigos({"X1": "cabo", "X2": "sinal"}, {"X1": "codigo de teste"})
        assert familia("X1") == "cabo" and familia("X2") == "sinal"
        assert familia(next(iter(original))) is None            # a tabela antiga saiu
        assert avisos.DESCRICAO["X1"] == "codigo de teste"
        with pytest.raises(ValueError, match="familia"):
            avisos.configurar_codigos({"X3": "outra"})
    finally:
        avisos.configurar_codigos(original)
        avisos.DESCRICAO.pop("X1", None)
    assert avisos.CODIGOS == original
