"""
Testes da arquitetura candidata CVA + Modelo 2 (arquiteturas/cva_m2.py).
Rodar na raiz:   python -m pytest -q tests/test_cva_m2.py

Sementes: treino mf2 1-2, calibracao mf2 3, analise 500-599 (politica de sementes do benchmark).
"""
import copy

import numpy as np
import pandas as pd
import pytest

from arquiteturas.cva_m2 import CVA, ArquiteturaCVAM2, MonitorTorqueAquecido
from monitor.conjunto import PARAM_TORQUE
from monitor.torque_v2 import MonitorTorqueV2
from poc.malha_fechada_v2 import simular_mf2


# ---------------------------------------------------------------- CVA isolada (dados sinteticos)
def _var2(n, rng):
    A1 = np.array([[0.5, 0.2], [0.0, 0.3]])
    A2 = np.array([[0.3, 0.0], [0.1, 0.4]])
    x = np.zeros((n, 2))
    for t in range(2, n):
        x[t] = A1 @ x[t - 1] + A2 @ x[t - 2] + rng.normal(size=2)
    return x


def test_ordem_do_var_pelo_aic():
    rng = np.random.default_rng(0)
    F = _var2(4000, rng)
    ini = np.zeros(len(F), bool)
    ini[0] = True
    assert CVA.ordem_var([(F, ini)], 5, "bic")[0] == 2
    assert CVA.ordem_var([(F, ini)], 5, "aic")[0] >= 2          # AIC nao subestima a ordem


def test_rbc_aponta_a_direcao_da_falha():
    rng = np.random.default_rng(1)
    F = rng.normal(size=(3000, 4))
    ini = np.zeros(len(F), bool)
    ini[0] = True
    c = CVA(p=2).treinar([(F, ini)])
    G = rng.normal(size=(50, 4))
    G[:, 2] += 6.0                                         # desvio so no atributo 2
    P, ok = c.passado(G, np.r_[True, np.zeros(49, bool)])
    dirs = [np.eye(4)[k] for k in range(4)]
    votos = [int(np.argmax(c.rbc(P[i], dirs))) for i in np.flatnonzero(ok)]
    assert np.mean(np.array(votos) == 2) > 0.9


# ---------------------------------------------------------------- Modelo 2 com partida a quente
@pytest.fixture(scope="module")
def treino():
    return [simular_mf2(4, seed=s) for s in (1, 2)]


def test_estado_frio_igual_ao_modelo2_original(treino):
    a = MonitorTorqueV2(usar_tensao_pontas=True, **PARAM_TORQUE).treinar(treino)
    b = MonitorTorqueAquecido(usar_tensao_pontas=True, estado_inicial="frio", **PARAM_TORQUE).treinar(treino)
    for r in a.coef:
        np.testing.assert_allclose(a.coef[r], b.coef[r])


def test_auto_quente_so_quando_o_arquivo_comeca_rodando(treino):
    m = MonitorTorqueAquecido(usar_tensao_pontas=True, estado_inicial="auto", **PARAM_TORQUE)
    df = treino[0]
    x_frio = m._x(df, "pb1")                             # simulador comeca parado -> frio
    i = int(np.flatnonzero(df["v_setpoint"].to_numpy() > 400)[0])
    x_quente = m._x(df.iloc[i:].reset_index(drop=True), "pb1")
    col = 4 + m.lags                                     # F15
    assert x_frio[0, col] == pytest.approx(0.0, abs=1e-3)
    assert x_quente[0, col] == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------- ponta a ponta
@pytest.fixture(scope="module")
def arq(treino):
    cal = [simular_mf2(8, seed=3)]
    return ArquiteturaCVAM2().treinar(treino).calibrar(cal)


def test_interface_e_meta_dividida(arq):
    r = arq.resumo()
    assert r["p"] >= 1 and 1 <= r["a"] <= r["p"] * r["n_atributos"]
    assert r["metas"]["cva"] + r["metas"]["torque"] == pytest.approx(1.0)
    assert np.isfinite(arq.L_cva) and np.isfinite(arq.m2.h)


def test_detecta_e_isola_deriva(arq):
    df = simular_mf2(6, seed=500, falha="deriva_ganho", rolo="pb2")
    ev = arq.analisar(df.drop(columns=["severidade"]))
    for e in ev:
        assert {"t_s", "rolo", "causa", "severidade", "modelo"} <= set(e)
        assert e["rolo"] in ("festoon", "pb1", "pb2", None)
        assert e["causa"] in ("encoder", "mecanica", None)
    dep = [e for e in ev if e["t_s"] >= 3600 and e["rolo"] is not None]
    assert dep, "deriva de 0,005 % no pb2 nao detectada"
    assert any((e["causa"], e["rolo"]) == ("encoder", "pb2") for e in dep[:3])


def test_readaptacao_a_draw_novo(arq):
    """Draw diferente do treino nas colunas draw_pb*: as primeiras horas em regime ficam cegas e as medias
    sao re-estimadas (sem a coluna, nada muda)."""
    df = simular_mf2(3, seed=501)
    a = copy.deepcopy(arq)
    a.draw_ref = {"receita": 0.0, "draw_pb1": 5e-5, "draw_pb2": 1e-4}
    s0 = a._series(df.assign(receita=0, draw_pb1=5e-5, draw_pb2=1e-4))   # nominal: nada muda
    assert not s0["mudancas"] and not s0["cego"].any()
    s1 = a._series(df.assign(receita=0, draw_pb1=5.25e-5, draw_pb2=1.05e-4))
    assert s1["mudancas"] == [(0, False)] and s1["cego"][: a.readapt_blocos].all()
    # troca para a receita 1 no meio e volta para a 0: so o trecho da receita 1 readapta
    n = len(df)
    rec = np.zeros(n, int)
    rec[n // 4: n // 2] = 1
    s2 = a._series(df.assign(receita=rec, draw_pb1=5e-5, draw_pb2=1e-4))
    assert [t for _, t in s2["mudancas"]] == [False, True]
    b_volta = s2["mudancas"][1][0]
    assert not s2["cego"][b_volta:].any()
    np.testing.assert_allclose(s2["Z"][b_volta:], s0["Z"][b_volta:])


def test_confirmacao_do_m2_e_o_mesmo_incidente(arq):
    df = simular_mf2(6, seed=502, falha="deriva_ganho", rolo="pb1").drop(columns=["severidade"])
    ev = arq.analisar(df)
    assert all(e.get("incidente") is not None for e in ev)
    cva = {e["incidente"] for e in ev if e["modelo"] == "cva"}
    for e in ev:
        if e["modelo"] == "cva+torque":
            assert e["incidente"] in cva
