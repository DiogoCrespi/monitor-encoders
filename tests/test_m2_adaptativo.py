"""
Testes rapidos da arquitetura candidata arquiteturas/m2_adaptativo.py.
Rodar na raiz:  python -m pytest -q tests/test_m2_adaptativo.py

Sementes: treino 1-2, calibracao 3-8 (politica de sementes do desenvolvimento); arquivos curtos.
"""
import copy

import numpy as np
import pandas as pd
import pytest

import poc.malha_fechada_v2 as sim
from arquiteturas.m2_adaptativo import ArquiteturaM2Adaptativo
from monitor.torque_v2 import HIPOTESES, matriz_assinaturas


def _com_clp(df, d1=sim.DRAW["pb1"], d2=sim.DRAW["pb2"], receita=0):
    df = df.copy()
    df["receita"], df["draw_pb1"], df["draw_pb2"] = receita, d1, d2
    return df


@pytest.fixture(scope="module")
def modelo():
    tr = [_com_clp(sim.simular_mf2(4, seed=1)), _com_clp(sim.simular_mf2(4, seed=2))]
    cal = [_com_clp(sim.simular_mf2(4, seed=s)) for s in (3, 4)]
    return ArquiteturaM2Adaptativo(bloco_h=0.5, semanas_boot=5).treinar(tr).calibrar(cal)


def test_regressor_fisico_de_receita():
    """n = r*EA*(-D1, D1-D2, D2): (-15, -15, +30) N.m no nominal."""
    m = ArquiteturaM2Adaptativo()
    df = _com_clp(sim.simular_mf2(0.05, seed=1))
    n, rid = m._fisico(df)
    np.testing.assert_allclose(n[0], [-15.0, -15.0, 30.0], rtol=1e-9)
    assert (rid == 0).all()


def test_draw_nao_informado_no_pb1_e_identico_a_deriva_do_encoder_pb1():
    """Fundamental: +x no draw do P&B1 (sem CLP) desloca os torques exatamente como o encoder do P&B1
    lendo a menos (rolo rapido) por x. So o setpoint do CLP separa os dois."""
    dD1 = 1e-6
    efeito_draw = sim.R * sim.EA * np.array([-dD1, dD1, 0.0])
    assinatura = matriz_assinaturas()[:, HIPOTESES.index(("encoder", "pb1"))]
    np.testing.assert_allclose(efeito_draw, sim.R * sim.EA * dD1 * assinatura)


def test_draw_lido_pelos_encoders_e_o_comandado_mesmo_com_deriva():
    """A malha faz a leitura seguir a referencia: o draw medido pelos encoders e o COMANDADO,
    com ou sem deriva de ganho - serve de substituto do setpoint, nao de detector."""
    m = ArquiteturaM2Adaptativo(fonte_draw="encoders")
    df = sim.simular_mf2(3, seed=5, falha="deriva_ganho", rolo="pb1", inicio_h=0.5, rampa_h=1.0)
    D1, D2, fonte = m._draw(df)
    ok = (df["severidade"] > 0.99) & (df["v_setpoint"] > 400)
    assert fonte == "encoders"
    assert abs(np.median(D1[ok]) - sim.DRAW["pb1"]) < 0.02 * sim.DRAW["pb1"]
    assert abs(np.median(D2[ok]) - sim.DRAW["pb2"]) < 0.02 * sim.DRAW["pb2"]


def test_reestimativa_de_material_na_quarentena(modelo):
    rng = np.random.default_rng(0)
    N = np.tile([-15.0, -15.0, 30.0], (20, 1))
    R0 = 0.93 * N + modelo.mu + rng.normal(0, 0.05, N.shape)
    th, _ = modelo._reestimar(list(zip(R0, N)), 1.0, 0.0, True, np.array([0, 0, 1.0]))
    assert abs(th - 0.93) < 0.01


def test_interface_e_arquivo_normal(modelo):
    assert modelo.h > 0
    df = _com_clp(sim.simular_mf2(4, seed=6))
    ev = modelo.analisar(df)
    assert isinstance(ev, list)
    for e in ev:
        assert {"t_s", "rolo", "causa", "severidade", "modelo"} <= set(e)
        assert e["causa"] in ("encoder", "mecanica")


def test_deriva_detectada_e_isolada(modelo):
    df = _com_clp(sim.simular_mf2(4, seed=7, falha="deriva_ganho", rolo="pb1", inicio_h=0.5, rampa_h=1.0))
    ev = [e for e in modelo.analisar(df) if e["t_s"] >= 1800]
    assert ev, "deriva de 0,005 % em 1 h nao detectada"
    assert (ev[0]["causa"], ev[0]["rolo"]) == ("encoder", "pb1")


def test_rastreamento_absorve_material_mas_nao_encoder(modelo):
    """Com residuo puro na direcao N (material), theta acompanha; com a assinatura do encoder do P&B1
    (ortogonal a N), a porta trava e theta fica parado."""
    modelo = copy.deepcopy(modelo)
    modelo.tau_ea_h = 1.0
    df = _com_clp(sim.simular_mf2(4, seed=8))
    R0, N, idx, rec = modelo._blocos(df)
    orig = modelo._blocos
    for desvio, deve_mover in ((-0.03 * N, True), (np.tile([-1.5, 1.5, 0.0], (len(N), 1)), False)):
        R1 = R0 + desvio
        modelo._blocos = lambda d, R1=R1: (R1, N, idx, rec)
        try:
            log = modelo._serie(df)[3]
        finally:
            modelo._blocos = orig
        if deve_mover:
            assert abs(log["theta"][-1] - 0.97) < 0.01
        else:
            assert abs(log["theta"][-1] - 1.0) < 0.005


def test_degrau_de_material_absorvido_rampa_nao(modelo):
    """Opcao 5: degrau de -5 % no EA no meio do arquivo (so ao longo de N) vira theta novo (os blocos
    do degrau sao descartados); uma rampa com a assinatura do encoder do FESTOON (que tem componente
    ao longo de N) nao move theta: a componente ortogonal reprova a hipotese de material."""
    modelo = copy.deepcopy(modelo)
    modelo.tau_ea_h = None                                     # isola o detector de degrau
    df = _com_clp(sim.simular_mf2(4, seed=8))
    R0, N, idx, rec = modelo._blocos(df)
    n = len(N)
    t = np.arange(n)
    degrau = (t >= n // 3)[:, None] * (-0.05 * N)
    rampa = np.clip((t - n // 3) / (n / 2), 0, 1)[:, None] * np.array([3.0, 0.0, -3.0])
    log = modelo._serie(None, _blocos=(R0 + degrau, N, idx, rec))[3]
    assert abs(log["theta"][-1] - 0.95) < 0.01
    log = modelo._serie(None, _blocos=(R0 + rampa, N, idx, rec))[3]
    assert abs(log["theta"][-1] - 1.0) < 0.01
