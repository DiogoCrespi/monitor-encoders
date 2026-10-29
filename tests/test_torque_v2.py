"""
Testes do Modelo 2 (torque esperado x real) e do simulador v2.
Rodar na raiz do repositorio:   python -m pytest -q tests/test_torque_v2.py

So usam sementes de desenvolvimento (<= 40).
"""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from monitor.torque_v2 import HIPOTESES, MonitorTorqueV2, matriz_assinaturas
from poc.malha_fechada_v2 import ASSINATURA, DELTA_DERIVA, DRAW, EA, R, ROLOS, _pb_variavel, simular_mf2

FS = 10
F0 = int(3600 * FS)


def _delta(df, col):
    rod = df["v_setpoint"] > 300
    a = df[(df.t_s > 1800) & (df.t_s < 3600) & rod]
    b = df[(df.severidade > 0.9) & rod]
    return b[col].mean() - a[col].mean()


# ---------------------------------------------------------------- simulador v2

def test_filtro_tau_variavel_igual_ao_laco():
    rng = np.random.default_rng(0)
    u, a = rng.normal(size=1000), rng.uniform(0.3, 1.0, 1000)
    y_ref, y = np.empty(1000), 5.0
    for i in range(1000):
        y = a[i] * y + (1 - a[i]) * u[i]
        y_ref[i] = y
    np.testing.assert_allclose(_pb_variavel(u, a, 5.0, bloco=64), y_ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("rolo", ROLOS)
def test_assinatura_da_deriva_nos_tres_torques(rolo):
    df = simular_mf2(6, seed=11, falha="deriva_ganho", rolo=rolo)
    d = np.array([_delta(df, f"tq_{r}") for r in ROLOS])
    k = EA * R * DELTA_DERIVA                                  # 15 N.m
    esperado = ASSINATURA[rolo] * k
    assert np.all(np.abs(d - esperado) < 0.25 * k), (rolo, d.round(1), esperado)


def test_malha_rapida_esconde_a_deriva_da_velocidade_lida():
    df = simular_mf2(6, seed=12, falha="deriva_ganho", rolo="pb1")
    fim = df[(df.severidade > 0.9) & (df.v_setpoint > 300)]
    lida_vs_ref = (fim.v_pb1 - fim.v_festoon * (1 + DRAW["pb1"])).mean()    # m/min, vs vizinho
    real_vs_lida = (fim.v_real_pb1 - fim.v_pb1).mean()
    assert abs(lida_vs_ref) < 0.005, lida_vs_ref               # o drive "corrige" a leitura
    assert real_vs_lida > 0.8 * DELTA_DERIVA * fim.v_setpoint.mean(), real_vs_lida   # o rolo gira rapido


def test_modo_tensao_leva_a_deriva_para_o_trim():
    df = simular_mf2(6, seed=13, falha="deriva_ganho", rolo="pb2", modo_tensao=("pb2",))
    k = EA * R * DELTA_DERIVA                                  # o que apareceria sem a malha de tensao
    assert abs(_delta(df, "tq_pb2")) < 0.25 * k and abs(_delta(df, "tq_pb1")) < 0.25 * k
    assert _delta(df, "trim_pb2") < -0.02                      # m/min: o trim compensa a deriva


# ---------------------------------------------------------------- monitor v2

def test_assinaturas_de_encoder_somam_zero():
    H = matriz_assinaturas()
    assert np.allclose(H[:, :3].sum(axis=0), 0)                # encoder nao muda a soma dos torques
    assert len(HIPOTESES) == 6


def test_cusum_vetorizada_igual_a_recursao():
    rng = np.random.default_rng(1)
    z = rng.normal(0.3, 1, (500, 4))
    m = MonitorTorqueV2(k=0.5)
    c0 = np.array([0.0, 1.0, 3.0, 0.2])
    C = m._passo(z, c0)
    c, ref = c0.copy(), []
    for t in range(len(z)):
        c = np.maximum(0, c + z[t] - 0.5)
        ref.append(c.copy())
    np.testing.assert_allclose(C, np.array(ref), atol=1e-9)


def test_ewma_com_estado_igual_a_inteira():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(300, 2))
    m = MonitorTorqueV2(estatistica="ewma", lam=0.1)
    inteira = m._passo(z, np.zeros(2))
    partes = m._passo(z[150:], inteira[149])
    np.testing.assert_allclose(partes, inteira[150:], atol=1e-9)


@pytest.fixture(scope="module")
def monitor():
    treino = [simular_mf2(12, seed=s) for s in (1, 2)]
    cal = [simular_mf2(24, seed=s) for s in (3, 4, 5)]
    return MonitorTorqueV2(k=3.0, bloco_h=2.0, semanas_boot=15).treinar(treino).calibrar(cal)


def test_limiar_respeita_a_meta_na_reamostragem(monitor):
    assert monitor.fa_boot_semana <= 1.0
    assert monitor.h > 0


def test_portao_suspende_parada_rampa_e_emenda(monitor):
    df = simular_mf2(3, seed=30)
    g = monitor.portao(df)
    sp = df["v_setpoint"].to_numpy()
    assert not g[sp < 200].any()
    mudou_2s = np.abs(sp - np.concatenate([np.full(20, sp[0]), sp[:-20]])) >= 1.0   # rampa/emenda
    assert not g[mudou_2s].any()
    assert g.mean() > 0.5


@pytest.mark.parametrize("rolo,seed", [("festoon", 31), ("pb1", 32), ("pb2", 33)])
def test_detecta_e_isola_deriva_de_cada_encoder(monitor, rolo, seed):
    _, ev = monitor.analisar(simular_mf2(6, seed=seed, falha="deriva_ganho", rolo=rolo))
    assert not [e for e in ev if e["indice"] < F0], "falso alarme antes da falha"
    dep = [e for e in ev if e["indice"] >= F0]
    assert dep, "deriva nao detectada"
    assert dep[0]["indice"] < F0 + 2 * 3600 * FS               # antes de 50 % de severidade
    conf = [e["confirmada"] for e in dep if e["confirmada"]]
    assert conf and conf[0] == ("encoder", rolo, 1), conf[:3]
    assert "ENCODER" in dep[-1]["mensagem"] and "a menos" in dep[-1]["mensagem"]


def test_atrito_e_classificado_como_mecanica(monitor):
    _, ev = monitor.analisar(simular_mf2(6, seed=34, falha="atrito", rolo="pb1"))
    dep = [e for e in ev if e["indice"] >= F0]
    conf = [e["confirmada"] for e in dep if e["confirmada"]]
    assert conf and conf[-1] == ("mecanica", "pb1", 1), [(e["tipo"], e["rolo"]) for e in dep]


def test_exportar_e_carregar_reproduz_resultado(monitor):
    df = simular_mf2(6, seed=35, falha="deriva_ganho", rolo="pb2")
    ind_a, ev_a = monitor.analisar(df)
    copia = MonitorTorqueV2.carregar(json.dumps(monitor.exportar()))
    ind_b, ev_b = copia.analisar(df)
    assert ev_a == ev_b
    np.testing.assert_allclose(ind_a["estatistica"], ind_b["estatistica"])


def test_exportar_preserva_parametros_da_calibracao(monitor):
    """Recalibrar um modelo carregado deve usar os mesmos blocos de 2 h (com 1 h o FA sai subestimado)."""
    copia = MonitorTorqueV2.carregar(json.dumps(monitor.exportar()))
    assert (copia.bloco, copia.meta, copia.semanas, copia.seed_boot) == \
        (monitor.bloco, monitor.meta, monitor.semanas, monitor.seed_boot)


def test_formato_invalido_e_rejeitado(monitor):
    with pytest.raises(ValueError, match="tq_pb2"):
        monitor.analisar(pd.DataFrame({"t_s": [0.0], "v_setpoint": [0.0], "v_festoon": [0.0], "v_pb1": [0.0],
                                       "v_pb2": [0.0], "tq_festoon": [0.0], "tq_pb1": [0.0]}))


def test_calibracao_curta_demais_da_erro_claro():
    """A reamostragem sorteia blocos de bloco_h horas de regime: com menos que isso, erro explicito."""
    m = MonitorTorqueV2(bloco_h=2.0, semanas_boot=2).treinar(simular_mf2(3, seed=14))
    with pytest.raises(ValueError, match="calibracao curta demais"):
        m.calibrar(simular_mf2(1, seed=15))
