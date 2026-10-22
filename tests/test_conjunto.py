"""
Testes do monitor conjunto (monitor/conjunto.py).  python -m pytest -q tests/test_conjunto.py

So sementes de DESENVOLVIMENTO (36-40); as de teste da validacao conjunta (191-195, 211-220)
ficam para validacao_conjunta.py. Modelo 3 com ordem fixa e Modelo 2 com menos semanas de
reamostragem, so para o teste rodar rapido.
"""
import json
import warnings

import numpy as np
import pytest

from monitor import MonitorConjunto, MonitorEncoder
from monitor.conjunto import MODELOS, SEVERIDADES
from poc.malha_fechada import simular_mf
from poc.malha_fechada_v2 import simular_mf2
from poc.simulador_avisos import ENCODERS, ENCODERS_LINHA, contadores, eventos_do_log, simular, simular_acoplado

FS = 10
RAPIDO = dict(param_torque={"semanas_boot": 15}, param_proximo={"ordem": (16, 3), "r_max": 10, "n_rep": 5})


def cenario(seed, falha=None, rolo="pb1", horas=6):
    so_avisos = falha in ("erros_com", "degradacao_sinal")
    df = simular_mf2(horas, seed=seed, falha=None if so_avisos else falha, rolo=rolo)
    ev, cont, info = simular_acoplado(seed, df["t_s"].to_numpy(), df["severidade"].to_numpy(), falha, rolo)
    return df.assign(t_s=df["t_s"] + info["t0_s"]), ev, cont, info


@pytest.fixture(scope="module")
def monitor():
    treino = [simular_mf2(12, seed=s) for s in (1, 2)]
    cal = [simular_mf2(24, seed=s) for s in (3, 4, 5)]
    return MonitorConjunto(**RAPIDO).treinar(treino).calibrar(cal)


def com_avisos(monitor, ev, cont, info):
    t0 = info["t0_s"]
    return monitor.treinar_avisos(ev[ev["t_s"] < t0], t0, ENCODERS_LINHA, cont[cont["t_s"] < t0])


def test_exportado_no_pacote_sem_quebrar_o_monitor_antigo():
    import monitor
    assert {"MonitorEncoder", "de_simulacao", "validar_formato", "MonitorConjunto"} <= set(monitor.__all__)
    assert MonitorEncoder("pb1", ["festoon", "pb2"]).alvo == "pb1"


def test_liga_os_modelos_conforme_os_dados(monitor):
    assert monitor.componentes["torque"] and monitor.componentes["proximo_passo"] == ["festoon", "pb1", "pb2"]
    assert monitor.torque.usar_tensao                  # o simulador v2 tem tensao das pontas


def test_operacao_normal_sem_alarme_de_sinal_ou_torque(monitor):
    df, ev, cont, info = cenario(36)
    eventos = com_avisos(monitor, ev, cont, info).analisar(df, ev, cont)
    assert not [e for e in eventos if e["modelo"] != "avisos"], [e["mensagem"] for e in eventos]
    for e in eventos:
        assert e["modelo"] in MODELOS and e["severidade"] in SEVERIDADES


def test_deriva_detectada_e_isolada_pelo_torque(monitor):
    df, ev, cont, info = cenario(37, "deriva_ganho", "pb2")
    eventos = com_avisos(monitor, ev, cont, info).analisar(df, ev, cont)
    t_on = info["t0_s"] + 3600
    assert not [e for e in eventos if e["t_s"] < t_on and e["modelo"] != "avisos"], "falso alarme antes da falha"
    dep = [e for e in eventos if e["t_s"] >= t_on and e["modelo"] == "torque"]
    assert dep and dep[0]["encoder"] == "pb2" and dep[0]["tipo"] == "encoder"
    assert any(e["severidade"] == "alta" for e in dep)             # isolacao confirmada
    assert "PB2" in dep[0]["mensagem"] and "Ação" in dep[0]["mensagem"]
    assert dep[0]["origem"]["indicador"] == "torque_v2"


def test_cabo_degradando_detectado_pelos_avisos(monitor):
    df, ev, cont, info = cenario(38, "erros_com", "pb1")
    eventos = com_avisos(monitor, ev, cont, info).analisar(df, ev, cont)
    certos = [e for e in eventos if e["modelo"] == "avisos" and e["encoder"] == "pb1"
              and info["t_inicio_s"] <= e["t_s"] < info["t_falha_s"]]
    assert certos, "degradacao do cabo nao detectada antes da falha final"
    assert certos[0]["tipo"] == "cabo" and "aviso_cabo" in certos[0]["mensagem"]


def test_ruido_detectado_pelo_proprio_sinal_sem_torque():
    """Sem colunas de torque o Modelo 2 fica desligado; o Modelo 3 continua (simulador v1, malha lenta)."""
    sem_tq = lambda d: d.drop(columns=[c for c in d.columns if c.startswith("tq_")])
    m = MonitorConjunto(param_proximo={"ordem": (16, 3), "r_max": 10, "n_rep": 5})
    m.treinar(sem_tq(simular_mf(8, seed=1))).calibrar([sem_tq(simular_mf(12, seed=s)) for s in (2, 3)])
    assert m.torque is None and m.avisos is None
    eventos = m.analisar(sem_tq(simular_mf(6, seed=39, falha="ruido")))
    assert not [e for e in eventos if e["t_s"] < 3600]
    pb1 = [e for e in eventos if e["encoder"] == "pb1"]
    assert pb1 and pb1[0]["modelo"] == "proximo_passo" and pb1[0]["tipo"] in ("ruido", "picos")


def _ev(t, modelo, encoder, sev="media"):
    return {"t_s": t, "modelo": modelo, "encoder": encoder, "rolo": encoder, "tipo": "x", "severidade": sev,
            "mensagem": "m", "corroborado_por": [], "origem": {}}


def test_correlacao_sobe_severidade_so_entre_modelos_diferentes():
    ev = [_ev(0, "avisos", "pb1"), _ev(600, "torque", "pb1"), _ev(700, "torque", "pb1"),
          _ev(800, "proximo_passo", "pb2"), _ev(20000, "proximo_passo", "pb1")]
    out = MonitorConjunto.correlacionar(ev, 7200)
    assert out[0]["severidade"] == "media"
    assert out[1]["severidade"] == "alta" and out[1]["corroborado_por"] == ["avisos"]
    assert out[3]["severidade"] == "media"                          # outro encoder
    assert out[4]["severidade"] == "media"                          # fora da janela


def test_causa_comum_e_mecanica_nao_corroboram():
    ev = [_ev(0, "avisos", "pb1", "baixa"), _ev(100, "torque", None), _ev(200, "proximo_passo", "pb1")]
    out = MonitorConjunto.correlacionar(ev, 7200)
    assert out[0]["severidade"] == "baixa" and out[2]["severidade"] == "media"


def test_avisos_sem_df_exigem_intervalo(monitor):
    _, ev, cont, info = cenario(40, "erros_com", "pb1", horas=2)
    com_avisos(monitor, ev, cont, info)
    with pytest.raises(ValueError, match="t_ini"):
        monitor.analisar(eventos=ev, contadores=cont)
    assert isinstance(monitor.analisar(eventos=ev, contadores=cont, t_ini=info["t0_s"], t_fim=info["t_fim_s"]), list)


def test_exportar_e_carregar_reproduz_eventos(monitor):
    df, ev, cont, info = cenario(37, "deriva_ganho", "pb2")
    com_avisos(monitor, ev, cont, info)
    a = monitor.analisar(df, ev, cont)
    copia = MonitorConjunto.carregar(json.dumps(monitor.exportar()))
    b = copia.analisar(df, ev, cont)
    assert [(e["t_s"], e["modelo"], e["encoder"], e["severidade"]) for e in a] == \
           [(e["t_s"], e["modelo"], e["encoder"], e["severidade"]) for e in b]


def test_mapa_encoder_rolo_e_aviso_quando_nenhum_nome_casa():
    """Log de avisos com nomes de encoder que nao sao nomes de rolo: sem mapa, aviso; com mapa, corrobora."""
    df, info = simular(36, None, dias_hist=7.0, dias_monit=1.0)          # ENCODERS: nomes genericos
    ev, cont = eventos_do_log(df), contadores(df, 60.0, info["t_fim_s"])
    th = info["t_fim_hist_s"]
    hist = (ev[ev["t_s"] < th], th, ENCODERS, cont[cont["t_s"] < th])
    with pytest.warns(UserWarning, match="mapa_encoders"):
        MonitorConjunto().treinar_avisos(*hist)
    enc = list(ENCODERS)[0]
    m = MonitorConjunto(mapa_encoders={enc: "pb1"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m.treinar_avisos(*hist)
    e = m._de_avisos({"t_s": 0.0, "encoder": enc, "familia": "cabo", "mensagem": "x", "causa_comum": False})
    assert e["encoder"] == e["rolo"] == "pb1" and e["origem"]["encoder"] == enc
    out = MonitorConjunto.correlacionar([e, _ev(600, "torque", "pb1")], 7200)
    assert out[1]["corroborado_por"] == ["avisos"] and out[1]["severidade"] == "alta"
    copia = MonitorConjunto.carregar(json.dumps(m.exportar()))
    assert copia.mapa_encoders == {enc: "pb1"}


def test_mensagem_de_picos_usa_o_normal_calibrado(monitor):
    m = monitor.proximo["pb1"]
    lim = m.limiar["contagem"]
    e = monitor._de_proximo("pb1", m, {"contagem": np.array([lim + 2])}, np.array([0]),
                            {"indice": 0, "estatistica": "contagem"}, np.array([0.0]))
    normal = "nenhum" if lim < 1 else f"até {lim:.0f}"
    assert f"(normal: {normal})" in e["mensagem"] and e["tipo"] == "picos"
