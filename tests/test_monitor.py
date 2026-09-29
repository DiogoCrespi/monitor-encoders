"""
Testes do monitor. Rodar na raiz do repositorio:   python -m pytest -q

Criterios:
  - zero falso alarme em 24 h de operacao normal nao vista no treino
  - toda falha simulada gera alarme, pelo indicador fisicamente esperado
  - alarme chega enquanto a falha ainda e leve (severidade < 50%)
  - modelo exportado em JSON reproduz exatamente o mesmo resultado
"""
import json

import numpy as np
import pandas as pd
import pytest

from monitor import MonitorEncoder, de_simulacao
from poc.poc_deteccao import simular

FS = 10
F0 = int(1.0 * 3600 * FS)          # falha comeca em 1 h


@pytest.fixture(scope="module")
def monitor():
    treino = de_simulacao(simular(8, seed=1))
    val = de_simulacao(simular(8, seed=2))
    return MonitorEncoder("pb1", ["festoon", "pb2"], fs=FS).treinar(treino).calibrar(val)


@pytest.fixture(scope="module")
def normal_24h():
    return de_simulacao(simular(24, seed=3))


def cenario(falha, seed):
    df = simular(6, seed=seed, falha=falha)
    return de_simulacao(df), df["severidade"].to_numpy()


def test_formato_invalido_e_rejeitado(monitor):
    with pytest.raises(ValueError, match="v_pb2"):
        monitor.analisar(pd.DataFrame({"t_s": [0.0], "v_setpoint": [0.0], "v_pb1": [0.0],
                                       "v_festoon": [0.0]}))


def test_sem_falso_alarme_em_24h_normal(monitor, normal_24h):
    _, eventos = monitor.analisar(normal_24h)
    assert eventos == [], f"falsos alarmes: {eventos}"


# falha -> indicadores que PODEM explicar o primeiro alarme
ESPERADO = {
    "perda_pulsos": {"vizinhos", "proprio"},
    "ruido": {"vizinhos", "proprio"},
    "deriva_ganho": {"vizinhos"},
    "erros_com": {"erros"},
}


@pytest.mark.parametrize("seed", [10, 20, 30])
@pytest.mark.parametrize("falha", list(ESPERADO))
def test_detecta_falha_cedo_e_pelo_indicador_certo(monitor, falha, seed):
    df, sev = cenario(falha, seed)
    _, eventos = monitor.analisar(df)
    antes = [e for e in eventos if e["indice"] < F0]
    depois = [e for e in eventos if e["indice"] >= F0]
    assert not antes, f"falso alarme antes da falha: {antes}"
    assert depois, "falha nao detectada"
    primeiro = depois[0]
    assert primeiro["indicador"] in ESPERADO[falha], primeiro
    assert sev[primeiro["indice"]] < 0.5, f"detectou tarde (severidade {sev[primeiro['indice']]:.2f})"


def test_mensagem_de_deriva_aponta_leitura_abaixo(monitor):
    df, _ = cenario("deriva_ganho", 10)
    _, eventos = monitor.analisar(df)
    viz = [e for e in eventos if e["indicador"] == "vizinhos"]
    assert viz and "abaixo" in viz[0]["mensagem"]


def test_exportar_e_carregar_reproduz_resultado(monitor):
    df, _ = cenario("perda_pulsos", 10)
    ind_a, ev_a = monitor.analisar(df)
    copia = MonitorEncoder.carregar(json.dumps(monitor.exportar()))
    ind_b, ev_b = copia.analisar(df)
    assert ev_a == ev_b
    np.testing.assert_allclose(ind_a["vizinhos"], ind_b["vizinhos"])
