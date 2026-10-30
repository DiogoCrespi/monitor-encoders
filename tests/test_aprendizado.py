"""
Testes da arquitetura autoencoder (arquiteturas/aprendizado.py).  python -m pytest -q tests/test_aprendizado.py

Sementes: treino/calibracao mf2 1-4 e desenvolvimento 500-599 (as de teste ficam de fora).
Configuracao reduzida (1 rede, gargalo fixo, poucas semanas de reamostragem) so para rodar rapido.
"""
import numpy as np
import pytest

import comparacao_metodos as cm
from arquiteturas.aprendizado import ArquiteturaAutoencoder, blocos, defasar, p_cauda
from monitor.conjunto import PARAM_TORQUE
from monitor.torque_v2 import MonitorTorqueV2
from poc.malha_fechada_v2 import simular_mf2

RAPIDO = dict(n_modelos=1, grade_k=(6,), max_iter=150, semanas_boot=10)


@pytest.fixture(scope="module")
def treinada():
    a = ArquiteturaAutoencoder(**RAPIDO)
    a.treinar([simular_mf2(6, seed=s) for s in (1, 2)])
    return a.calibrar([simular_mf2(12, seed=s) for s in (3, 4)])


def test_blocos_iguais_aos_da_comparacao():
    df = simular_mf2(2, seed=500)
    F, t, ini = blocos(df)
    F0, t0, ini0, _ = cm.Blocos(MonitorTorqueV2(usar_tensao_pontas=True, **PARAM_TORQUE))(df)
    assert np.allclose(F, F0) and np.allclose(t, t0) and (ini == ini0).all()
    X, ok = defasar(F, ini, 3)
    X0, ok0 = cm.defasar(F0, ini0, 3)
    assert np.allclose(X, X0) and (ok == ok0).all()


def test_p_cauda_igual_ao_da_comparacao():
    rng = np.random.default_rng(500)
    r = np.sort(rng.exponential(size=2000))
    x = np.linspace(0, 12, 50)
    for dentro in (False, True):
        assert np.allclose(p_cauda(r, x, dentro), cm.p_cauda(r, x, dentro))


def test_interface_e_deteccao(treinada):
    a = treinada
    assert a.k == 6 and a.limite is not None and 0 <= a.fa_boot_semana <= 1.0 + 1e-9
    df = simular_mf2(6, seed=501, falha="atrito", rolo="pb1")
    ev = a.analisar(df)
    assert ev, "atrito forte no fim da rampa deveria alarmar"
    for e in ev:
        assert {"t_s", "rolo", "causa", "severidade", "modelo"} <= set(e)
        assert e["rolo"] in ("festoon", "pb1", "pb2") and e["causa"] is None
        assert e["severidade"] in ("baixa", "media", "alta") and e["modelo"] == "autoencoder"
    assert any(e["t_s"] >= 3600 for e in ev)


def test_ordem_das_chamadas():
    with pytest.raises(RuntimeError):
        ArquiteturaAutoencoder().calibrar([])
    with pytest.raises(RuntimeError):
        ArquiteturaAutoencoder().analisar(simular_mf2(1, seed=502))
