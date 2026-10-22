"""
Testes do Modelo 3 v2 (monitor/proximo_passo_v2.py). Rodar na raiz do repositorio:
    python -m pytest -q -p no:cacheprovider tests/test_proximo_passo_v2.py

Usa sementes de DESENVOLVIMENTO (1-40); as de teste (91-95, 101-110) ficam para validacao_proximo_passo_v2.py.
"""
import numpy as np
import pytest

from monitor.proximo_passo_v2 import (ProximoPassoV2, juntar_trechos, cauda_exponencial, de_dataframe, est_cusum_sq,
                                      est_contagem, est_kurt_jan, est_var_jan, eventos, limiar_contagem,
                                      ljung_box, portao)
from poc.malha_fechada import simular_mf

FS = 10
F0 = int(3600 * FS)
CONFIG = {"var_jan": {"W": 3000}, "contagem": {"W": 6000, "k_pico": 5.0}}


@pytest.fixture(scope="module")
def modelo_mf():
    m = ProximoPassoV2(estatisticas=CONFIG, n_rep=5).treinar(*de_dataframe(simular_mf(8, seed=1)))
    return m.calibrar([de_dataframe(simular_mf(12, seed=s)) for s in (2, 3)])


# ---------------------------------------------------------------- pecas
def test_cusum_vetorizado_igual_a_recursao():
    z = np.random.default_rng(0).normal(size=5000) * 1.2
    c, esperado = 0.0, []
    for x in z:
        c = max(0.0, c + x * x - 1.5)
        esperado.append(c)
    np.testing.assert_allclose(est_cusum_sq(z, k=0.5), esperado, atol=1e-9)


def test_janelas_aquecem_sem_alarmar_e_medem_certo():
    z = np.random.default_rng(1).normal(size=20000)
    v, k, c = est_var_jan(z, W=1000), est_kurt_jan(z, W=1000), est_contagem(z, W=1000, k_pico=2.0)
    assert (v[:999] == 0).all() and (k[:999] == 0).all()
    assert abs(v[1000:].mean() - 1) < 0.05 and abs(np.median(k[1000:]) - 3) < 0.2
    assert c[-1] == (np.abs(z[-1000:]) > 2).sum()


def test_portao_suspende_parada_rampa_e_volta_depois():
    sp = np.concatenate([np.zeros(100), np.linspace(0, 600, 300), np.full(1000, 600.0)])
    g = portao(sp, sp > 50, FS, portao_s=20)
    assert not g[:400].any()                       # parada e rampa
    assert not g[400:599].any() and g[600:].all()  # 20 s depois da rampa
    assert (portao(sp, sp > 50, FS, portao_s=None) == (sp > 50)).all()


def test_eventos_respeitam_persistencia_intervalo_e_mapeamento():
    flag = np.zeros(1000, bool)
    flag[10:15] = True                  # curto demais para persist=10
    flag[100:120] = True
    flag[150:170] = True                # dentro do intervalo minimo
    flag[600:620] = True
    idx = np.arange(1000) * 2           # sequencia comprimida -> indice original
    assert eventos(flag, 10, 500, idx) == [2 * 109, 2 * 609]


def test_cauda_exponencial_recupera_quantil():
    x = np.random.default_rng(2).exponential(1.0, 20000)
    # P(X > q) = 1e-6  ->  q = ln(1e6) = 13,8 (muito alem do maior valor observado)
    assert abs(cauda_exponencial(x, 1e-6) - np.log(1e6)) < 1.0


def test_limiar_contagem_exige_mais_de_um_pico_quando_pico_e_raro():
    z = np.random.default_rng(3).normal(size=2_000_000)
    c, p = limiar_contagem([z], W=6000, k_pico=5.0, fs=FS, fa_semana=1.0)
    assert c == 1 and p < 5e-6


# ---------------------------------------------------------------- modelo na malha fechada
def test_arx_com_setpoint_branqueia_o_residuo(modelo_mf):
    p, q = modelo_mf.ordem
    assert q > 0, "o BIC deveria escolher usar o setpoint"
    assert modelo_mf.lb_depois[0] < modelo_mf.lb_antes[0]
    z, _ = modelo_mf.z(*de_dataframe(simular_mf(8, seed=4)))
    assert abs(z.std() - 1) < 0.05
    assert np.abs(ljung_box(z, 30)[2]).max() < 0.05


def test_sem_alarme_em_operacao_normal(modelo_mf):
    _, _, alarmes = modelo_mf.analisar(*de_dataframe(simular_mf(24, seed=31)))
    assert alarmes == []


@pytest.mark.parametrize("falha,estatistica", [("perda_pulsos", "contagem"), ("ruido", "var_jan")])
def test_pega_picos_e_ruido_em_malha_fechada_cedo(modelo_mf, falha, estatistica):
    df = simular_mf(6, seed=12, falha=falha)
    est, idx, alarmes = modelo_mf.analisar(*de_dataframe(df))
    assert not [a for a in alarmes if a["indice"] < F0]
    depois = [a for a in alarmes if a["indice"] >= F0]
    assert depois, "falha nao detectada"
    assert df["severidade"].to_numpy()[depois[0]["indice"]] < 0.5
    assert estatistica in {a["estatistica"] for a in depois}


def test_nao_e_indicador_de_deriva_em_malha_fechada(modelo_mf):
    """Papel do modelo: deriva lenta e com 'vizinhos'/'torque' (o drive esconde a deriva da velocidade lida)."""
    _, _, alarmes = modelo_mf.analisar(*de_dataframe(simular_mf(6, seed=12, falha="deriva_ganho")))
    assert alarmes == []


def test_exportar_e_carregar_reproduz(modelo_mf):
    serie = de_dataframe(simular_mf(6, seed=13, falha="perda_pulsos"))
    copia = ProximoPassoV2.carregar(modelo_mf.exportar())
    assert copia.analisar(*serie)[2] == modelo_mf.analisar(*serie)[2]


def test_treino_em_trechos_fecha_o_portao_na_juncao():
    a, b = de_dataframe(simular_mf(2, seed=4)), de_dataframe(simular_mf(2, seed=5))
    y, sp, rod = juntar_trechos([a, b], 61)
    n = len(a[0])
    assert len(y) == len(sp) == n + len(b[0])
    assert (rod[:n] == a[2]).all() and not rod[n:n + 61].any() and (rod[n + 61:] == b[2][61:]).all()
    m1 = ProximoPassoV2(ordem=(5, 1), r_max=3).treinar(*a)
    m2 = ProximoPassoV2(ordem=(5, 1), r_max=3).treinar_trechos([a])      # um trecho = treino normal
    np.testing.assert_allclose(m1.coef, m2.coef)
    np.testing.assert_allclose(m1.phi, m2.phi)


def test_branqueamento_nao_atravessa_o_portao_fechado(modelo_mf):
    """Depois de o portao fechar e reabrir, z so depende do trecho novo (sem transiente da emenda)."""
    y, sp, rod = de_dataframe(simular_mf(2, seed=6))
    k = len(y) // 2
    rod = rod.copy()
    rod[k:k + 600] = False                                 # 60 s fora do portao
    z1, idx1 = modelo_mf.z(y, sp, rod)
    y2 = y.copy()
    y2[:k] += 50.0                                         # muda so o trecho anterior
    z2, idx2 = modelo_mf.z(y2, sp, rod)
    depois = idx1 >= k + 600
    assert depois.any() and (idx1 == idx2).all()
    np.testing.assert_allclose(z1[depois], z2[depois])
    assert not np.allclose(z1[~depois], z2[~depois])
