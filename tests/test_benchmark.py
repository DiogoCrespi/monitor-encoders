"""Testes rapidos do benchmark (benchmark/, arquiteturas/). Nao usam sementes 'final' (800-999)."""
import numpy as np
import pandas as pd
import pytest

from arquiteturas.base import Arquitetura, alarme
from benchmark import avaliar as A
from benchmark import cenarios as C
from benchmark.sim_ext import simular_acoplado_ext, simular_ext
from poc.malha_fechada_v2 import simular_mf2
from poc.simulador_avisos import simular_acoplado


@pytest.mark.parametrize("kw", [{}, {"falha": "ruido", "rolo": "pb2"}, {"falha": "perda_pulsos", "rolo": "festoon"},
                                {"falha": "deriva_ganho", "modo_tensao": ("pb2",)}])
def test_sim_ext_padrao_igual_ao_original(kw):
    a = simular_mf2(0.5, seed=3, inicio_h=0.1, rampa_h=0.2, **kw)
    b = simular_ext(0.5, seed=3, inicio_h=0.1, rampa_h=0.2, **kw)
    pd.testing.assert_frame_equal(a, b[a.columns])


@pytest.mark.parametrize("falha", [None, "ruido", "perda_pulsos", "erros_com"])
def test_avisos_ext_padrao_igual_ao_original(falha):
    t = np.arange(0, 3600 * 2, 0.1)
    sev = np.clip((t - 1800) / 3600, 0, 1)
    x, y = simular_acoplado(5, t, sev, falha, "pb1"), simular_acoplado_ext(5, t, sev, falha, "pb1")
    pd.testing.assert_frame_equal(x[0], y[0])
    pd.testing.assert_frame_equal(x[1], y[1])


def test_variantes_mudam_o_sinal_e_sao_deterministicas():
    base = simular_ext(0.5, seed=4)
    for kw in ({"ea_fator": 0.95}, {"draw_fator": 1.05}, {"atrito_fator": 1.1}, {"ruido_hf": 1e-5},
               {"raio": {"pb2": 0.2}}):
        a, b = simular_ext(0.5, seed=4, **kw), simular_ext(0.5, seed=4, **kw)
        pd.testing.assert_frame_equal(a, b)
        assert not a.equals(base), kw
    q = simular_ext(0.5, seed=4, pre_h=2.0)
    assert q["v_setpoint"].iloc[0] > 400 and q["t_s"].iloc[0] == 0 and len(q) == len(base)


def test_cenario_troca_receita_e_sem_verdade():
    c = C.Cenario("troca_receita", "completa", None, "pb1", 550, 3.0, trocas=((1.0, 1, C.RECEITAS["completa"]),))
    g = C.gerar(c)
    df = g["df"]
    assert set(df["receita"]) == {0, 1}
    assert not set(C.COLUNAS_VERDADE) & set(df.columns)
    assert df["t_s"].iloc[0] == g["t0"] and (g["eventos"]["t_s"] < g["t0"]).any()
    assert "contadores" in g["eventos"].attrs
    i = df["receita"].diff().fillna(0).ne(0).idxmax()
    assert df["v_setpoint"].iloc[i] == 0                    # troca feita com a linha parada


def test_sementes_dev_e_final_disjuntas_e_tamanho_final():
    dev = C.montar(list(C.FAMILIAS), "dev", 1, 1)
    assert all(500 <= c.seed <= 599 for c in dev)
    assert set(C.SEMENTES["final"]["falha"]) | set(C.SEMENTES["final"]["normal"]) == set(range(800, 1000))
    nf, nn = C.TAMANHO["final"]["nominal"]
    assert nf >= 10 and nn * C.HORAS_NORMAL >= 960
    with pytest.raises(SystemExit):
        A.main(["--seeds", "final"])                        # sem --confirmar-final


def test_sementes_confirmacao_reservadas_e_disjuntas():
    conf = set(C.SEMENTES["confirmacao"]["falha"]) | set(C.SEMENTES["confirmacao"]["normal"])
    assert conf == set(range(1000, 1200))
    outras = {s for k in ("dev", "final") for v in C.SEMENTES[k].values() for s in v}
    assert not conf & outras and not conf & set(C.TREINO + C.CAL_LONGA)
    assert C.TAMANHO["confirmacao"] == C.TAMANHO["final"] and "confirmacao" in C.RESERVADAS
    for argv in (["--seeds", "confirmacao"], ["--seeds", "confirmacao", "--confirmar-final"],
                 ["--seeds", "final", "--confirmar-confirmacao"],
                 ["--seeds", "confirmacao", "--confirmar-confirmacao", "--cache-modelos"],
                 ["--seeds", "confirmacao", "--confirmar-confirmacao", "--n-falha", "1"]):
        with pytest.raises(SystemExit):
            A.main(argv)                                    # nunca chega a rodar


def test_limite_95_poisson():
    assert A.limite_95(0, 168) == pytest.approx(-np.log(0.05), rel=1e-6)   # 3.0 por semana
    assert A.limite_95(1, 240) == pytest.approx(4.744 / 240 * 168, rel=1e-3)


def test_medir_deteccao_isolacao_e_fa():
    g = {"falha": "atrito", "rolo": "pb1", "causa": "mecanica", "t_on": 1000.0, "t_lim": np.inf, "t0": 0.0,
         "horas": 2.0}
    al = [alarme(50, "pb2", "encoder"), alarme(2000, "pb1", "encoder"), alarme(3000, "pb1", "mecanica"),
          alarme(4000, "pb1", "mecanica")]
    r = A.medir(al, g)
    assert r["fa_pre"] == 1 and r["detectado"] and r["atraso_min"] == pytest.approx(1000 / 60)
    assert r["iso1"] is False and r["iso_conf"] is True and r["errados"] == 1
    assert A.medir(al, {**g, "falha": None})["fa"] == 4


class Nula(Arquitetura):
    nome = "nula"

    def treinar(self, normais, eventos_normais=None, duracao_eventos_s=None):
        return self

    def calibrar(self, normais, meta_fa_semana=1.0):
        return self

    def analisar(self, df, eventos=None):
        return []


class Sempre(Nula):
    nome = "sempre"

    def analisar(self, df, eventos=None):
        return [alarme(t, "pb1", "encoder", "media", "teste") for t in df["t_s"].iloc[::36000]]


def test_avaliar_ponta_a_ponta(monkeypatch):
    # treino/arquivos curtos para o teste ficar rapido
    monkeypatch.setattr(C, "HORAS_TREINO", 0.5)
    monkeypatch.setattr(C, "HORAS_CAL", 0.5)
    monkeypatch.setattr(C, "CAL", (3,))
    monkeypatch.setattr(C, "HORAS_NORMAL", 2)
    monkeypatch.setattr(C, "HORAS_FALHA", 2)
    raw, treinos = A.avaliar(["tests.test_benchmark:Nula", "tests.test_benchmark:Sempre"], ["nominal"], "dev",
                             n_falha=1, n_normal=1, processos=1, log=lambda *_: None)
    assert len(raw) == 2 * 13 and (raw["erro"] == "").all()
    fam = A.agregar(raw, ["familia", "arquitetura"]).set_index("arquitetura")
    assert fam.loc["nula", "deteccao"] == "0/12" and fam.loc["nula", "FA"] == 0
    assert fam.loc["sempre", "FA"] == 2 and fam.loc["sempre", "FA_semana"] == pytest.approx(168.0)


def test_incidentes_holdoff_e_ligacao_explicita():
    al = [alarme(0, None, None, modelo="cva", incidente=7), alarme(300, "pb1", "encoder", modelo="torque"),
          alarme(700, "pb1", "encoder"), alarme(5000, "pb1", "encoder", modelo="torque", incidente=7),
          alarme(9000, "pb2", "encoder"), alarme(9500, "pb2", "encoder"), alarme(9700, "pb2", "encoder")]
    incs = A.incidentes(al)
    assert [len(x) for x in incs] == [3, 1, 2, 1]          # 0+300+5000 (ligado), 700, 9000+9500, 9700
    assert A._isolacao(incs[0]) == ("pb1", "encoder")      # 1o alarme do incidente que aponta um rolo
    g = {"falha": None, "horas": 4.0}
    assert A.medir(al, g)["fa"] == 4 and A.medir(al, g)["n_alarmes"] == 7
