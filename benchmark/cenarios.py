"""
Cenarios do benchmark: familias de robustez sobre o simulador v2 (benchmark/sim_ext.py).

Cada Cenario e deterministico dado (familia, variante, caso, seed). gerar(c) devolve o arquivo
que a arquitetura ve (sem colunas de verdade), o log de avisos (com 14 dias de historico normal da
mesma planta antes do arquivo) e a verdade de campo (inicio da falha, rolo, causa esperada).

PLANTAS: planta de cada familia. O treino (mf2 1-2, 12 h) e a calibracao (mf2 3-8, 48 h) sao
gerados NA PLANTA da familia, com os parametros nominais; o que muda "depois do treino" fica so
nos cenarios (familias planta_muda_apos_treino, troca_receita, arquivo_quente, avisos).

SEMENTES: 'dev' = 500-599 (falhas 500-549, normais 550-599); 'final' = 800-999 (falhas 800-899,
normais 900-999) - reservada ao orquestrador, que roda 'final' uma vez (ja usada); 'confirmacao' =
1000-1199 (falhas 1000-1099, normais 1100-1199), mesmo tamanho da 'final' - reservada ao orquestrador para
a rodada de CONFIRMACAO da arquitetura corrigida (nunca usar em desenvolvimento). A repeticao j de todos
os casos de uma familia usa a mesma semente (comparacao pareada entre casos e variantes).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from poc.simulador_avisos import ENCODERS_LINHA

from .sim_ext import COLUNAS_VERDADE, perfil, simular_acoplado_ext, simular_ext, tempo_troca

SEMENTES = {"dev": {"falha": range(500, 550), "normal": range(550, 600)},
            "final": {"falha": range(800, 900), "normal": range(900, 1000)},
            "confirmacao": {"falha": range(1000, 1100), "normal": range(1100, 1200)}}
RESERVADAS = ("final", "confirmacao")     # so o orquestrador, com --confirmar-<nome> (benchmark/avaliar.py)
TREINO, CAL = (1, 2), tuple(range(3, 9))
CAL_LONGA = tuple(range(3, 21))           # opcao: 18 x 48 h de calibracao (sementes 9-20 nunca usadas)
HORAS_TREINO, HORAS_CAL, HORAS_NORMAL, HORAS_FALHA = 12, 48, 48, 6
DIAS_HIST = 14.0
SO_AVISOS = ("erros_com", "degradacao_sinal")

CASOS = [("deriva_ganho", "festoon"), ("deriva_ganho", "pb1"), ("deriva_ganho", "pb2"),
         ("atrito", "festoon"), ("atrito", "pb1"), ("atrito", "pb2"),
         ("perda_pulsos", "pb1"), ("perda_pulsos", "pb2"),
         ("ruido", "pb1"), ("ruido", "pb2"),
         ("erros_com", "pb1"), ("degradacao_sinal", "festoon")]       # = validacao_conjunta.CASOS
CASOS_SINAIS = CASOS[:10]
CASOS_DERIVA_ATRITO = [("deriva_ganho", "festoon"), ("deriva_ganho", "pb1"), ("deriva_ganho", "pb2"),
                       ("atrito", "pb1")]

PLANTAS = {
    "padrao": {},
    "tensao_pb2": {"modo_tensao": ("pb2",)},
    # bracos efetivos r/i = 0,10 / 0,15 / 0,25 m (o simulador original usa 0,15 nos tres)
    "relacoes": {"raio": {"festoon": 0.125, "pb1": 0.15, "pb2": 0.20},
                 "relacao": {"festoon": 1.25, "pb1": 1.0, "pb2": 0.8}},
    "ruido_hf": {"ruido_hf": 1.0e-5},       # jitter de banda larga visivel na velocidade gravada (m/s)
}

# receitas (fatores sobre a receita 0 = a do treino)
RECEITAS = {"draw": {"draw": 1.6}, "tensao": {"tensao": 1.15}, "vel_baixa": {"vel": 0.7},
            "vel_alta": {"vel": 1.2}, "completa": {"draw": 1.6, "tensao": 1.15, "vel": 0.8}}


@dataclass
class Cenario:
    familia: str
    variante: str
    falha: "str | None"
    rolo: str
    seed: int
    horas: float
    planta: str = "padrao"
    sim: dict = field(default_factory=dict)          # kwargs extras de simular_ext
    avisos: dict = field(default_factory=dict)       # kwargs extras de simular_acoplado_ext
    inicio_h: float = 1.0
    trocas: tuple = ()                               # ((t_alvo_h, id_receita, fatores), ...)

    @property
    def caso(self):
        return "normal" if self.falha is None else f"{self.falha} {self.rolo}"

    @property
    def chave(self):
        return (self.familia, self.variante, self.caso, self.seed)


def causa_esperada(falha):
    return "mecanica" if falha == "atrito" else "encoder"


def _sem_verdade(df):
    return df.drop(columns=[c for c in COLUNAS_VERDADE if c in df.columns])


def gerar(c: Cenario):
    """-> dict(df, eventos, t_on, t_lim, falha, rolo, causa, horas)."""
    sim = {**PLANTAS[c.planta], **c.sim}
    if c.trocas:
        sp = perfil(c.horas + sim.get("pre_h", 0.0), c.seed)
        sim["receitas"] = tuple((tempo_troca(sp, t_h * 3600), ident, fat) for t_h, ident, fat in c.trocas)
    falha_sim = None if c.falha in SO_AVISOS else c.falha
    df = simular_ext(c.horas, c.seed, falha=falha_sim, rolo=c.rolo, inicio_h=c.inicio_h, **sim)
    ev, cont, info = simular_acoplado_ext(c.seed, df["t_s"].to_numpy(), df["severidade"].to_numpy(), c.falha,
                                          c.rolo, inicio_s=c.inicio_h * 3600, dias_hist=DIAS_HIST, **c.avisos)
    t0 = info["t0_s"]
    df = _sem_verdade(df)
    df["t_s"] = df["t_s"] + t0
    ev.attrs.update(contadores=cont, encoders=dict(ENCODERS_LINHA), t_ini_s=t0, t_fim_s=info["t_fim_s"])
    return {"df": df, "eventos": ev, "falha": c.falha, "rolo": c.rolo, "causa": causa_esperada(c.falha),
            "t_on": None if c.falha is None else t0 + c.inicio_h * 3600,
            "t_lim": info["t_falha_s"] if info["t_falha_s"] is not None else np.inf,
            "horas": len(df) / 10 / 3600, "t0": t0}


def dados_treino(planta="padrao", cal="padrao"):
    """(treino, calibracao, eventos_normais, duracao_s) da planta, sem colunas de verdade."""
    p = PLANTAS[planta]
    treino = [_sem_verdade(simular_ext(HORAS_TREINO, s, **p)) for s in TREINO]
    cal_s = CAL_LONGA if cal == "longa" else CAL
    calib = [_sem_verdade(simular_ext(HORAS_CAL, s, **p)) for s in cal_s]
    return treino, calib, *eventos_treino()


def eventos_treino(seed=TREINO[0]):
    t = np.arange(0, HORAS_TREINO * 3600, 0.1)
    ev, cont, info = simular_acoplado_ext(seed, t, np.zeros(len(t)), None, "pb1", dias_hist=DIAS_HIST)
    t0 = info["t0_s"]
    ev = ev[ev["t_s"] < t0].reset_index(drop=True)
    ev.attrs.update(contadores=cont[cont["t_s"] < t0].reset_index(drop=True), encoders=dict(ENCODERS_LINHA),
                    t_ini_s=0.0, t_fim_s=t0)
    return ev, t0


# ------------------------------------------------------------------ familias
def _bloco(familia, variante, casos, n_falha, n_normal, sementes, horas_falha=None, **kw):
    out = []
    horas_falha = HORAS_FALHA if horas_falha is None else horas_falha
    for falha, rolo in casos:
        out += [Cenario(familia, variante, falha, rolo, s, horas_falha, **kw)
                for s in list(sementes["falha"])[:n_falha]]
    out += [Cenario(familia, variante, None, "pb1", s, HORAS_NORMAL, **kw)
            for s in list(sementes["normal"])[:n_normal]]
    return out


def f_nominal(nf, nn, sem):
    return _bloco("nominal", "base", CASOS, nf, nn, sem)


def f_planta_muda(nf, nn, sem):
    var = {"ea-2%": {"ea_fator": 0.98}, "ea-5%": {"ea_fator": 0.95},
           "draw+3%": {"draw_fator": 1.03}, "draw+5%": {"draw_fator": 1.05},
           "atrito+10%": {"atrito_fator": 1.10}, "atrito-10%": {"atrito_fator": 0.90}}
    return [c for v, sim in var.items() for c in _bloco("planta_muda_apos_treino", v, CASOS_DERIVA_ATRITO,
                                                         nf, nn, sem, sim=sim)]


def f_troca_receita(nf, nn, sem):
    out = []
    for v, fat in RECEITAS.items():
        # normais: receita 1 a partir da 1a parada depois de 12 h, volta a 0 em 24 h, 1 de novo em 36 h
        trocas_n = ((12.0, 1, fat), (24.0, 0, {}), (36.0, 1, fat))
        out += _bloco("troca_receita", v, [], 0, nn, sem, trocas=trocas_n)
        # falhas depois da troca: troca na 1a parada depois de 1 h, falha comeca em 3 h (arquivo de 8 h)
        out += _bloco("troca_receita", v, [("deriva_ganho", "pb1"), ("atrito", "pb2")], nf, 0, sem,
                      horas_falha=8, inicio_h=3.0, trocas=((1.0, 1, fat),))
    return out


def f_arquivo_quente(nf, nn, sem):
    casos = CASOS_DERIVA_ATRITO[:3] + [("atrito", "festoon"), ("atrito", "pb1"), ("atrito", "pb2"),
                                       ("perda_pulsos", "pb1")]
    return _bloco("arquivo_quente", "pre_3h", casos, nf, nn, sem, sim={"pre_h": 3.0})


def f_ruido_alta_freq(nf, nn, sem):
    casos = [("deriva_ganho", "pb1"), ("atrito", "pb1"), ("perda_pulsos", "pb1"), ("perda_pulsos", "pb2"),
             ("ruido", "pb1"), ("ruido", "pb2")]
    out = _bloco("ruido_alta_freq", "falha_50%_hf", [("ruido", "pb1"), ("ruido", "pb2")], nf, 0, sem,
                 sim={"frac_hf_ruido": 0.5})
    out += _bloco("ruido_alta_freq", "jitter_normal+falha_50%_hf", casos, nf, nn, sem, planta="ruido_hf",
                  sim={"frac_hf_ruido": 0.5})
    return out


def f_avisos(nf, nn, sem):
    casos = [("perda_pulsos", "pb1"), ("perda_pulsos", "pb2"), ("ruido", "pb1"), ("ruido", "pb2")]
    out = []
    for taxa in (2.0, 0.5, 0.0):
        out += _bloco("avisos", f"taxa_acoplada_{taxa:g}/h", casos, nf, 0, sem, avisos={"taxa_acoplada_h": taxa})
    out += _bloco("avisos", "aglomerados_normais", [], 0, nn, sem,
                  avisos={"aglomerado": {"media": 3.0, "intervalo_s": 900.0}})
    out += _bloco("avisos", "fundo_x2_apos_treino", [], 0, nn, sem, avisos={"fator_pos": 2.0})
    return out


def f_malha_tensao(nf, nn, sem):
    return _bloco("malha_tensao_pb2", "trim_pi_pb2", CASOS_SINAIS, nf, nn, sem, planta="tensao_pb2")


def f_relacoes(nf, nn, sem):
    return _bloco("relacoes_diferentes", "r_ef_0.10/0.15/0.25", CASOS_SINAIS, nf, nn, sem, planta="relacoes")


FAMILIAS = {"nominal": f_nominal, "planta_muda_apos_treino": f_planta_muda, "troca_receita": f_troca_receita,
            "arquivo_quente": f_arquivo_quente, "ruido_alta_freq": f_ruido_alta_freq, "avisos": f_avisos,
            "malha_tensao_pb2": f_malha_tensao, "relacoes_diferentes": f_relacoes}

# tamanho por familia: (repeticoes por caso de falha, arquivos normais de 48 h por variante)
TAMANHO = {"dev": {f: (3, 2) for f in FAMILIAS} | {"nominal": (10, 5)},
           "final": {f: (10, 10) for f in FAMILIAS} | {"nominal": (10, 20)}}
TAMANHO["confirmacao"] = dict(TAMANHO["final"])       # mesmo tamanho da final


def montar(familias, seeds="dev", n_falha=None, n_normal=None):
    if seeds not in SEMENTES:
        raise ValueError(f"conjunto de sementes desconhecido: {seeds}")
    sem = SEMENTES[seeds]
    out = []
    for f in familias:
        nf, nn = TAMANHO[seeds][f]
        nf = nf if n_falha is None else n_falha
        nn = nn if n_normal is None else n_normal
        if nf > len(sem["falha"]) or nn > len(sem["normal"]):
            raise ValueError("repeticoes demais para o conjunto de sementes")
        out += FAMILIAS[f](nf, nn, sem)
    return out


__all__ = ["Cenario", "gerar", "dados_treino", "montar", "FAMILIAS", "PLANTAS", "SEMENTES", "RESERVADAS", "pd"]
