"""
Arquitetura candidata "autoencoder" (proposta da reuniao com a empresa: autoencoder como modelo
principal, Isolation Forest como referencia). Resultados de desenvolvimento e comparacao honesta com
a CVA em arquiteturas/aprendizado.md.

O que faz
    Atributos por bloco de 60 s em regime - os MESMOS da CVA em comparacao_metodos.py (classe Blocos):
        media do setpoint, das 3 velocidades, dos 3 torques e das 2 tensoes das pontas + aspereza
        (RMS da diferenca a 10 Hz) das 3 velocidades = 12 atributos; mesmo portao de regime do
        Modelo 2 (setpoint >= 200 m/min sem mudanca em 30 s, >= 80 % do bloco em regime).
    Janela temporal [F_t, F_t-1, ..., F_t-L] (L = 3, como o passado de 3 blocos da CVA), so dentro
    do mesmo trecho de regime; padronizada com a media e o desvio do treino.
    Autoencoder denso (sklearn MLPRegressor X -> X, tanh, camadas ocultas-gargalo-ocultas, L-BFGS),
    conjunto de `n_modelos` redes com sementes diferentes (media das reconstrucoes).
        gargalo k = menor valor da grade com R2 de VALIDACAO >= 0,95 (validacao = ultimos 20 % de cada
        arquivo de treino, sem embaralhar no tempo) - o analogo dos 95 % de variancia da PCA. Nenhum
        hiperparametro foi escolhido com cenarios de falha.
    Estatisticas por bloco: T2 do codigo latente (branqueado pela covariancia do treino) e distancia de
    Mahalanobis do residuo de reconstrucao (covariancia do residuo no treino). p-valor contra a
    calibracao (empirico deixa-um-de-fora + cauda exponencial alem dos 50 maiores), combinado pelo
    minimo (escala -log10) - o mesmo procedimento de comparacao_metodos.py (p_cauda, pvalor).
    Limite: reamostragem de semanas de calendario em blocos de 2 h da calibracao para `meta_fa_semana`
    alarmes falsos por semana (bissecao); intervalo minimo de 10 min entre alarmes.
    Rolo: contribuicao de cada atributo ao residuo (e_j^2 / var_j, somada nas defasagens), agrupada por
    rolo (velocidade, torque e aspereza do rolo); cada grupo vira p-valor contra a propria
    distribuicao na calibracao e o rolo e o de menor p-valor. Acerta o rolo em ~55-70 % dos alarmes
    no simulador (aprendizado.md); a CAUSA (encoder x mecanica) nao e separavel por essa contribuicao
    (em malha fechada a deriva do encoder tambem aparece no torque), entao causa = None.

Eventos do drive (log de avisos) nao sao usados: este modelo so olha os sinais.

    from arquiteturas.aprendizado import ArquiteturaAutoencoder
    a = ArquiteturaAutoencoder().treinar([df1, df2]).calibrar([df3, df4, df5])
    for e in a.analisar(df): print(e["t_s"], e["rolo"], e["severidade"])
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

try:
    from threadpoolctl import threadpool_limits       # redes pequenas: 1 thread e mais rapido
except Exception:                                      # pragma: no cover
    threadpool_limits = None

try:                                                   # interface comum (arquiteturas/base.py)
    from .base import Arquitetura as _Base
except Exception:                                      # pragma: no cover - base ausente
    class _Base:
        """Base minima: treinar -> calibrar -> analisar."""

ROLOS = ("festoon", "pb1", "pb2")
COLUNAS = ["v_setpoint"] + [f"v_{r}" for r in ROLOS] + [f"tq_{r}" for r in ROLOS] + ["tensao_entrada", "tensao_saida"]
NOMES_ATRIB = COLUNAS + [f"aspereza_v_{r}" for r in ROLOS]
GRUPO_ROLO = {r: [1 + i, 4 + i, 9 + i] for i, r in enumerate(ROLOS)}   # v_r, tq_r, aspereza_v_r
SEMANA_H = 168.0


# ------------------------------------------------------------------ blocos (= comparacao_metodos.Blocos)
def portao(sp, fs=10, janela_s=30.0, v_min=200.0):
    """Linha em regime: setpoint >= v_min e faixa < 1 m/min na janela (= MonitorTorqueV2.portao)."""
    s = pd.Series(np.asarray(sp, float))
    j = int(janela_s * fs)
    faixa = (s.rolling(j, min_periods=1).max() - s.rolling(j, min_periods=1).min()).to_numpy()
    ok = (s.to_numpy() >= v_min) & (faixa < 1.0)
    ok[:j] = False
    return ok


def blocos(df, fs=10, media_s=60):
    """(F [n, 12], t do fim do bloco, inicio de trecho) dos blocos com >= 80 % em regime."""
    nb = media_s * fs
    g = portao(df["v_setpoint"].to_numpy(float), fs)
    n = len(df) // nb
    gg = g[: n * nb].reshape(n, nb)
    cont = gg.sum(axis=1)
    sel = np.flatnonzero(cont >= 0.8 * nb)
    X = df[COLUNAS].to_numpy(float)[: n * nb].reshape(n, nb, len(COLUNAS))
    w = gg[..., None]
    med = (X * w).sum(axis=1) / np.maximum(cont, 1)[:, None]
    dv = np.diff(X[..., 1:4], axis=1)
    asp = np.sqrt((dv ** 2 * w[:, 1:]).sum(axis=1) / np.maximum(cont, 2)[:, None])
    F = np.hstack([med, asp])[sel]
    inicio = np.ones(len(sel), bool)
    inicio[1:] = np.diff(sel) > 1
    t = df["t_s"].to_numpy(float)[np.minimum((sel + 1) * nb - 1, len(df) - 1)]
    return F, t, inicio


def defasar(F, inicio, L):
    """[F_t, F_t-1, ..., F_t-L] so dentro do mesmo trecho; devolve (X, valido)."""
    if not L:
        return F, np.ones(len(F), bool)
    seg = np.cumsum(inicio)
    X, ok = [F], np.ones(len(F), bool)
    for i in range(1, L + 1):
        X.append(np.vstack([np.zeros((i, F.shape[1])), F[:-i]]))
        ok[i:] &= seg[i:] == seg[:-i]
    ok[:L] = False
    return np.hstack(X), ok


# ------------------------------------------------------------------ p-valor e limite (= comparacao_metodos)
def p_cauda(r, x, dentro=False, m=50):
    """p-valor da cauda superior contra a referencia ORDENADA r; acima do m-esimo maior, cauda
    exponencial (POT). dentro=True: x pertence a r (deixa-um-de-fora)."""
    x = np.asarray(x, float)
    k = len(r) - np.searchsorted(r, x, side="left")
    p = np.maximum(k, 1) / len(r) if dentro else (k + 1) / (len(r) + 1)
    n = len(r)
    if n <= m:
        return p
    u = r[n - m - 1]
    beta = max(float(np.mean(r[n - m:] - u)), 1e-12)
    return np.where(x > u, np.maximum((m / n) * np.exp(-(x - u) / beta), 1e-300), p)


def menos_log_p(ref_ord, S, dentro=False):
    """-log10 do menor p-valor entre as colunas de S (ref_ord: lista de colunas ordenadas)."""
    out = np.zeros(len(S))
    for c, r in enumerate(ref_ord):
        out = np.maximum(out, -np.log10(p_cauda(r, S[:, c], dentro)))
    return out


def alarmes(s, L, holdoff):
    acima = np.flatnonzero(s > L)
    out, prox = [], -1
    for i in acima:
        if i >= prox:
            out.append(int(i))
            prox = i + holdoff
    return out


def limite_boot(series, n_sem, meta, holdoff, bloco=120, semanas=40, seed=0, lo=1e-3, hi=1e4):
    """L com taxa de alarmes = meta por semana na reamostragem de semanas em blocos de `bloco`."""
    rng = np.random.default_rng(seed)
    S = np.concatenate(series)
    nb = int(np.ceil(n_sem / bloco))
    sem = [np.concatenate([S[i:i + bloco] for i in rng.integers(0, max(1, len(S) - bloco), nb)])[:n_sem]
           for _ in range(semanas)]

    def taxa(L, parar=True):
        tot = 0
        for w in sem:
            tot += len(alarmes(w, L, holdoff))
            if parar and tot > meta * semanas:
                break
        return tot / semanas

    while hi / lo > 1.01:
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if taxa(mid) > meta else (lo, mid)
    return float(hi), taxa(hi, parar=False)


def _isqrt(M, reg=1e-6):
    M = M + reg * np.eye(len(M))
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 1e-12 * w.max())
    return V @ np.diag(1 / np.sqrt(w)) @ V.T


# ------------------------------------------------------------------ arquitetura
class ArquiteturaAutoencoder(_Base):
    nome = "autoencoder"

    def __init__(self, defasagens=3, ocultas=(32,), grade_k=(1, 2, 3, 4, 6, 8, 12), r2_alvo=0.95,
                 alpha=1e-3, max_iter=400, n_modelos=3, seed=0, frac_val=0.2, fs=10, media_s=60,
                 holdoff_min=10.0, bloco_boot_h=2.0, semanas_boot=40):
        self.L, self.ocultas, self.grade_k, self.r2_alvo = int(defasagens), tuple(ocultas), tuple(grade_k), r2_alvo
        self.alpha, self.max_iter, self.n_modelos, self.seed, self.frac_val = alpha, max_iter, n_modelos, seed, frac_val
        self.fs, self.media_s = fs, media_s
        self.holdoff = max(1, int(round(holdoff_min * 60 / media_s)))
        self.bloco_boot = max(1, int(bloco_boot_h * 3600 / media_s))
        self.semanas_boot = semanas_boot
        self.k = None
        self.limite = None

    # ---------------------------------------------------------- rede
    def _rede(self, k, seed):
        return MLPRegressor(hidden_layer_sizes=self.ocultas + (k,) + self.ocultas[::-1], activation="tanh",
                            solver="lbfgs", alpha=self.alpha, max_iter=self.max_iter, tol=1e-7, random_state=seed)

    def _ajustar(self, k, Z):
        def fit():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                return [self._rede(k, self.seed + j).fit(Z, Z) for j in range(self.n_modelos)]
        if threadpool_limits is None:
            return fit()
        with threadpool_limits(1):
            return fit()

    @staticmethod
    def _rec(redes, Z):
        return np.mean([r.predict(Z) for r in redes], axis=0)

    def _codigo(self, Z):
        """Codigo latente de cada rede (concatenado)."""
        out = []
        nl = len(self.ocultas)
        for r in self.redes:
            h = Z
            for W, b in zip(r.coefs_[: nl + 1], r.intercepts_[: nl + 1]):
                h = np.tanh(h @ W + b)
            out.append(h)
        return np.hstack(out)

    # ---------------------------------------------------------- sinais
    def _janelas(self, df):
        F, t, inicio = blocos(df, self.fs, self.media_s)
        X, ok = defasar(F, inicio, self.L)
        return X, ok, t

    def _estat(self, X):
        Z = (X - self.m) / self.s
        E = Z - self._rec(self.redes, Z)
        t2 = (((self._codigo(Z) - self.mc) @ self.Wc) ** 2).sum(1)
        q = (((E - self.mu_e) @ self.We) ** 2).sum(1)
        c = ((E - self.mu_e) ** 2 / self.var_e).reshape(len(E), -1, len(NOMES_ATRIB)).sum(1)
        return np.column_stack([t2, q]), c

    # ---------------------------------------------------------- interface
    def treinar(self, normais, eventos_normais=None, duracao_eventos_s=None):
        tr, va = [], []
        for df in normais:
            X, ok, _ = self._janelas(df)
            X = X[ok]
            n = int(len(X) * (1 - self.frac_val))
            tr.append(X[:n]); va.append(X[n:])
        Xtr, Xva = np.vstack(tr), np.vstack(va)
        if len(Xtr) < 50 or len(Xva) < 10:
            raise ValueError("poucos blocos de 60 s em regime para treinar o autoencoder")
        X = np.vstack([Xtr, Xva])
        self.m, self.s = X.mean(0), X.std(0) + 1e-9
        Ztr, Zva = (Xtr - self.m) / self.s, (Xva - self.m) / self.s
        self.selecao = {}
        for k in self.grade_k:                     # menor gargalo com R2 de validacao >= alvo
            r = self._ajustar(k, Ztr)
            self.selecao[k] = float(1 - ((Zva - self._rec(r, Zva)) ** 2).sum() / ((Zva - Zva.mean(0)) ** 2).sum())
            if self.selecao[k] >= self.r2_alvo:
                break
        self.k = k
        Z = np.vstack([Ztr, Zva])
        self.redes = self._ajustar(self.k, Z)
        E = Z - self._rec(self.redes, Z)
        self.mu_e, self.var_e = E.mean(0), E.var(0) + 1e-9
        self.We = _isqrt(np.cov(E.T))
        C = self._codigo(Z)
        self.mc = C.mean(0)
        self.Wc = _isqrt(np.atleast_2d(np.cov(C.T)))
        return self

    def calibrar(self, normais, meta_fa_semana=1.0):
        if self.k is None:
            raise RuntimeError("treinar antes de calibrar")
        S, Cs, oks, n_amostras = [], [], [], 0
        for df in normais:
            X, ok, _ = self._janelas(df)
            s, c = self._estat(X)
            S.append(s); Cs.append(c); oks.append(ok)
            n_amostras += len(df)
        ref = np.vstack([s[ok] for s, ok in zip(S, oks)])
        self.ref = [np.sort(ref[:, j]) for j in range(ref.shape[1])]
        Cc = np.vstack([c[ok] for c, ok in zip(Cs, oks)])
        self.ref_rolo = {r: np.sort(Cc[:, g].sum(1)) for r, g in GRUPO_ROLO.items()}
        series = [np.where(ok, menos_log_p(self.ref, s, dentro=True), 0.0) for s, ok in zip(S, oks)]
        n_blocos = sum(len(s) for s in S)
        horas = n_amostras / self.fs / 3600
        n_sem = int(n_blocos * SEMANA_H / horas)
        self.meta = meta_fa_semana
        self.limite, self.fa_boot_semana = limite_boot(series, n_sem, meta_fa_semana, self.holdoff,
                                                       self.bloco_boot, self.semanas_boot)
        return self

    def indicador(self, df):
        """(t_s, -log10 p combinado, contribuicoes por atributo [n, 12]) por bloco de 60 s em regime."""
        X, ok, t = self._janelas(df)
        s, c = self._estat(X)
        return t, np.where(ok, menos_log_p(self.ref, s), 0.0), c

    def analisar(self, df, eventos=None):
        if self.limite is None:
            raise RuntimeError("calibrar antes de analisar")
        t, s, c = self.indicador(df)
        out = []
        for i in alarmes(s, self.limite, self.holdoff):
            sc = {r: float(-np.log10(p_cauda(self.ref_rolo[r], np.array([c[i, g].sum()]))[0]))
                  for r, g in GRUPO_ROLO.items()}
            rolo = max(sc, key=sc.get)
            margem = s[i] - self.limite
            sev = "baixa" if margem < 0.5 else ("media" if margem < 2.0 else "alta")
            top = int(np.argmax(c[i]))
            out.append({"t_s": float(t[i]), "rolo": rolo, "causa": None, "severidade": sev,
                        "modelo": self.nome, "menos_log10_p": float(s[i]), "limite": self.limite,
                        "rolo_menos_log10_p": sc, "atributo_maior": NOMES_ATRIB[top],
                        "mensagem": f"autoencoder: padrao dos sinais fora do normal (-log10 p = {s[i]:.1f} > "
                                    f"{self.limite:.1f}); maior contribuicao em {NOMES_ATRIB[top]}, rolo "
                                    f"suspeito {rolo}"})
        return out
