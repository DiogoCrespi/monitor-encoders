"""
Modelo 3 - "prever o proximo passo", versao 2.

Papel no conjunto: picos, ruido e intermitencia do PROPRIO sinal do encoder.
NAO e indicador de deriva lenta (o preditor acompanha a degradacao; para isso
existem os indicadores 'vizinhos' e 'torque' de monitor.py).

Diferencas para o indicador 'proprio' de monitor.py (AR(5) + EWMA|erro| + 1,1 x maximo):

1. ARX: a previsao usa o passado do sinal E o setpoint (valor atual e derivada,
   com atrasos). Rampas e emendas deixam de gerar erro de previsao grande.
       y^(t) = a0 + sum_{k=1..p} a_k y(t-k) + b0 sp(t) + sum_{j=0..q-1} c_j dsp(t-j)
   Ordem (p, q) escolhida por BIC no treino. O residuo passa por um AR(r) de
   branqueamento (r por BIC, pode ser 0) e e padronizado:  z(t) = eps(t) / sigma.
2. Portao de regime: so monitora com a linha rodando e o setpoint parado ha
   'portao_s' segundos (suspende em parada, rampa e emenda). As estatisticas sao
   calculadas so sobre as amostras dentro do portao (o estado "congela" fora dele).
   O branqueamento roda na serie continua (antes de comprimir no portao): os atrasos do AR
   nunca ligam o fim de um trecho monitorado ao inicio do seguinte. Varios trechos de treino
   entram por treinar_trechos(), com o portao fechado em cada juncao.
3. Estatisticas de alarme sobre z (todas disponiveis, ver ESTATISTICAS):
       ewma_abs   EWMA_lam(|z|)                         (a do indicador atual)
       ewma_sq    EWMA_lam(z^2)                         (variancia -> ruido)
       cusum_sq   C_t = max(0, C_{t-1} + z_t^2 - 1 - k) (CUSUM de escala)
       var_jan    media de z^2 em janela de W amostras
       kurt_jan   curtose de z em janela de W amostras  (rajadas)
       contagem   n. de amostras com |z| > k_pico em janela de W amostras (picos)
4. Limite por META de falso alarme em tempo (ex.: 1 por semana de operacao
   monitorada). Estatisticas continuas: reamostragem em blocos de 10 min (moving
   block bootstrap) dos residuos normais de calibracao, reproduzindo a regra
   completa (estatistica -> persistencia -> intervalo minimo entre alarmes); a cauda
   dos picos e extrapolada por uma exponencial ate a taxa-alvo (limiar_bootstrap).
   Contagem de picos: modelo de Poisson com a taxa de |z| > k_pico da calibracao
   (limiar_contagem). Com varias estatisticas, a meta e dividida entre elas.

Configuracao escolhida no desenvolvimento (validacao_proximo_passo_v2.py):
   var_jan(W = 300 s)  OU  contagem(|z| > 5 em 10 min), portao de 20 s, meta 1 FA/semana.

So usa numpy/scipy; aplicar o modelo exige somas, multiplicacoes e comparacoes.
"""
import json

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import lfilter

from . import V_RODANDO

SEMANA_S = 7 * 24 * 3600.0
ORDENS_P = (1, 2, 3, 5, 8, 12, 16, 20, 25, 30)   # ordens testadas pelo BIC


# ---------------------------------------------------------------- utilitarios
def atraso(x, k):
    if k == 0:
        return np.asarray(x, float)
    return np.concatenate([np.full(k, x[0]), x[:-k]])


def ewma(x, lam):
    return lfilter([lam], [1, -(1 - lam)], x)


def _janela_soma(x, W):
    """Soma movel causal das ultimas W amostras (as primeiras usam o que houver)."""
    c = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
    i = np.arange(1, len(x) + 1)
    return c[i] - c[np.maximum(i - W, 0)]


def portao(sp, rodando, fs, portao_s=20.0, tol=1e-6):
    """True quando a linha roda e o setpoint nao mudou nos ultimos portao_s segundos."""
    rodando = np.asarray(rodando, bool)
    if sp is None or not portao_s:
        return rodando.copy()
    sp = np.asarray(sp, float)
    G = max(1, int(portao_s * fs))
    mudou = (np.abs(np.diff(sp, prepend=sp[0])) > tol).astype(float)
    return rodando & (_janela_soma(mudou, G) == 0)


# ---------------------------------------------------------------- ARX + branqueamento
def matriz_arx(y, sp, p, q, fs):
    cols = [atraso(y, k) for k in range(1, p + 1)]
    if sp is not None and q > 0:
        dsp = np.diff(sp, prepend=sp[0]) * fs
        cols += [np.asarray(sp, float)] + [atraso(dsp, j) for j in range(q)]
    if not cols:
        return np.ones((len(y), 0))
    return np.column_stack(cols)


def _mq(X, y, alpha=1e-6):
    A = np.column_stack([X, np.ones(len(X))])
    return np.linalg.solve(A.T @ A + alpha * len(X) * np.eye(A.shape[1]), A.T @ y)


def _prever(coef, X):
    return X @ coef[:-1] + coef[-1]


def _bic(res, n_par):
    n = len(res)
    return n * np.log(np.mean(res ** 2) + 1e-300) + n_par * np.log(n)


def ajustar_ar(e, r_max=10, mask=None):
    """AR(r) do residuo para branquear; r por BIC (0 = ja esta branco)."""
    e = np.asarray(e, float)
    m = np.ones(len(e), bool) if mask is None else mask
    melhor = (_bic(e[m], 0), np.zeros(0))
    for r in range(1, r_max + 1):
        X = np.column_stack([atraso(e, k) for k in range(1, r + 1)])
        phi = np.linalg.lstsq(X[m], e[m], rcond=None)[0]
        b = _bic(e[m] - X[m] @ phi, r)
        if b < melhor[0]:
            melhor = (b, phi)
    return melhor[1]


def branquear(e, phi):
    return lfilter(np.concatenate([[1.0], -np.asarray(phi)]), [1.0], e) if len(phi) else np.asarray(e, float)


def juntar_trechos(series, n_quebra):
    """Concatena varios (y, sp, rodando) num so, com 'rodando' = False nas n_quebra primeiras
    amostras de cada trecho depois do primeiro: o portao fica fechado na juncao e nenhum atraso
    (ARX ou branqueamento) mistura o fim de um trecho com o inicio do outro."""
    ys, sps, rods = [], [], []
    for i, (y, sp, rod) in enumerate(series):
        rod = np.array(rod, bool)
        if i:
            rod[:n_quebra] = False
        ys.append(np.asarray(y, float))
        sps.append(sp)
        rods.append(rod)
    sp = None if any(s is None for s in sps) else np.concatenate([np.asarray(s, float) for s in sps])
    return np.concatenate(ys), sp, np.concatenate(rods)


def ljung_box(z, lags=20):
    """Estatistica Q de Ljung-Box e p-valor (qui-quadrado com 'lags' g.l.)."""
    from scipy.stats import chi2
    z = np.asarray(z, float) - np.mean(z)
    n = len(z)
    d = (z ** 2).sum()
    rho = np.array([(z[:-k] * z[k:]).sum() / d for k in range(1, lags + 1)])
    Q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
    return float(Q), float(chi2.sf(Q, lags)), rho


# ---------------------------------------------------------------- estatisticas
def est_ewma_abs(z, lam=0.05, **_):
    return ewma(np.abs(z), lam)


def est_ewma_sq(z, lam=0.05, **_):
    return ewma(z ** 2, lam)


def est_cusum_sq(z, k=0.5, **_):
    s = np.cumsum(z ** 2 - 1.0 - k)
    return s - np.minimum(np.minimum.accumulate(s), 0.0)


def _aquecer(s, W):
    """Janela ainda incompleta no inicio da serie: sem estatistica (0 = nao alarma)."""
    s[:min(W - 1, len(s))] = 0.0
    return s


def est_var_jan(z, W=600, **_):
    return _aquecer(_janela_soma(z ** 2, W) / W, W)


def est_kurt_jan(z, W=600, **_):
    n = float(W)
    m1 = _janela_soma(z, W) / n
    m2 = _janela_soma(z ** 2, W) / n
    m3 = _janela_soma(z ** 3, W) / n
    m4 = _janela_soma(z ** 4, W) / n
    var = np.maximum(m2 - m1 ** 2, 1e-12)
    c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    return _aquecer(c4 / var ** 2, W)


def est_contagem(z, W=6000, k_pico=5.0, **_):
    return _janela_soma((np.abs(z) > k_pico).astype(float), W)


ESTATISTICAS = {
    "ewma_abs": est_ewma_abs,
    "ewma_sq": est_ewma_sq,
    "cusum_sq": est_cusum_sq,
    "var_jan": est_var_jan,
    "kurt_jan": est_kurt_jan,
    "contagem": est_contagem,
}


# ---------------------------------------------------------------- regra de alarme
def persistente(s, persist):
    """Valor que precisa ser ultrapassado para ficar 'persist' amostras seguidas acima."""
    if persist <= 1:
        return s
    return minimum_filter1d(s, persist, origin=(persist - 1) // 2, mode="nearest")


def eventos(flag, persist, holdoff, idx=None):
    """Indices dos alarmes: flag por 'persist' amostras seguidas; depois, 'holdoff' amostras de silencio.
    idx mapeia a sequencia comprimida (so amostras no portao) para o indice original."""
    flag = np.asarray(flag, bool)
    cheio = np.convolve(flag.astype(int), np.ones(persist, int), "full")[:len(flag)] >= persist
    pos = np.flatnonzero(cheio)
    if idx is not None:
        pos = np.asarray(idx)[pos]
    out, ultimo = [], -np.inf
    for i in pos:
        if i - ultimo >= holdoff:
            out.append(int(i))
            ultimo = i
    return out


def cauda_exponencial(picos, p_alvo, frac=0.02, m_min=10):
    """Valor x com P(pico > x) = p_alvo, extrapolando a cauda dos picos por uma exponencial
    (picos acima de limiar, GPD com forma 0). Sem extrapolacao se p_alvo couber nos dados."""
    M = np.sort(np.asarray(picos, float))[::-1]
    nb = len(M)
    m = min(max(m_min, int(frac * nb)), nb - 1)
    if p_alvo * nb >= m:
        return float(M[int(p_alvo * nb)])
    u = M[m]
    beta = max(float(np.mean(M[:m] - u)), 1e-12)
    return float(u + beta * np.log(m / (nb * p_alvo)))


def _picos(s, H):
    """Maximo da estatistica (ja com persistencia) em janelas do tamanho do intervalo entre alarmes:
    cada janela gera no maximo um alarme (agrupamento de excursoes)."""
    nb = len(s) // H
    return s[:nb * H].reshape(nb, H).max(1)


def limiar_bootstrap(z_fontes, estatistica, params, persist, holdoff, fs, fa_semana=1.0,
                     n_rep=20, bloco_s=600.0, conf=0.5, seed=0):
    """Limite para ~fa_semana falsos alarmes por semana de operacao monitorada.

    z_fontes: vetores z NORMAIS (ja restritos ao portao), separados do treino.
    Reamostragem em blocos (moving block bootstrap, blocos de bloco_s segundos, que preservam a
    autocorrelacao): cada replica tem o mesmo tamanho dos dados de calibracao. Em cada replica
    reproduz-se a regra inteira - estatistica, persistencia, agrupamento pelo intervalo minimo
    entre alarmes - e extrapola-se a cauda dos picos ate a taxa-alvo (cauda_exponencial).
    Devolve o quantil 'conf' dos limites das replicas (0,5 = taxa-alvo em media; 0,9 = taxa
    garantida com ~90% de confianca, no espirito de Gandy & Kvaloy 2013) e a lista de limites.
    """
    rng = np.random.default_rng(seed)
    L = int(bloco_s * fs)
    H = max(int(holdoff), 1)
    blocos = [z[i:i + L] for z in z_fontes for i in range(0, len(z) - L + 1, max(1, L // 4))]
    if not blocos:
        raise ValueError("dados de calibracao mais curtos que um bloco")
    n_total = sum(len(z) for z in z_fontes)
    p_alvo = fa_semana * H / (SEMANA_S * fs)
    fn = ESTATISTICAS[estatistica]
    lims = []
    for _ in range(n_rep):
        esc = rng.integers(0, len(blocos), int(np.ceil(n_total / L)))
        z = np.concatenate([blocos[j] for j in esc])[:n_total]
        lims.append(cauda_exponencial(_picos(persistente(fn(z, **params), persist), H), p_alvo))
    return float(np.quantile(lims, conf)), lims


def limiar_contagem(z_fontes, W, k_pico, fs, fa_semana=1.0):
    """Contagem de picos: limite inteiro c (alarma se contagem > c, i.e. >= c+1 picos em W amostras).
    Picos raros ~ Poisson: taxa de alarmes por amostra ~ p * P(Poisson(W p) >= c), com
    p = (n. de |z| > k_pico na calibracao + 1) / n. de amostras (o +1 evita p = 0)."""
    from scipy.stats import poisson
    n = sum(len(z) for z in z_fontes)
    p = (sum(int((np.abs(z) > k_pico).sum()) for z in z_fontes) + 1) / n
    n_sem = SEMANA_S * fs
    c = 0
    while n_sem * p * poisson.sf(c - 1, W * p) > fa_semana and c < W:
        c += 1
    return float(c), p


# ---------------------------------------------------------------- o detector
class ProximoPassoV2:
    """Preditor ARX do proprio sinal + branqueamento + estatistica(s) com limite por meta de FA.

    estatisticas: dict nome -> params, ex. {"ewma_sq": {"lam": 0.05}, "contagem": {"W": 6000, "k_pico": 5}}
    O alarme dispara quando QUALQUER estatistica passa do seu limite (a meta de FA e dividida
    igualmente entre elas).
    """

    PERSIST_PADRAO = {"contagem": 1}

    def __init__(self, fs=10, p_max=30, q_max=3, r_max=30, ordem=None, portao_s=20.0,
                 estatisticas=None, persist_s=2.0, holdoff_min=10.0, fa_semana=1.0,
                 n_rep=20, conf=0.5, treinar_em="rodando"):
        self.fs, self.p_max, self.q_max, self.r_max = fs, p_max, q_max, r_max
        self.ordem, self.portao_s = ordem, portao_s
        self.estatisticas = estatisticas or {"var_jan": {"W": int(300 * fs)},
                                             "contagem": {"W": int(600 * fs), "k_pico": 5.0}}
        self.persist = max(1, int(persist_s * fs))
        self.holdoff = int(holdoff_min * 60 * fs)
        self.fa_semana, self.n_rep, self.conf = fa_semana, n_rep, conf
        self.treinar_em = treinar_em

    def _persist(self, nome):
        return self.PERSIST_PADRAO.get(nome, self.persist)

    # --------------------------------------------------------- treino
    def treinar(self, y, sp, rodando):
        y = np.asarray(y, float)
        sp = None if sp is None else np.asarray(sp, float)
        g = portao(sp, rodando, self.fs, self.portao_s)
        m = np.asarray(rodando, bool) if self.treinar_em == "rodando" else g
        m = m & (np.arange(len(y)) >= max(self.p_max, self.q_max) + 1)
        if self.ordem is None:
            qs = [0] if sp is None else range(0, self.q_max + 1)
            melhor = None
            for p in [k for k in ORDENS_P if k <= self.p_max] or [self.p_max]:
                for q in qs:
                    X = matriz_arx(y, sp, p, q, self.fs)
                    c = _mq(X[m], y[m])
                    b = _bic(y[m] - _prever(c, X[m]), X.shape[1] + 1)
                    if melhor is None or b < melhor[0]:
                        melhor = (b, p, q)
            self.ordem = (melhor[1], melhor[2])
        p, q = self.ordem
        X = matriz_arx(y, sp, p, q, self.fs)
        self.coef = _mq(X[m], y[m])
        e = y - _prever(self.coef, X)
        self.media = float(e[g].mean())
        ec = e - self.media
        # AR do branqueamento ajustado so nas amostras do portao, mas com os atrasos tirados da
        # serie continua (nao da serie comprimida, que emendaria trechos separados por paradas)
        self.phi = ajustar_ar(ec, self.r_max, mask=g) if self.r_max else np.zeros(0)
        eps = branquear(ec, self.phi)[g]
        self.sigma = float(eps.std())
        self.lb_antes = ljung_box(e[g])[:2]
        self.lb_depois = ljung_box(eps)[:2]
        return self

    def treinar_trechos(self, series):
        """Treino com varios trechos normais (lista de (y, sp, rodando)) sem atravessar as juncoes."""
        return self.treinar(*juntar_trechos(series, max(self.p_max, self.q_max) + self.r_max + 1))

    def z(self, y, sp, rodando):
        """Residuo padronizado e branqueado (vetor comprimido no portao) e indices originais."""
        y = np.asarray(y, float)
        sp = None if sp is None else np.asarray(sp, float)
        g = portao(sp, rodando, self.fs, self.portao_s)
        p, q = self.ordem
        e = y - _prever(self.coef, matriz_arx(y, sp, p, q, self.fs)) - self.media
        idx = np.flatnonzero(g)
        return branquear(e, self.phi)[idx] / self.sigma, idx        # branqueia na serie continua

    # --------------------------------------------------------- calibracao
    def calibrar(self, series):
        """series: lista de (y, sp, rodando) NORMAIS, separadas do treino."""
        zs = [self.z(*s)[0] for s in series]
        n = len(self.estatisticas)
        self.limiar, self.limiar_reps, self.max_calib = {}, {}, {}
        for nome, par in self.estatisticas.items():
            if nome == "contagem":
                lim, _ = limiar_contagem(zs, par.get("W", 6000), par.get("k_pico", 5.0), self.fs,
                                         fa_semana=self.fa_semana / n)
                reps = [lim]
            else:
                lim, reps = limiar_bootstrap(zs, nome, par, self._persist(nome), self.holdoff, self.fs,
                                             fa_semana=self.fa_semana / n, n_rep=self.n_rep, conf=self.conf)
            self.limiar[nome], self.limiar_reps[nome] = lim, reps
            self.max_calib[nome] = max(float(persistente(ESTATISTICAS[nome](z, **par), self._persist(nome)).max())
                                       for z in zs)
        self.horas_calib = sum(len(z) for z in zs) / self.fs / 3600
        return self

    # --------------------------------------------------------- uso
    def estatisticas_de(self, y, sp, rodando):
        z, idx = self.z(y, sp, rodando)
        return {nome: ESTATISTICAS[nome](z, **par) for nome, par in self.estatisticas.items()}, z, idx

    def analisar(self, y, sp, rodando):
        """Devolve (dict de estatisticas comprimidas, idx, lista de alarmes)."""
        est, z, idx = self.estatisticas_de(y, sp, rodando)
        alarmes = []
        for nome, s in est.items():
            for i in eventos(s > self.limiar[nome], self._persist(nome), self.holdoff, idx):
                alarmes.append({"indice": i, "estatistica": nome})
        alarmes.sort(key=lambda a: a["indice"])
        # intervalo minimo entre alarmes vale para o indicador como um todo
        out, ultimo = [], -np.inf
        for a in alarmes:
            if a["indice"] - ultimo >= self.holdoff:
                out.append(a)
                ultimo = a["indice"]
        return est, idx, out

    # --------------------------------------------------------- exportar
    def exportar(self):
        return json.dumps({"fs": self.fs, "ordem": list(self.ordem), "coef": self.coef.tolist(),
                           "phi": self.phi.tolist(), "media": self.media, "sigma": self.sigma,
                           "portao_s": self.portao_s, "estatisticas": self.estatisticas,
                           "persist": self.persist, "holdoff": self.holdoff, "limiar": self.limiar})

    @classmethod
    def carregar(cls, texto):
        d = json.loads(texto)
        m = cls(fs=d["fs"], ordem=tuple(d["ordem"]), portao_s=d["portao_s"], estatisticas=d["estatisticas"])
        m.persist, m.holdoff = d["persist"], d["holdoff"]
        m.coef, m.phi = np.array(d["coef"]), np.array(d["phi"])
        m.media, m.sigma, m.limiar = d["media"], d["sigma"], d["limiar"]
        return m


def de_dataframe(df, alvo="pb1", vizinhos=("festoon", "pb2"), v_rodando=V_RODANDO):
    """(y, setpoint, rodando) a partir do formato de monitor.py (v_setpoint, v_<encoder>)."""
    rod = df[[f"v_{v}" for v in vizinhos]].mean(axis=1).to_numpy() > v_rodando
    return df[f"v_{alvo}"].to_numpy(float), df["v_setpoint"].to_numpy(float), rod
