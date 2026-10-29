"""
Comparacao de metodos de deteccao no MESMO protocolo da validacao conjunta -> comparacao_metodos.md/.csv

Simulador poc/malha_fechada_v2.py; sementes iguais as de validacao_conjunta.py:
    treino mf2 1-2 (12 h), calibracao mf2 3-8 (48 h), TESTE falhas 221-230, normais 196-200 (48 h), 2a rodada.
So os cenarios com falha nos sinais (velocidade/torque); os cenarios so de avisos (erros_com,
degradacao_sinal) nao mudam esses sinais e ficam com o Modelo 1 (resultado em validacao_conjunta.md).

Regra comum a todos os metodos novos: estatistica por bloco de 60 s com a linha em regime (mesmo
portao do Modelo 2), alarme quando passa do limite, intervalo minimo de 10 min entre alarmes, limite
calibrado por reamostragem em blocos de 2 h dos dados de calibracao para 1 alarme falso por semana
de calendario (mesmo criterio do Modelo 2).
Estatisticas por bloco (PCA, DPCA, CVA, IF): p-valor contra a calibracao (deixa-um-de-fora dentro da
calibracao, cauda exponencial alem dos 50 maiores). Estatisticas de janela (GLR, ADWIN): janela
atravessa paradas e recomeca depois de cada alarme, como a CUSUM do Modelo 2.

Metodos:
    M2        residuo de torque + CUSUM (monitor/torque_v2.py, parametros do conjunto) - referencia
    CONF      martingale conforme indutivo sobre o residuo de torque (p-valor empirico do max |z|)
    GLR       razao de verossimilhanca generalizada em janela de 4 h sobre as projecoes do Modelo 2
    ADWIN     janela adaptativa simplificada: maior diferenca de media entre duas subjanelas (4 h)
    PCA       PCA estatica, T2 + SPE (Q) sobre medias de 60 s dos sinais
    DPCA      PCA dinamica (2 defasagens), T2 + SPE
    CVA       analise de variaveis canonicas (passado/futuro de 3 blocos), T2 + Q
    IF        Isolation Forest sobre as mesmas medias de 60 s
Page-Hinkley nao e rodado: e a mesma recursao da CUSUM do Modelo 2 (k = delta, h = lambda).

    python comparacao_metodos.py        (roda uma vez; ~10-20 min)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from monitor.conjunto import PARAM_TORQUE
from monitor.torque_v2 import HIPOTESES, ROLOS, MonitorTorqueV2
from poc.malha_fechada_v2 import simular_mf2

FS, MEDIA_S, HOLDOFF, BLOCO_BOOT, SEMANAS_BOOT, META = 10, 60, 10, 120, 40, 1.0
TREINO, CAL = (1, 2), range(3, 9)
FALHAS, NORMAIS = range(221, 231), range(196, 201)   # 2a rodada de teste (ver validacao_conjunta.py)
INICIO_S, HORAS_NORMAL = 3600.0, 48
CASOS = [("deriva_ganho", "festoon"), ("deriva_ganho", "pb1"), ("deriva_ganho", "pb2"),
         ("atrito", "festoon"), ("atrito", "pb1"), ("atrito", "pb2"),
         ("perda_pulsos", "pb1"), ("perda_pulsos", "pb2"), ("ruido", "pb1"), ("ruido", "pb2")]
NOMES = {"M2": "Resíduo de torque + CUSUM (Modelo 2)", "CONF": "Conforme sobre o resíduo de torque",
         "GLR": "GLR sobre o resíduo de torque", "ADWIN": "Janela adaptativa (ADWIN simplificado)",
         "PCA": "PCA estática (T² e Q)", "DPCA": "PCA dinâmica (T² e Q)", "CVA": "Variáveis canônicas (CVA)",
         "IF": "Isolation Forest"}
OUT = Path(__file__).with_name("comparacao_metodos.md")
SEMANA_S = 168 * 3600.0
CAUDA = 50          # pontos da cauda empirica usados para extrapolar o p-valor alem do maximo (pvalor)


# ---------------------------------------------------------------- blocos e atributos
class Blocos:
    """Divide um df em blocos de 60 s com >= 80 % das amostras em regime (portao do Modelo 2)."""

    def __init__(self, mt):
        self.mt = mt

    def __call__(self, df):
        nb = MEDIA_S * FS
        g = self.mt.portao(df)
        n = len(df) // nb
        gg = g[: n * nb].reshape(n, nb)
        cont = gg.sum(axis=1)
        sel = np.flatnonzero(cont >= 0.8 * nb)
        cols = ["v_setpoint"] + [f"v_{r}" for r in ROLOS] + [f"tq_{r}" for r in ROLOS] + ["tensao_entrada",
                                                                                        "tensao_saida"]
        X = df[cols].to_numpy(float)[: n * nb].reshape(n, nb, len(cols))
        w = gg[..., None]
        med = (X * w).sum(axis=1) / np.maximum(cont, 1)[:, None]
        V = X[..., 1:4]
        dv = np.diff(V, axis=1)
        ruido = np.sqrt((dv ** 2 * w[:, 1:]).sum(axis=1) / np.maximum(cont, 2)[:, None])
        F = np.hstack([med, ruido])[sel]
        inicio = np.ones(len(sel), bool)
        inicio[1:] = np.diff(sel) > 1
        t = df["t_s"].to_numpy()[np.minimum((sel + 1) * nb - 1, len(df) - 1)]
        return F, t, inicio, sel


def defasar(F, inicio, L):
    """[F_t, F_{t-1}, ..., F_{t-L}] so dentro do mesmo trecho; devolve (X, valido)."""
    seg = np.cumsum(inicio)
    X = [F]
    ok = np.ones(len(F), bool)
    for i in range(1, L + 1):
        X.append(np.vstack([np.zeros((i, F.shape[1])), F[:-i]]))
        ok[i:] &= seg[i:] == seg[:-i]
    ok[:L] = False
    return np.hstack(X), ok


# ---------------------------------------------------------------- estatisticas (sem estado)
class PCAStat:
    def __init__(self, lags=0, var=0.95):
        self.lags, self.var = lags, var

    def treinar(self, Fs):
        X = np.vstack([defasar(F, i, self.lags)[0][defasar(F, i, self.lags)[1]] for F, i in Fs])
        self.m, self.s = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.m) / self.s
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        lam = S ** 2 / (len(Z) - 1)
        self.a = int(np.searchsorted(np.cumsum(lam) / lam.sum(), self.var) + 1)
        self.P, self.lam = Vt[: self.a].T, lam[: self.a]
        return self

    def stat(self, F, inicio):
        X, ok = defasar(F, inicio, self.lags)
        Z = (X - self.m) / self.s
        T = Z @ self.P
        t2 = (T ** 2 / self.lam).sum(1)
        q = ((Z - T @ self.P.T) ** 2).sum(1)
        return np.column_stack([t2, q]), ok


class CVAStat:
    def __init__(self, p=3, f=3, var=0.95):
        self.p, self.f, self.var = p, f, var

    def _passado(self, F, inicio):
        """p_t = [F_{t-1} .. F_{t-p}] (estado estimado para o instante t, sem usar F_t)."""
        Xp, ok = defasar(F, inicio, self.p)
        return Xp[:, F.shape[1]:], ok

    def _pf(self, F, inicio):
        """Pares (passado, futuro) SEM sobreposicao, alinhados no fim do futuro tau = t + f - 1:
        futuro f_t = [F_{t+f-1} .. F_t] (defasagens 0..f-1 de tau), passado p_t = [F_{t-1} .. F_{t-p}]
        (defasagens f..f+p-1 de tau). Antes o passado eram as defasagens 1..p de tau e repetia
        f-1 blocos do futuro (correlacao canonica trivial = 1)."""
        nF = F.shape[1]
        D, ok = defasar(F, inicio, self.p + self.f - 1)
        return D[:, self.f * nF:], D[:, : self.f * nF], ok

    def treinar(self, Fs):
        P, Fu = [], []
        for F, i in Fs:
            p, f, ok = self._pf(F, i)
            P.append(p[ok]); Fu.append(f[ok])
        P, Fu = np.vstack(P), np.vstack(Fu)
        self.mp, self.sp = P.mean(0), P.std(0) + 1e-9
        self.mf, self.sf = Fu.mean(0), Fu.std(0) + 1e-9
        P, Fu = (P - self.mp) / self.sp, (Fu - self.mf) / self.sf
        n = len(P)
        Spp, Sff, Sfp = P.T @ P / n + 1e-6 * np.eye(P.shape[1]), Fu.T @ Fu / n + 1e-6 * np.eye(Fu.shape[1]), Fu.T @ P / n

        def isqrt(M):
            w, V = np.linalg.eigh(M)
            return V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-12))) @ V.T

        H = isqrt(Sff) @ Sfp @ isqrt(Spp)
        U, S, Vt = np.linalg.svd(H)
        e = S ** 2
        self.a = int(np.searchsorted(np.cumsum(e) / e.sum(), self.var) + 1)
        J = Vt.T
        self.J = (J.T @ isqrt(Spp))[: self.a]                 # estados canonicos
        self.Lr = (J.T @ isqrt(Spp))[self.a:]                 # residuo
        return self

    def stat(self, F, inicio):
        p, ok = self._passado(F, inicio)
        P = (p - self.mp) / self.sp
        z = P @ self.J.T
        r = P @ self.Lr.T
        return np.column_stack([(z ** 2).sum(1), (r ** 2).sum(1)]), ok


class IFStat:
    def treinar(self, Fs):
        X = np.vstack([F for F, _ in Fs])
        self.m, self.s = X.mean(0), X.std(0) + 1e-9
        self.f = IsolationForest(n_estimators=200, random_state=0).fit((X - self.m) / self.s)
        return self

    def stat(self, F, inicio):
        return -self.f.score_samples((F - self.m) / self.s)[:, None], np.ones(len(F), bool)


class GLRStat:
    """max sobre hipoteses e instantes de mudanca na janela de (soma z)^2 / n (media desconhecida)."""

    def __init__(self, W=240):
        self.W = W

    def stat(self, Z, inicio):
        n, h = Z.shape
        C = np.vstack([np.zeros((1, h)), np.cumsum(Z, axis=0)])
        seg = np.cumsum(inicio)
        g = np.zeros(n)
        arg = np.zeros(n, int)
        for m in range(1, self.W + 1):
            t = np.arange(m - 1, n)
            ok = seg[t] == seg[t - m + 1]
            s = (C[t + 1] - C[t + 1 - m]) ** 2 / m
            best = s.max(1)
            upd = ok & (best > g[t])
            g[t[upd]] = best[upd]
            arg[t[upd]] = s.argmax(1)[upd]
        return g[:, None], np.ones(n, bool), arg


class ADWINStat:
    """Para cada t: maior |media(ultimos m) - media(anteriores na janela)| / limite de Hoeffding/Bernstein
    simplificado, com janela de W blocos, na norma do residuo branqueado."""

    def __init__(self, W=240, passo=5):
        self.W, self.passo = W, passo

    def stat(self, x, inicio):
        n = len(x)
        C = np.concatenate([[0.0], np.cumsum(x)])
        seg = np.cumsum(inicio)
        g = np.zeros(n)
        for W in range(20, self.W + 1, 20):
            t = np.arange(W - 1, n)
            ok = seg[t] == seg[t - W + 1]
            for m in range(self.passo, W - self.passo + 1, self.passo):
                n1, n0 = m, W - m
                mu1 = (C[t + 1] - C[t + 1 - m]) / n1
                mu0 = (C[t + 1 - m] - C[t + 1 - W]) / n0
                z = np.abs(mu1 - mu0) / np.sqrt(1 / n1 + 1 / n0)
                upd = ok & (z > g[t])
                g[t[upd]] = z[upd]
        return g[:, None], np.ones(n, bool)


# ---------------------------------------------------------------- p-valor combinado e alarmes
def p_empirico(r, x, dentro=False):
    """p-valor conforme da cauda superior contra a referencia ORDENADA r.
    dentro=True: x pertence a r (calibracao) -> deixa-um-de-fora, (#{r_j >= x}) / n, que ja conta o
    proprio ponto uma vez; a formula de fora, (#{r_j >= x} + 1)/(n + 1), contaria o ponto duas vezes e
    deixaria os p-valores da calibracao maiores que os do teste (limite frouxo -> mais FA no teste)."""
    k = len(r) - np.searchsorted(r, x, side="left")
    return np.maximum(k, 1) / len(r) if dentro else (k + 1) / (len(r) + 1)


def p_cauda(r, x, dentro=False, m=CAUDA):
    """p_empirico com a cauda acima do m-esimo maior valor extrapolada por exponencial (POT):
    p = (m/n) exp(-(x - u)/beta), beta = excesso medio dos m maiores sobre u.
    Sem isso o p-valor tem piso 1/n (n ~ 14 mil blocos de calibracao, ~8 mil por semana): so o maior
    valor de cada coluna ja da ~1 alarme por semana na reamostragem, a bissecao termina ACIMA do maior
    -log10 p atingivel no teste (log10(n+1)) e o metodo fica mudo (CVA: 0 deteccoes no dev), ou o
    limite cai num degrau da escala discreta definido por 1-2 pontos."""
    p = p_empirico(r, x, dentro)
    n = len(r)
    if n <= m:
        return p
    u = r[n - m - 1]
    beta = max(float(np.mean(r[n - m:] - u)), 1e-12)
    return np.where(x > u, np.maximum((m / n) * np.exp(-(x - u) / beta), 1e-300), p)   # sem log10(0)


def pvalor(ref, x, dentro=False):
    """p-valor da cauda superior (empirico + cauda exponencial), coluna a coluna; combina pelo minimo
    (escala -log10)."""
    out = np.zeros(len(x))
    for c in range(x.shape[1]):
        out = np.maximum(out, -np.log10(p_cauda(np.sort(ref[:, c]), x[:, c], dentro)))
    return out


def alarmes(s, L):
    """indices onde s > L respeitando intervalo minimo (sem estado alem do intervalo)."""
    acima = np.flatnonzero(s > L)
    out, prox = [], -1
    for i in acima:
        if i >= prox:
            out.append(i)
            prox = i + HOLDOFF
    return out


def cusum_logp(p, eps, L):
    """martingale conforme 'power' em forma de CUSUM: S = max(0, S + log(eps p^(eps-1))); reinicia no alarme."""
    inc = np.log(eps) + (eps - 1) * np.log(p)
    S, out, prox = 0.0, [], -1
    for i, v in enumerate(inc):
        S = max(0.0, S + v)
        if S > L and i >= prox:
            out.append(i)
            prox, S = i + HOLDOFF, 0.0
    return out


def alarmes_janela(fn, X, W, L, maximo=None, cache=None):
    """Alarmes de uma estatistica de janela (GLR, ADWIN) que RECOMECA depois de cada alarme, como a
    CUSUM do Modelo 2 (volta a zero) e o martingale conforme: sem isso uma unica excursao deixa a janela
    de 4 h acima do limite por horas e conta um alarme a cada 10 min (FA em rajada, ~20 por arquivo).
    fn(X) -> (g, arg ou None) com X tratado como um trecho unico; depois de um alarme em i so os W
    blocos seguintes mudam (a janela comeca em i + 1), o resto e a estatistica sem reinicio."""
    if cache is not None and id(X) in cache:
        g0, a0 = cache[id(X)]
    else:
        g0, a0 = fn(X)
        if cache is not None:
            cache[id(X)] = (g0, a0)
    out, pos, prox, n = [], 0, 0, len(X)
    while pos < n and (maximo is None or len(out) < maximo):
        fim = min(n, pos + W) if pos else 0
        gl, al = fn(X[pos:fim]) if fim > pos else (np.empty(0), None)
        g = np.concatenate([gl, g0[fim:]])
        cand = np.flatnonzero(g > L) + pos
        cand = cand[cand >= prox]
        if not len(cand):
            break
        i = int(cand[0])
        out.append((i, None if a0 is None else int(al[i - pos] if i < fim else a0[i])))
        pos, prox = i + 1, i + HOLDOFF
    return out


def calibrar(series, contar, n_sem, rng_seed=0, lo=1e-3, hi=1e4):
    """L tal que a taxa de alarmes na reamostragem de semanas (blocos de 2 h) = META por semana."""
    rng = np.random.default_rng(rng_seed)
    S = np.concatenate(series)
    nb = int(np.ceil(n_sem / BLOCO_BOOT))
    semanas = [np.concatenate([S[i:i + BLOCO_BOOT] for i in rng.integers(0, len(S) - BLOCO_BOOT, nb)])[:n_sem]
               for _ in range(SEMANAS_BOOT)]

    def taxa(L, parar=True):
        tot = 0
        for w in semanas:
            tot += len(contar(w, L))
            if parar and tot > META * SEMANAS_BOOT:             # ja passou da meta: decisao igual
                break
        return tot / SEMANAS_BOOT

    while hi / lo > 1.01:
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if taxa(mid) > META else (lo, mid)
    return float(hi), taxa(hi, parar=False)


# ---------------------------------------------------------------- execucao
def um_trecho(inicio):
    """Um unico trecho por arquivo: GLR e ADWIN somam o residuo branqueado (ordem_ar = 0, sem
    defasagens) atravessando paradas, como a CUSUM do Modelo 2 faz. Com o corte por trecho de regime
    (mediana ~15 blocos, maximo ~60) a janela de 4 h nunca passava de ~1 h e o ADWIN (W >= 20) ficava
    cego em ~1/3 dos blocos."""
    out = np.zeros(len(inicio), bool)
    out[:1] = True
    return out


def correto_iso(falha, rolo, hip):
    return HIPOTESES[hip] == (("mecanica" if falha == "atrito" else "encoder"), rolo)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    global FALHAS, NORMAIS, OUT
    if len(sys.argv) > 1 and sys.argv[1] == "dev":          # conferencia rapida do pipeline
        FALHAS, NORMAIS, OUT = range(41, 43), range(46, 47), OUT.with_name("comparacao_metodos_dev.md")
    t0 = time.time()
    treino = [simular_mf2(12, seed=s) for s in TREINO]
    cal = [simular_mf2(48, seed=s) for s in CAL]
    mt = MonitorTorqueV2(usar_tensao_pontas=True, **PARAM_TORQUE).treinar(treino).calibrar(cal)
    blocos = Blocos(mt)
    horas_cal = sum(len(d) for d in cal) / FS / 3600

    # entradas por metodo: atributos F (PCA, DPCA, CVA, IF) e projecoes Z do Modelo 2 (CONF, GLR, ADWIN)
    def zinfo(df):
        R, idx, inicio = mt._blocos(df)
        E, val = mt._branquear(R, inicio)
        Z = E[val] @ mt.W
        ini = inicio[val]
        return Z, df["t_s"].to_numpy()[idx[val]], ini, np.linalg.norm(E[val], axis=1)

    Ftr = [blocos(d)[:3] for d in treino]
    Fcal = [blocos(d)[:3] for d in cal]
    Zcal = [zinfo(d) for d in cal]
    n_reg = sum(len(F) for F, _, _ in Fcal)
    n_sem = int(n_reg * 168 / horas_cal)                    # blocos de regime por semana de calendario

    est = {"PCA": PCAStat(0).treinar([(F, i) for F, _, i in Ftr]),
           "DPCA": PCAStat(2).treinar([(F, i) for F, _, i in Ftr]),
           "CVA": CVAStat().treinar([(F, i) for F, _, i in Ftr]),
           "IF": IFStat().treinar([(F, i) for F, _, i in Ftr])}
    ref, lim, info = {}, {}, {}
    for k, e in est.items():                                    # p-valores contra a calibracao
        st = [e.stat(F, i) for F, _, i in Fcal]
        ref[k] = np.vstack([s[ok] for s, ok in st])
        series = [np.where(ok, pvalor(ref[k], s, dentro=True), 0.0) for s, ok in st]   # p deixa-um-de-fora
        lim[k], info[k] = calibrar(series, alarmes, n_sem)
    glr, adw = GLRStat(), ADWINStat()

    def f_glr(Z):
        g, _, arg = glr.stat(Z, um_trecho(np.ones(len(Z), bool)))
        return g[:, 0], arg

    def f_adw(x):
        return adw.stat(x, um_trecho(np.ones(len(x), bool)))[0][:, 0], None

    # a reamostragem e feita sobre as entradas (Z, |e|) e a estatistica recalculada com reinicio, como
    # no Modelo 2; teto por semana > META * SEMANAS_BOOT nao muda a decisao da bissecao
    teto = int(META * SEMANAS_BOOT) + 1
    cg, ca = {}, {}
    lim["GLR"], info["GLR"] = calibrar([Z for Z, _, _, _ in Zcal],
                                       lambda w, L: alarmes_janela(f_glr, w, glr.W, L, teto, cg), n_sem)
    lim["ADWIN"], info["ADWIN"] = calibrar([nrm for _, _, _, nrm in Zcal],
                                           lambda w, L: alarmes_janela(f_adw, w, adw.W, L, teto, ca), n_sem)
    sc_cal = np.concatenate([np.abs(Z).max(1) for Z, _, _, _ in Zcal])
    sc_sorted = np.sort(sc_cal)

    def pconf(Z, dentro=False):
        return p_empirico(sc_sorted, np.abs(Z).max(1), dentro)

    EPS = 0.5
    lim["CONF"], info["CONF"] = calibrar([pconf(Z, dentro=True) for Z, _, _, _ in Zcal],
                                         lambda w, L: cusum_logp(w, EPS, L), n_sem, lo=0.1, hi=1e3)
    print(f"calibracao: {time.time() - t0:.0f} s; limites {lim}", flush=True)

    def alarmes_df(df):
        """{metodo: [(t_s, hipotese ou None)]}"""
        out = {"M2": [(e["t_s"], HIPOTESES.index((e["tipo"], e["rolo"]))) for e in mt.analisar(df)[1]]}
        F, t, inicio, _ = blocos(df)
        for k, e in est.items():
            s, ok = e.stat(F, inicio)
            out[k] = [(t[i], None) for i in alarmes(np.where(ok, pvalor(ref[k], s), 0.0), lim[k])]
        Z, tz, iz, nrm = zinfo(df)
        out["GLR"] = [(tz[i], a % len(HIPOTESES)) for i, a in alarmes_janela(f_glr, Z, glr.W, lim["GLR"])]
        out["ADWIN"] = [(tz[i], None) for i, _ in alarmes_janela(f_adw, nrm, adw.W, lim["ADWIN"])]
        out["CONF"] = [(tz[i], None) for i in cusum_logp(pconf(Z), EPS, lim["CONF"])]
        return out

    fa = {k: 0 for k in NOMES}
    for s in NORMAIS:
        for k, a in alarmes_df(simular_mf2(HORAS_NORMAL, seed=s)).items():
            fa[k] += len(a)
    print(f"normais: {fa} ({time.time() - t0:.0f} s)", flush=True)

    linhas = []
    pre = {k: 0 for k in NOMES}
    for falha, rolo in CASOS:
        det = {k: [] for k in NOMES}
        iso = {k: 0 for k in NOMES}
        for s in FALHAS:
            df = simular_mf2(6, seed=s, falha=falha, rolo=rolo)
            for k, a in alarmes_df(df).items():
                pre[k] += sum(t < INICIO_S for t, _ in a)
                dep = [(t, h) for t, h in a if t >= INICIO_S]
                if dep:
                    det[k].append((dep[0][0] - INICIO_S) / 60)
                    if dep[0][1] is not None:
                        iso[k] += correto_iso(falha, rolo, dep[0][1])
        for k in NOMES:
            linhas.append({"metodo": k, "falha": falha, "rolo": rolo, "detectados": len(det[k]),
                           "n": len(FALHAS), "atraso_mediano_min": float(np.median(det[k])) if det[k] else np.nan,
                           "isolacao_certa": iso[k] if k in ("M2", "GLR") else np.nan})
        print(f"  {falha} {rolo}: " + ", ".join(f"{k} {len(det[k])}" for k in NOMES), flush=True)

    df = pd.DataFrame(linhas)
    horas_fa = HORAS_NORMAL * len(NORMAIS) + len(CASOS) * len(FALHAS)
    resumo = []
    for k in NOMES:
        d = df[df.metodo == k]
        por = d.groupby("falha")["detectados"].sum()
        resumo.append({"metodo": NOMES[k], "chave": k,
                       **{f: int(por.get(f, 0)) for f in ("deriva_ganho", "atrito", "perda_pulsos", "ruido")},
                       "total": int(d.detectados.sum()), "n_total": int(d.n.sum()),
                       "atraso_mediano_min": round(float(np.nanmedian(d.atraso_mediano_min)), 0)
                       if d.detectados.sum() else None,
                       "fa": fa[k] + pre[k], "fa_semana": round((fa[k] + pre[k]) / horas_fa * 168, 2),
                       # separados: a 1a hora de cada cenario e partida a frio (redutor frio), 100 partidas
                       # em 100 h contra 6 em 288 h na calibracao; o Modelo 2 modela o aquecimento, os
                       # metodos sobre medias brutas (PCA, DPCA, CVA, IF) nao
                       "fa_normais": fa[k], "fa_1a_hora": pre[k],
                       "isolacao": f"{int(d.isolacao_certa.sum())}/{int(d.detectados.sum())}"
                       if k in ("M2", "GLR") else "-",
                       "fa_boot_cal": round(info[k], 2) if k in info else None})
    R = pd.DataFrame(resumo)
    R.to_csv(OUT.with_suffix(".csv"), index=False)
    df.to_csv(OUT.with_name(OUT.stem + "_casos.csv"), index=False)
    texto = ["# Comparação de métodos no simulador de malha fechada v2", "",
             "Gerado por `comparacao_metodos.py` (protocolo no cabeçalho). Sementes de teste 221–230 (falhas) e "
             "196–200 (normais, 5 × 48 h), as mesmas de `validacao_conjunta.md`. Só cenários com falha nos sinais "
             f"(100 cenários). Alarmes falsos contados em {horas_fa} h (normais + 1ª hora dos cenários; "
             f"`fa_normais` = {HORAS_NORMAL * len(NORMAIS)} h normais, `fa_1a_hora` = 1ª hora dos "
             f"{len(CASOS) * len(FALHAS)} cenários, que começa com o redutor frio — uma partida a frio por "
             f"hora contra {len(CAL)} em {len(CAL) * 48} h na calibração). "
             "Todos os limites calibrados para 1 alarme falso por semana na mesma calibração (mf2 3–8).", "",
             R.drop(columns=["chave"]).to_markdown(index=False), "",
             "Por caso:", "", df.to_markdown(index=False), "", f"Tempo: {time.time() - t0:.0f} s."]
    OUT.write_text("\n".join(texto), encoding="utf-8")
    print("\n".join(texto[:6]))


if __name__ == "__main__":
    main()
