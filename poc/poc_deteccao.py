"""
Simulacao e comparacao de detectores de anomalia em encoders sem historico de falhas.

Simula 3 encoders da mesma linha de papel (festoon, P&B1, P&B2), injeta falhas
no P&B1 e compara detectores treinados SOMENTE com dados normais:

  D1  "prever o proximo passo" univariado (AR nos valores passados do proprio P&B1)
  D1b Isolation Forest sobre janelas de 30 s do residuo do D1 (D1 avaliado por janelas)
  D2  redundancia analitica: estima o P&B1 a partir dos vizinhos (festoon, P&B2)
      + EWMA do residuo
  D3  Isolation Forest sobre janelas de 30 s do residuo do D2
  D4  regra sobre contador de erros de comunicacao do drive

Limiares sao calibrados em um trecho normal separado (validacao) com a regra
"maior valor observado + 10%".

Uso:  python poc_deteccao.py            (gera resultados.md e dados_sinteticos/)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge

FS = 10                      # Hz - taxa realista de leitura via CLP
DRAW = {"festoon": 0.000, "pb1": 0.002, "pb2": 0.004}   # diferenca de velocidade entre nips
TAU = {"festoon": 0.3, "pb1": 0.4, "pb2": 0.5}          # dinamica propria de cada nip (s)
V_RODANDO = 50.0             # m/min - abaixo disso a linha esta parada/lenta
PERSIST = 2 * FS             # alarme precisa persistir 2 s
HOLDOFF = 10 * 60 * FS       # 10 min entre eventos de alarme
FALHAS = ["perda_pulsos", "ruido", "deriva_ganho", "erros_com"]
OUT = Path(__file__).parent


# ----------------------------------------------------------------- simulacao
def perfil_linha(horas, rng):
    """Setpoint de velocidade da linha (m/min): paradas, rampas, producao, emendas."""
    n = int(horas * 3600 * FS)
    sp = np.zeros(n)
    i = 0

    def put(seg):
        nonlocal i
        k = max(0, min(len(seg), n - i))
        sp[i:i + k] = seg[:k]
        i += len(seg)

    while i < n:
        put(np.zeros(int(rng.uniform(60, 300) * FS)))                     # parada
        alvo = rng.choice([450, 550, 600, 650])
        put(np.linspace(0, alvo, int(rng.uniform(40, 90) * FS)))          # rampa
        fim = i + int(rng.uniform(20, 60) * 60 * FS)
        while i < min(fim, n):
            put(np.full(int(rng.uniform(5, 15) * 60 * FS), alvo))         # producao
            if rng.random() < 0.4:                                        # emenda de bobina
                d = 10 * FS
                put(np.concatenate([np.linspace(alvo, 0.7 * alvo, d),
                                    np.full(d, 0.7 * alvo),
                                    np.linspace(0.7 * alvo, alvo, d)]))
        put(np.linspace(alvo, 0, int(rng.uniform(30, 60) * FS)))          # desaceleracao
    return sp


def lag1(x, tau):
    a = (1 / FS) / tau
    return lfilter([a], [1, -(1 - a)], x)


def simular(horas, seed, falha=None, inicio_h=1.0, rampa_h=3.0):
    rng = np.random.default_rng(seed)
    sp = perfil_linha(horas, rng)
    n = len(sp)
    t = np.arange(n) / FS
    v = lag1(sp, 2.0)
    escala = v / 600
    osc = 0.3 * np.sin(2 * np.pi * 0.4 * t + rng.uniform(0, 2 * np.pi)) * escala  # malha de tensao
    comum = lfilter([1], [1, -0.95], rng.normal(0, 0.05, n)) * escala

    df = pd.DataFrame({"t_s": t, "setpoint": sp, "linha": v})
    for nome, d in DRAW.items():
        ruido = lfilter([1], [1, -0.7], rng.normal(0, 0.1, n))
        df[nome] = lag1(v, TAU[nome]) * (1 + d) + osc + comum + ruido

    sev = np.zeros(n)
    lam_erros = np.full(n, 0.2 / 3600 / FS)          # ~0,2 erro/h em condicao normal
    if falha:
        f0 = int(inicio_h * 3600 * FS)
        sev = np.clip((np.arange(n) - f0) / (rampa_h * 3600 * FS), 0, 1)
        rodando = v > V_RODANDO
        pb1 = df["pb1"].to_numpy().copy()
        if falha == "perda_pulsos":        # canal intermitente: quedas curtas de leitura
            p = np.where(np.arange(n) >= f0, 2e-4 + sev * 3e-3, 0.0)
            for i in np.flatnonzero((rng.random(n) < p) & rodando):
                pb1[i:i + rng.integers(2, 10)] *= 1 - rng.uniform(0.01, 0.05)
        elif falha == "ruido":             # contaminacao do disco / EMI
            pb1 += lfilter([1], [1, -0.7], rng.normal(0, 1, n)) * 0.1 * 6 * sev
        elif falha == "deriva_ganho":      # acoplamento folgando / escorregamento
            pb1 *= 1 - 0.006 * sev
        elif falha == "erros_com":         # cabo/conector degradando (encoder digital)
            lam_erros = lam_erros + 30 * sev / 3600 / FS
        df["pb1"] = pb1
    df["pb1_erros_com"] = rng.poisson(lam_erros)
    df["severidade"] = sev
    for c in ["festoon", "pb1", "pb2"]:
        df[c] = df[c].round(1)             # resolucao do valor lido
    return df


# ---------------------------------------------------------------- detectores
def rodando(df):
    return df["festoon"].to_numpy() > V_RODANDO


def atraso(x, k):
    return np.concatenate([np.full(k, x[0]), x[:-k]])


def feats_d1(df):   # so o passado do proprio encoder
    y = df["pb1"].to_numpy()
    return np.column_stack([atraso(y, k) for k in range(1, 6)]), y


def feats_d2(df):   # vizinhos na mesma linha de papel, sem o passado do proprio encoder
    f, p = df["festoon"].to_numpy(), df["pb2"].to_numpy()
    X = np.column_stack([f, p] + [atraso(f, k) for k in (1, 2, 3)] + [atraso(p, k) for k in (1, 2, 3)])
    return X, df["pb1"].to_numpy()


class DetectorResiduo:
    def __init__(self, feats, lam=0.05):
        self.feats, self.lam = feats, lam

    def fit(self, df):
        X, y = self.feats(df)
        m = rodando(df)
        self.reg = Ridge(alpha=1e-3).fit(X[m], y[m])
        self.sigma = (y[m] - self.reg.predict(X[m])).std()
        return self

    def z(self, df):
        X, y = self.feats(df)
        z = (y - self.reg.predict(X)) / self.sigma
        z[~rodando(df)] = 0.0
        return z

    def estatistica(self, df):
        return np.abs(lfilter([self.lam], [1, -(1 - self.lam)], self.z(df)))

    def calibrar(self, df_val):
        self.limiar = 1.1 * self.estatistica(df_val).max()
        return self

    def alarmes(self, df):
        flag = (self.estatistica(df) > self.limiar) & rodando(df)
        return eventos(flag, PERSIST)


def janelas(z, rod, W=30 * FS, S=10 * FS):
    F, fim = [], []
    for e in range(W, len(z) + 1, S):
        if rod[e - W:e].mean() < 0.9:
            continue
        s = z[e - W:e]
        F.append([s.mean(), s.std(), s.min(), s.max(), (s < -4).mean(), np.abs(np.diff(s)).mean()])
        fim.append(e - 1)
    return np.array(F), np.array(fim)


class DetectorIForest:
    def __init__(self, base):
        self.base = base

    def fit(self, df):
        F, _ = janelas(self.base.z(df), rodando(df))
        self.m = IsolationForest(n_estimators=200, random_state=0).fit(F)
        return self

    def calibrar(self, df_val):
        F, _ = janelas(self.base.z(df_val), rodando(df_val))
        self.limiar = self.m.score_samples(F).min() - 0.02
        return self

    def alarmes(self, df):
        F, fim = janelas(self.base.z(df), rodando(df))
        flag = np.zeros(len(df), bool)
        ruim = self.m.score_samples(F) < self.limiar
        dois = ruim & np.concatenate([[False], ruim[:-1]])   # 2 janelas seguidas
        flag[fim[dois]] = True
        return eventos(flag, 1)


class DetectorContador:
    def fit(self, df):
        return self

    def soma_1h(self, df):
        c = np.cumsum(df["pb1_erros_com"].to_numpy())
        return c - atraso(c, 3600 * FS)

    def calibrar(self, df_val):
        self.limiar = self.soma_1h(df_val).max() + 2
        return self

    def alarmes(self, df):
        return eventos(self.soma_1h(df) > self.limiar, 1)          # como monitor.py: > (maximo + 2)


def eventos(flag, persist):
    cheio = np.convolve(flag.astype(int), np.ones(persist, int), "full")[:len(flag)] >= persist
    ev, ultimo = [], -HOLDOFF
    for i in np.flatnonzero(cheio):
        if i - ultimo >= HOLDOFF:
            ev.append(i)
            ultimo = i
    return np.array(ev, int)


# ---------------------------------------------------------------- avaliacao
def main():
    treino = simular(8, seed=1)
    val = simular(8, seed=2)
    teste_normal = simular(24, seed=3)

    d1 = DetectorResiduo(feats_d1).fit(treino).calibrar(val)
    d2 = DetectorResiduo(feats_d2).fit(treino).calibrar(val)
    detectores = {
        "D1 prever proximo passo (AR univariado)": d1,
        "D1b Isolation Forest (janelas do residuo D1)": DetectorIForest(d1).fit(treino).calibrar(val),
        "D2 redundancia vizinhos + EWMA": d2,
        "D3 Isolation Forest (janelas do residuo D2)": DetectorIForest(d2).fit(treino).calibrar(val),
        "D4 contador de erros de comunicacao": DetectorContador().fit(treino).calibrar(val),
    }

    sementes = [10, 20, 30]
    cenarios = {(f, s): simular(6, seed=s + k, falha=f) for k, f in enumerate(FALHAS) for s in sementes}
    f0 = int(1.0 * 3600 * FS)

    linhas = []
    for nome, det in detectores.items():
        linha = {"detector": nome, "FA / 24 h normal": len(det.alarmes(teste_normal))}
        fa_antes = 0
        for falha in FALHAS:
            atrasos = []
            for s in sementes:
                ev = det.alarmes(cenarios[(falha, s)])
                fa_antes += int((ev < f0).sum())
                depois = ev[ev >= f0]
                if len(depois):
                    atrasos.append((depois[0] - f0) / FS / 60)
            linha[falha] = (f"{len(atrasos)}/{len(sementes)}, {np.median(atrasos):.0f} min"
                            if atrasos else f"0/{len(sementes)}")
        linha["FA antes da falha (12 cenarios)"] = fa_antes
        linhas.append(linha)
    res = pd.DataFrame(linhas)

    dados = OUT / "dados_sinteticos"
    dados.mkdir(exist_ok=True)
    treino.to_csv(dados / "normal_treino_8h.csv.gz", index=False)
    for falha in FALHAS:
        cenarios[(falha, sementes[0])].to_csv(dados / f"falha_{falha}_6h.csv.gz", index=False)

    texto = [
        "# Resultados da simulacao",
        "",
        f"Dados simulados a {FS} Hz. Treino: 8 h normal; calibracao: 8 h normal; teste normal: 24 h.",
        "Cenarios de falha: 6 h, falha comeca em 1 h e cresce linearmente ate severidade 1 na marca de 4 h (3 h de rampa).",
        "3 sementes por tipo de falha. Celula = cenarios detectados / total e atraso MEDIANO",
        "entre o inicio da falha e o primeiro alarme (severidade sobe ~0,33 a cada hora).",
        "FA = falsos alarmes (eventos separados por 10 min).",
        "",
        f"Sigma do residuo no treino: D1 = {d1.sigma:.3f} m/min, D2 = {d2.sigma:.3f} m/min.",
        "",
        res.to_markdown(index=False),
    ]
    (OUT / "resultados.md").write_text("\n".join(texto), encoding="utf-8")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
