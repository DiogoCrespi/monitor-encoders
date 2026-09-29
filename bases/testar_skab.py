"""
Testa os detectores na base publica SKAB
(Skoltech Anomaly Benchmark - bancada real de bomba/circulacao de agua, 1 Hz,
8 sensores, 34 experimentos com anomalias rotuladas).
https://github.com/waico/SKAB  (GPL-3.0)

Protocolo oficial do SKAB (problema de 'outlier detection'):
  - para cada arquivo: treino = 400 primeiras amostras (normais), teste = restante
  - metricas binarias somadas em todos os arquivos: F1, FAR (falso alarme), MAR (alarme perdido)
  - leaderboard publicado no README do SKAB para comparacao

Detectores:
  T2   PCA T-quadrado - baseline do leaderboard, serve para validar este script
  D1   'prever o proximo passo': AR(5) de cada sensor so com o proprio passado
  D2   redundancia: cada sensor estimado pelos OUTROS sensores no mesmo instante
  D3   Isolation Forest em janelas de 10 s dos residuos do D2
Linha de corte = maior valor da estatistica no treino.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "SKAB" / "data"
N_TREINO = 400


def carregar():
    arquivos = sorted(p for p in DATA.rglob("*.csv") if "anomaly-free" not in str(p))
    return [(p.parent.name + "/" + p.name,
             pd.read_csv(p, sep=";", index_col="datetime", parse_dates=True)) for p in arquivos]


def atraso(X, k):
    return np.vstack([np.repeat(X[:1], k, 0), X[:-k]])


# ---- estatisticas: recebem (Xtr, Xte) padronizados, devolvem (s_treino, s_teste)
def stat_t2(Xtr, Xte):
    pca = PCA(n_components=0.9).fit(Xtr)
    var = pca.explained_variance_

    def t2(X):
        return ((pca.transform(X) ** 2) / var).sum(1)
    return t2(Xtr), t2(Xte)


def residuos_d1(Xtr, Xte):
    def feats(X):
        return [np.column_stack([atraso(X[:, [j]], k) for k in range(1, 6)]) for j in range(X.shape[1])]
    Ftr, Fte = feats(Xtr), feats(Xte)
    Rtr, Rte = np.zeros_like(Xtr), np.zeros_like(Xte)
    for j in range(Xtr.shape[1]):
        m = Ridge(alpha=1e-2).fit(Ftr[j], Xtr[:, j])
        Rtr[:, j] = Xtr[:, j] - m.predict(Ftr[j])
        Rte[:, j] = Xte[:, j] - m.predict(Fte[j])
    s = Rtr.std(0) + 1e-9
    return Rtr / s, Rte / s


def residuos_d2(Xtr, Xte):
    Rtr, Rte = np.zeros_like(Xtr), np.zeros_like(Xte)
    for j in range(Xtr.shape[1]):
        outros = [k for k in range(Xtr.shape[1]) if k != j]
        m = Ridge(alpha=1.0).fit(Xtr[:, outros], Xtr[:, j])
        Rtr[:, j] = Xtr[:, j] - m.predict(Xtr[:, outros])
        Rte[:, j] = Xte[:, j] - m.predict(Xte[:, outros])
    s = Rtr.std(0) + 1e-9
    return Rtr / s, Rte / s


def norma(R):
    return (R ** 2).sum(1)


def stat_d1(Xtr, Xte):
    a, b = residuos_d1(Xtr, Xte)
    return norma(a), norma(b)


def stat_d2(Xtr, Xte):
    a, b = residuos_d2(Xtr, Xte)
    return norma(a), norma(b)


def feats_janela(R, W=10):
    df = pd.DataFrame(R)
    F = pd.concat([df.rolling(W, min_periods=1).mean(), df.rolling(W, min_periods=1).std().fillna(0)], axis=1)
    return F.to_numpy()


def stat_d3(Xtr, Xte):
    a, b = residuos_d2(Xtr, Xte)
    Ftr, Fte = feats_janela(a), feats_janela(b)
    m = IsolationForest(n_estimators=200, random_state=0).fit(Ftr)
    return -m.score_samples(Ftr), -m.score_samples(Fte)


DETECTORES = {
    "T2 (PCA, baseline do leaderboard)": stat_t2,
    "D1 prever proximo passo (AR por sensor)": stat_d1,
    "D2 redundancia entre sensores": stat_d2,
    "D3 Isolation Forest no residuo D2": stat_d3,
}


def avaliar(stat_fn, arquivos, suavizar):
    TP = FP = TN = FN = 0
    por_grupo = {}
    for nome, df in arquivos:
        X = df.drop(columns=["anomaly", "changepoint"]).to_numpy(float)
        y = df["anomaly"].to_numpy(int)[N_TREINO:]
        sc = StandardScaler().fit(X[:N_TREINO])
        s_tr, s_te = stat_fn(sc.transform(X[:N_TREINO]), sc.transform(X[N_TREINO:]))
        if suavizar:  # EWMA
            s_tr = pd.Series(s_tr).ewm(alpha=0.2).mean().to_numpy()
            s_te = pd.Series(s_te).ewm(alpha=0.2).mean().to_numpy()
        pred = (s_te > s_tr.max()).astype(int)
        pred = pd.Series(pred).rolling(3, min_periods=1).median().to_numpy().astype(int)
        tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
        tn, fn = int(((pred == 0) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
        TP, FP, TN, FN = TP + tp, FP + fp, TN + tn, FN + fn
        g = nome.split("/")[0]
        acc = por_grupo.setdefault(g, [0, 0, 0, 0])
        for i, v in enumerate((tp, fp, tn, fn)):
            acc[i] += v

    def met(tp, fp, tn, fn):
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        return f1, 100 * fp / max(fp + tn, 1), 100 * fn / max(fn + tp, 1)
    return met(TP, FP, TN, FN), {g: met(*v) for g, v in por_grupo.items()}


def main():
    arquivos = carregar()
    linhas, grupos = [], []
    for suav in (False, True):
        for nome, fn in DETECTORES.items():
            (f1, far, mar), pg = avaliar(fn, arquivos, suav)
            rotulo = nome + (" + EWMA" if suav else "")
            linhas.append({"detector": rotulo, "F1": round(f1, 2), "FAR %": round(far, 1), "MAR %": round(mar, 1)})
            grupos.append({"detector": rotulo, **{g: round(v[0], 2) for g, v in pg.items()}})
    res = pd.DataFrame(linhas).sort_values("F1", ascending=False)
    grp = pd.DataFrame(grupos)
    texto = [
        "# Teste na base publica SKAB",
        "",
        f"{len(arquivos)} experimentos, 1 Hz, 8 sensores (vibracao, corrente, pressao, temperatura, tensao, vazao).",
        "Protocolo oficial: treino = 400 primeiras amostras de cada arquivo; teste = restante.",
        "Linha de corte = maior valor no treino; mediana movel de 3 amostras (como o SKAB faz).",
        "",
        "## Resultado geral",
        res.to_markdown(index=False),
        "",
        "Leaderboard oficial (README do SKAB) para comparar: Conv-AE F1 0.78 / FAR 13.55 / MAR 28.02;",
        "MSET 0.78; T2+Q 0.76; LSTM-AE 0.74; T2 0.66 / FAR 19.21 / MAR 42.6; Vanilla LSTM 0.54;",
        "Isolation Forest 0.29.",
        "",
        "## F1 por grupo de experimento",
        "valve1/valve2 = fechamento de valvula; other = vazamento, desbalanceamento do rotor, cavitacao, temperatura",
        "",
        grp.to_markdown(index=False),
    ]
    out = Path(__file__).parent / "resultados_skab.md"
    out.write_text("\n".join(texto), encoding="utf-8")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
