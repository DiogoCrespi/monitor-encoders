"""
Diagnostico: por que o Isolation Forest detectou 14/100 em comparacao_metodos.py?  -> diagnostico_isolation_forest.md

Hipotese: o Isolation Forest isola pontos com cortes paralelos aos eixos. A falha nao tira nenhum sinal
da faixa normal; ela muda a RELACAO entre os sinais (torque x velocidade x tensao), que a PCA (Q) ve e
cortes eixo a eixo nao veem.

Mesmo protocolo de comparacao_metodos.py (sementes, blocos de 60 s, limite para 1 alarme falso/semana).
Variantes, todas com o mesmo IsolationForest(n_estimators=200):
    IF        atributos originais padronizados (= comparacao_metodos.py)
    IF_PCA    atributos rotacionados e branqueados pela PCA (todas as componentes)
    IF_Q      so o residuo da PCA (componentes fora das 95 % de variancia)
    IF_M2     projecoes branqueadas do residuo de torque do Modelo 2 (entrada do CONF e do GLR)
    PCA       PCA estatica T2 + Q (referencia, = comparacao_metodos.py)
Medida extra: fracao dos blocos com falha (segunda metade do cenario) com TODOS os atributos dentro
da faixa [min, max] do treino.

    python diagnostico_isolation_forest.py        (~25 min)
    python diagnostico_isolation_forest.py dev    (conferencia rapida)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import comparacao_metodos as cm
from monitor.conjunto import PARAM_TORQUE
from monitor.torque_v2 import MonitorTorqueV2
from poc.malha_fechada_v2 import simular_mf2

OUT = Path(__file__).with_name("diagnostico_isolation_forest.md")
NOMES = {"IF": "Isolation Forest nos atributos originais", "IF_PCA": "Isolation Forest nos atributos rotacionados pela PCA",
         "IF_Q": "Isolation Forest no resíduo da PCA", "IF_M2": "Isolation Forest no resíduo de torque (Modelo 2)",
         "PCA": "PCA estática T² e Q (referência)"}


class IFTransf:
    """Isolation Forest sobre uma transformacao linear dos atributos."""

    def __init__(self, modo, pca=None):
        self.modo, self.pca = modo, pca

    def _x(self, F):
        Z = (F - self.pca.m) / self.pca.s
        if self.modo == "rot":
            return Z @ self.V.T / self.sv
        if self.modo == "q":
            return Z - (Z @ self.pca.P) @ self.pca.P.T
        return Z

    def treinar(self, Fs):
        X = np.vstack([F for F, _ in Fs])
        Z = (X - self.pca.m) / self.pca.s
        _, S, self.V = np.linalg.svd(Z, full_matrices=False)
        self.sv = S / np.sqrt(len(Z) - 1) + 1e-9
        self.f = IsolationForest(n_estimators=200, random_state=0).fit(self._x(X))
        return self

    def stat(self, F, inicio):
        return -self.f.score_samples(self._x(F))[:, None], np.ones(len(F), bool)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    falhas, normais, out = cm.FALHAS, cm.NORMAIS, OUT
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        falhas, normais, out = range(41, 43), range(46, 47), OUT.with_name("diagnostico_isolation_forest_dev.md")
    t0 = time.time()
    treino = [simular_mf2(12, seed=s) for s in cm.TREINO]
    cal = [simular_mf2(48, seed=s) for s in cm.CAL]
    mt = MonitorTorqueV2(usar_tensao_pontas=True, **PARAM_TORQUE).treinar(treino).calibrar(cal)
    blocos = cm.Blocos(mt)
    horas_cal = sum(len(d) for d in cal) / cm.FS / 3600

    def zinfo(df):
        R, idx, inicio = mt._blocos(df)
        E, val = mt._branquear(R, inicio)
        return E[val] @ mt.W, df["t_s"].to_numpy()[idx[val]], inicio[val]

    Ftr = [blocos(d)[:3] for d in treino]
    Fcal = [blocos(d)[:3] for d in cal]
    Ztr = [zinfo(d) for d in treino]
    Zcal = [zinfo(d) for d in cal]
    n_reg = sum(len(F) for F, _, _ in Fcal)
    n_sem = int(n_reg * 168 / horas_cal)
    Xtr = np.vstack([F for F, _, _ in Ftr])
    lo, hi = Xtr.min(0), Xtr.max(0)

    pca = cm.PCAStat(0).treinar([(F, i) for F, _, i in Ftr])
    est = {"IF": cm.IFStat().treinar([(F, i) for F, _, i in Ftr]),
           "IF_PCA": IFTransf("rot", pca).treinar([(F, i) for F, _, i in Ftr]),
           "IF_Q": IFTransf("q", pca).treinar([(F, i) for F, _, i in Ftr]),
           "PCA": pca}
    est_z = {"IF_M2": cm.IFStat().treinar([(Z, i) for Z, _, i in Ztr])}
    ref, lim = {}, {}
    for k, e in est.items():
        st = [e.stat(F, i) for F, _, i in Fcal]
        ref[k] = np.vstack([s[ok] for s, ok in st])
        lim[k], _ = cm.calibrar([np.where(ok, cm.pvalor(ref[k], s, dentro=True), 0.0) for s, ok in st], cm.alarmes, n_sem)
    for k, e in est_z.items():
        st = [e.stat(Z, i) for Z, _, i in Zcal]
        ref[k] = np.vstack([s[ok] for s, ok in st])
        lim[k], _ = cm.calibrar([np.where(ok, cm.pvalor(ref[k], s, dentro=True), 0.0) for s, ok in st], cm.alarmes, n_sem)
    print(f"calibracao: {time.time() - t0:.0f} s", flush=True)

    def alarmes_df(df):
        F, t, inicio, _ = blocos(df)
        out_ = {}
        for k, e in est.items():
            s, ok = e.stat(F, inicio)
            out_[k] = [t[i] for i in cm.alarmes(np.where(ok, cm.pvalor(ref[k], s), 0.0), lim[k])]
        Z, tz, iz = zinfo(df)
        for k, e in est_z.items():
            s, ok = e.stat(Z, iz)
            out_[k] = [tz[i] for i in cm.alarmes(np.where(ok, cm.pvalor(ref[k], s), 0.0), lim[k])]
        return out_, F, t

    fa = {k: 0 for k in NOMES}
    for s in normais:
        for k, a in alarmes_df(simular_mf2(cm.HORAS_NORMAL, seed=s))[0].items():
            fa[k] += len(a)

    det = {k: {} for k in NOMES}
    dentro = {}
    for falha, rolo in cm.CASOS:
        n_in = n_tot = 0
        for s in falhas:
            a, F, t = alarmes_df(simular_mf2(6, seed=s, falha=falha, rolo=rolo))
            for k in NOMES:
                fa[k] += sum(x < cm.INICIO_S for x in a[k])
                det[k][(falha, rolo)] = det[k].get((falha, rolo), 0) + any(x >= cm.INICIO_S for x in a[k])
            seg = t >= cm.INICIO_S + 2.5 * 3600                  # segunda metade da rampa em diante
            n_in += int(((F[seg] >= lo) & (F[seg] <= hi)).all(1).sum())
            n_tot += int(seg.sum())
        dentro[(falha, rolo)] = n_in / max(n_tot, 1)
        print(f"  {falha} {rolo}: " + ", ".join(f"{k} {det[k][(falha, rolo)]}" for k in NOMES)
              + f"; blocos dentro da faixa {dentro[(falha, rolo)]:.0%}", flush=True)

    horas_fa = cm.HORAS_NORMAL * len(normais) + len(cm.CASOS) * len(falhas)
    linhas = []
    for k in NOMES:
        por = {f: sum(v for (ff, _), v in det[k].items() if ff == f) for f in ("deriva_ganho", "atrito", "perda_pulsos", "ruido")}
        linhas.append({"metodo": NOMES[k], **por, "total": sum(por.values()),
                       "fa_semana": round(fa[k] / horas_fa * 168, 2)})
    R = pd.DataFrame(linhas)
    D = pd.DataFrame([{"falha": f, "rolo": r, "blocos com falha dentro da faixa normal": f"{v:.0%}"}
                      for (f, r), v in dentro.items()])
    texto = ["# Diagnóstico do Isolation Forest (simulador de malha fechada v2)", "",
             "Gerado por `diagnostico_isolation_forest.py`. Mesmo protocolo e sementes de `comparacao_metodos.py` "
             f"({len(cm.CASOS) * len(falhas)} cenários; alarmes falsos em {horas_fa} h).", "",
             R.to_markdown(index=False), "",
             "Blocos de 60 s na segunda metade da rampa da falha com **todos** os atributos dentro da faixa "
             "[mín, máx] do treino:", "", D.to_markdown(index=False), "", f"Tempo: {time.time() - t0:.0f} s."]
    out.write_text("\n".join(texto), encoding="utf-8")
    print("\n".join(texto[:5]))


if __name__ == "__main__":
    main()
