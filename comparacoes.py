"""
Comparacoes complementares do monitor de encoders.

  A  curva linha de corte x (falso alarme, deteccao, atraso)
  B  atributo fisico (residuo vs vizinhos) contra Isolation Forest nos sinais brutos
  C  classificador supervisionado (Random Forest, class_weight
     balanced) treinado com falhas SINTETICAS - funciona na falha que viu, e na que nao viu?

Teste sempre em cenarios nunca usados no desenvolvimento (sementes 101-110) e em
240 h de operacao normal (5 x 48 h). Gera comparacoes.md.

    python comparacoes.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from monitor import MonitorEncoder, de_simulacao
from poc.poc_deteccao import simular

FS = 10
F0 = int(3600 * FS)
FALHAS = ["perda_pulsos", "ruido", "deriva_ganho", "erros_com"]
TESTE = range(101, 111)
TREINO_FALHAS = range(201, 206)          # sementes so para o classificador supervisionado
NORMAL_SEEDS = [91, 92, 93, 94, 95]      # 5 x 48 h = 240 h normais
HOLDOFF = 10 * 60 * FS


def carregar():
    treino = de_simulacao(simular(8, seed=1))
    val = de_simulacao(simular(8, seed=2))
    normais = [de_simulacao(simular(48, seed=s)) for s in NORMAL_SEEDS]
    teste = {(f, s): simular(6, seed=s, falha=f) for f in FALHAS for s in TESTE}
    return treino, val, normais, teste


def resumo(eventos_normais, eventos_falha):
    """eventos_normais: lista de listas de indices; eventos_falha: {(f,s): lista de indices}"""
    horas = 48 * len(eventos_normais)
    fa = sum(len(e) for e in eventos_normais)
    linha = {"FA em 240 h": fa}
    total = 0
    for f in FALHAS:
        atrasos = [(min(i for i in ev if i >= F0) - F0) / FS / 60
                   for (ff, s), ev in eventos_falha.items() if ff == f and any(i >= F0 for i in ev)]
        total += len(atrasos)
        linha[f] = f"{len(atrasos)}/10" + (f" · {np.median(atrasos):.0f} min" if atrasos else "")
    linha["detectadas"] = f"{total}/40"
    linha["_fa_h"] = fa / horas
    return linha


# ------------------------------------------------------------------ janelas
def janelas(df, mon, W=30 * FS, S=10 * FS):
    ind = mon._indicadores(df)
    rod = ind["rodando"]
    z = np.where(rod, (ind["medida"] - ind["estimada"]) / mon.sigma_viz, 0.0)
    bruto = df[["v_setpoint", "v_festoon", "v_pb1", "v_pb2"]].to_numpy()
    Fz, Fb, fim = [], [], []
    for e in range(W, len(z) + 1, S):
        if rod[e - W:e].mean() < 0.9:
            continue
        s, b = z[e - W:e], bruto[e - W:e]
        Fz.append([s.mean(), s.std(), s.min(), s.max(), (s < -4).mean(),
                   np.abs(np.diff(s)).mean(), ind["erros"][e - 1]])
        Fb.append(np.concatenate([b.mean(0), b.std(0), b.min(0), b.max(0),
                                  np.abs(np.diff(b, axis=0)).mean(0)]))
        fim.append(e - 1)
    return np.array(Fz), np.array(Fb), np.array(fim)


def eventos_de_janelas(flag, fim):
    dois = flag & np.concatenate([[False], flag[:-1]])
    out, ultimo = [], -HOLDOFF
    for i in fim[dois]:
        if i - ultimo >= HOLDOFF:
            out.append(int(i))
            ultimo = i
    return out


# ------------------------------------------------------------------ A
def licao_a(treino, val, normais, teste):
    linhas = []
    for margem in [0.8, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0, 3.0]:
        m = MonitorEncoder("pb1", ["festoon", "pb2"], fs=FS, margem=margem).treinar(treino).calibrar(val)
        ev_n = [[e["indice"] for e in m.analisar(n)[1]] for n in normais]
        ev_f = {k: [e["indice"] for e in m.analisar(de_simulacao(df))[1]] for k, df in teste.items()}
        linhas.append({"margem sobre o máx. normal": f"{margem:.1f}x", **resumo(ev_n, ev_f)})
    return linhas


# ------------------------------------------------------------------ B e C
def licao_b_c(treino, val, normais, teste):
    mon = MonitorEncoder("pb1", ["festoon", "pb2"], fs=FS).treinar(treino).calibrar(val)
    jan = {"treino": janelas(treino, mon), "val": janelas(val, mon)}
    jan_n = [janelas(n, mon) for n in normais]
    jan_t = {k: janelas(de_simulacao(df), mon) for k, df in teste.items()}

    # B - nao supervisionado: mesmo algoritmo, atributos diferentes
    linhas_b = []
    for nome, col in [("Isolation Forest nos sinais BRUTOS (automático)", 1),
                      ("Isolation Forest no resíduo vs vizinhos (domínio)", 0)]:
        iso = IsolationForest(n_estimators=200, random_state=0).fit(jan["treino"][col])
        corte = iso.score_samples(jan["val"][col]).min() - 0.02
        ev_n = [eventos_de_janelas(iso.score_samples(j[col]) < corte, j[2]) for j in jan_n]
        ev_f = {k: eventos_de_janelas(iso.score_samples(j[col]) < corte, j[2]) for k, j in jan_t.items()}
        linhas_b.append({"detector": nome, **resumo(ev_n, ev_f)})
    ev_n = [[e["indice"] for e in mon.analisar(n)[1]] for n in normais]
    ev_f = {k: [e["indice"] for e in mon.analisar(de_simulacao(df))[1]] for k, df in teste.items()}
    linhas_b.append({"detector": "Monitor (resíduo vs vizinhos + EWMA + contador)", **resumo(ev_n, ev_f)})

    # C - supervisionado com falhas sinteticas
    # normal extra so para o classificador (o monitor usa so 'treino')
    normal_extra = [janelas(de_simulacao(simular(8, seed=s)), mon)[0] for s in (301, 302)]
    faltas_cache = {(f, s): janelas(de_simulacao(df := simular(6, seed=s, falha=f)), mon)
                    + (df["severidade"].to_numpy(),)
                    for f in FALHAS for s in TREINO_FALHAS}

    def rotulado(falhas):
        X = [jan["treino"][0]] + normal_extra
        y = [np.zeros(len(x)) for x in X]
        for f in falhas:
            for s in TREINO_FALHAS:
                Fz, _, fim, sev = faltas_cache[(f, s)]
                sv = sev[fim]
                ok = (sv == 0) | (sv >= 0.3)     # descarta a zona ambigua (falha ainda invisivel)
                X.append(Fz[ok])
                y.append((sv[ok] >= 0.3).astype(float))
        return np.vstack(X), np.concatenate(y)

    linhas_c = []
    Xall, yall = rotulado(FALHAS)
    for corte_nome in ("0,5 (padrão)", "calibrado: 0 FA no normal de validação"):
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0).fit(Xall, yall)
        corte = 0.5 if corte_nome.startswith("0,5") else rf.predict_proba(jan["val"][0])[:, 1].max() + 1e-9
        ev_n = [eventos_de_janelas(rf.predict_proba(j[0])[:, 1] > corte, j[2]) for j in jan_n]
        ev_f = {k: eventos_de_janelas(rf.predict_proba(j[0])[:, 1] > corte, j[2]) for k, j in jan_t.items()}
        linhas_c.append({"cenário": f"viu os 4 tipos no treino · corte {corte_nome}", **resumo(ev_n, ev_f)})
    for fora in FALHAS:
        X, y = rotulado([f for f in FALHAS if f != fora])
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0).fit(X, y)
        corte = rf.predict_proba(jan["val"][0])[:, 1].max() + 1e-9
        ev_n = [eventos_de_janelas(rf.predict_proba(j[0])[:, 1] > corte, j[2]) for j in jan_n]
        ev_f = {k: eventos_de_janelas(rf.predict_proba(j[0])[:, 1] > corte, j[2]) for k, j in jan_t.items()}
        r = resumo(ev_n, ev_f)
        linhas_c.append({"cenário": f"NÃO viu '{fora}' no treino · corte calibrado",
                         "FA em 240 h": r["FA em 240 h"], "falha nunca vista": r[fora]})
    return linhas_b, linhas_c


def tabela(linhas):
    return pd.DataFrame(linhas).drop(columns=["_fa_h"], errors="ignore").fillna("").to_markdown(index=False)


def main():
    treino, val, normais, teste = carregar()
    a = licao_a(treino, val, normais, teste)
    b, c = licao_b_c(treino, val, normais, teste)
    texto = [
        "# Comparações complementares",
        "",
        "Teste: 40 cenários de falha nunca usados no desenvolvimento (10 por tipo) e 240 h normais.",
        "Célula = cenários detectados / 10 · atraso mediano desde o início da falha.",
        "",
        "## A · Curva da linha de corte",
        tabela(a),
        "",
        "## B · Atributo físico contra sinais brutos",
        tabela(b),
        "",
        "## C · Classificador supervisionado com falhas simuladas (Random Forest balanceado)",
        tabela(c),
    ]
    out = Path(__file__).with_name("comparacoes.md")
    out.write_text("\n".join(texto), encoding="utf-8")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
