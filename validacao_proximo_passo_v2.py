"""
Validacao do Modelo 3 v2 ("prever o proximo passo": ARX + branqueamento + limite por meta de FA).

Etapas (cada uma grava uma secao em validacao_proximo_passo_v2.md):
  python validacao_proximo_passo_v2.py dev        # cada estatistica sozinha (so desenvolvimento)
  python validacao_proximo_passo_v2.py dev-final  # configuracao escolhida (CONFIG_FINAL) nas sementes de
                                                  # desenvolvimento, com a comparacao Isolation Forest
  python validacao_proximo_passo_v2.py teste      # mede UMA vez, configuracao congelada (CONFIG_FINAL),
                                                  # tambem com a comparacao Isolation Forest
  python validacao_proximo_passo_v2.py bases      # SKAB (protocolo oficial) e PMSM

Separacao das sementes (duas simulacoes: poc_deteccao.simular SEM malha e malha_fechada.simular_mf COM malha):
  treino               semente 1 (8 h normal)
  calibracao do limite sementes 2-6 (24 h normais cada; bootstrap em blocos)
  desenvolvimento      falhas sementes 11-20 (6 h), normal sementes 31-35 (48 h)
  TESTE (uma vez)      falhas sementes 101-110 (6 h), normal sementes 91-95 (48 h)
O indicador atual ('proprio' de monitor.py) e treinado/calibrado como em monitor.py (sementes 1 e 2).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from monitor import MonitorEncoder, de_simulacao
from monitor.proximo_passo_v2 import (ESTATISTICAS, ProximoPassoV2, de_dataframe, eventos, ljung_box,
                                      limiar_bootstrap, limiar_contagem)
from poc.malha_fechada import simular_mf
from poc.poc_deteccao import simular

FS = 10
F0 = int(3600 * FS)
FALHAS = ["perda_pulsos", "ruido", "deriva_ganho", "erros_com"]
SEM_CAL = [2, 3, 4, 5, 6]
DEV_FALHA, DEV_NORMAL = range(11, 21), [31, 32, 33, 34, 35]
TESTE_FALHA, TESTE_NORMAL = range(101, 111), [91, 92, 93, 94, 95]
OUT = Path(__file__).with_name("validacao_proximo_passo_v2.md")

SIMS = {
    "sem malha (poc_deteccao)": lambda h, s, f=None: de_simulacao(simular(h, seed=s, falha=f)),
    "com malha (malha_fechada)": lambda h, s, f=None: simular_mf(h, seed=s, falha=f),
}

# grade avaliada no desenvolvimento (cada estatistica sozinha, meta 1 FA/semana)
GRADE = [
    ("ewma_abs", {"lam": 0.05}), ("ewma_abs", {"lam": 0.01}),
    ("ewma_sq", {"lam": 0.05}), ("ewma_sq", {"lam": 0.01}), ("ewma_sq", {"lam": 0.002}),
    ("cusum_sq", {"k": 0.25}), ("cusum_sq", {"k": 0.5}), ("cusum_sq", {"k": 1.0}),
    ("var_jan", {"W": 100}), ("var_jan", {"W": 600}), ("var_jan", {"W": 3000}),
    ("kurt_jan", {"W": 600}), ("kurt_jan", {"W": 3000}),
    ("contagem", {"W": 600, "k_pico": 5.0}), ("contagem", {"W": 6000, "k_pico": 4.0}),
    ("contagem", {"W": 6000, "k_pico": 5.0}), ("contagem", {"W": 6000, "k_pico": 6.0}),
]

# configuracao congelada depois do desenvolvimento (preenchida a partir de 'dev')
CONFIG_FINAL = {"var_jan": {"W": 3000}, "contagem": {"W": 6000, "k_pico": 5.0}}


def _rotulo(nome, par):
    return nome + "(" + ", ".join(f"{k}={v}" for k, v in par.items()) + ")"


def _celula(atrasos, n):
    return f"{len(atrasos)}/{n}" + (f" · {np.median(atrasos):.0f} min" if atrasos else "")


def _resumo(ev_normal, ev_falha, sementes, horas_normal):
    linha = {f"FA em {horas_normal:.0f} h": sum(len(e) for e in ev_normal)}
    antes = 0
    for f in FALHAS:
        atr = []
        for s in sementes:
            ev = np.array(ev_falha[(f, s)], int)
            antes += int((ev < F0).sum())
            d = ev[ev >= F0]
            if len(d):
                atr.append((d[0] - F0) / FS / 60)
        linha[f] = _celula(atr, len(sementes))
    linha["FA antes da falha"] = antes
    return linha


def _carregar(sim, normais, falhas):
    norm = {s: sim(48, s) for s in normais}
    fal = {(f, s): sim(6, s, f) for f in FALHAS for s in falhas}
    return norm, fal


def _baseline(sim, norm, fal):
    mon = MonitorEncoder("pb1", ["festoon", "pb2"], fs=FS).treinar(sim(8, 1)).calibrar(sim(8, 2))

    def ev(df):
        return [e["indice"] for e in mon.analisar(df)[1] if e["indicador"] == "proprio"]
    return [ev(df) for df in norm.values()], {k: ev(df) for k, df in fal.items()}


def _modelo(sim, estatisticas, portao_s=20.0):
    m = ProximoPassoV2(estatisticas=estatisticas, portao_s=portao_s).treinar(*de_dataframe(sim(8, 1)))
    return m, [de_dataframe(sim(24, s)) for s in SEM_CAL]


def _limiar(m, zcal, nome, par):
    if nome == "contagem":
        return limiar_contagem(zcal, par["W"], par["k_pico"], FS, fa_semana=1.0)[0]
    return limiar_bootstrap(zcal, nome, par, m._persist(nome), m.holdoff, FS, fa_semana=1.0,
                            n_rep=m.n_rep, conf=m.conf)[0]


GRADE_SEM_PORTAO = [("ewma_abs", {"lam": 0.01}), ("ewma_sq", {"lam": 0.01}), ("var_jan", {"W": 100}),
                    ("contagem", {"W": 6000, "k_pico": 5.0})]


def etapa_sim(normais, falhas):
    texto = []
    for nome_sim, sim in SIMS.items():
        t0 = time.time()
        norm, fal = _carregar(sim, normais, falhas)
        linhas = [{"detector": "ATUAL: proprio de monitor.py (AR(5), EWMA|e|, 1,1×máx)",
                   **_resumo(*_baseline(sim, norm, fal), falhas, 48 * len(normais))}]
        for portao_s, grade in [(20.0, GRADE), (None, GRADE_SEM_PORTAO)]:
            m, cal = _modelo(sim, dict([grade[0]]), portao_s)
            zcal = [m.z(*c)[0] for c in cal]
            zn = [m.z(*de_dataframe(df)) for df in norm.values()]
            zf = {k: m.z(*de_dataframe(df)) for k, df in fal.items()}
            if portao_s:
                horas_mon = sum(len(z) for z in zcal) / FS / 3600
                rho = ljung_box(np.concatenate(zcal), 30)[2]
                cab = (f"ARX escolhido por BIC: p = {m.ordem[0]} atrasos do sinal, q = {m.ordem[1]} termos de setpoint "
                       f"(sp e {m.ordem[1]} atrasos da derivada); branqueamento AR({len(m.phi)}); "
                       f"σ do resíduo branqueado = {m.sigma:.4f} m/min. "
                       f"Ljung-Box no treino: Q = {m.lb_antes[0]:.0f} antes → {m.lb_depois[0]:.0f} depois; "
                       f"na calibração, máx |ρ(k)|, k ≤ 30 = {np.abs(rho).max():.3f}. "
                       f"Calibração: {horas_mon:.0f} h monitoradas (portão aberto) das sementes {SEM_CAL[0]}–{SEM_CAL[-1]}.")
            for nome, par in grade:
                pers = m._persist(nome)
                lim = _limiar(m, zcal, nome, par)

                def ev(zi):
                    z, idx = zi
                    return eventos(ESTATISTICAS[nome](z, **par) > lim, pers, m.holdoff, idx)
                rot = "v2 " + _rotulo(nome, par) + ("" if portao_s else " SEM portão")
                linhas.append({"detector": rot, "limite": round(lim, 3),
                               **_resumo([ev(x) for x in zn], {k: ev(x) for k, x in zf.items()}, falhas,
                                         48 * len(normais))})
        texto += [f"### {nome_sim}", "", cab, "", pd.DataFrame(linhas).to_markdown(index=False), ""]
        print(f"  {nome_sim}: {time.time() - t0:.0f} s", flush=True)
    return texto


def _janelas_if(z, idx, W=30 * FS, S=10 * FS):
    """Features por janela de 30 s (passo 10 s) do residuo branqueado z: as mesmas grandezas do modelo."""
    F, fim = [], []
    for e in range(W, len(z) + 1, S):
        s = z[e - W:e]
        sd = s.std() + 1e-12
        F.append([s.mean(), sd, np.abs(s).max(), ((s - s.mean()) ** 4).mean() / sd ** 4,
                  (np.abs(s) > 4).sum(), np.abs(np.diff(s)).mean()])
        fim.append(idx[e - 1])
    return np.array(F), np.array(fim, int)


class IFResiduo:
    """Isolation Forest nas features de janela do residuo do Modelo 3 v2 (comparacao justa).
    Limite = menor score na calibracao - 0,02 e 2 janelas seguidas (mesma regra de poc_deteccao.D1b)."""

    def __init__(self, m):
        from sklearn.ensemble import IsolationForest
        self.m, self.IF = m, IsolationForest(n_estimators=200, random_state=0)

    def fit(self, treino, cal):
        self.IF.fit(_janelas_if(*self.m.z(*treino))[0])
        self.limiar = min(self.IF.score_samples(_janelas_if(*self.m.z(*c))[0]).min() for c in cal) - 0.02
        return self

    def alarmes(self, serie):
        F, fim = _janelas_if(*self.m.z(*serie))
        ruim = self.IF.score_samples(F) < self.limiar
        dois = ruim & np.concatenate([[False], ruim[:-1]])
        return eventos(dois, 1, self.m.holdoff, fim)


def etapa_final(normais, falhas, com_if=True):
    """Configuracao congelada (CONFIG_FINAL): alarme quando QUALQUER estatistica passa (meta dividida)."""
    texto = []
    for nome_sim, sim in SIMS.items():
        t0 = time.time()
        norm, fal = _carregar(sim, normais, falhas)
        treino = de_dataframe(sim(8, 1))
        m, cal = _modelo(sim, CONFIG_FINAL)
        m.calibrar(cal)
        b_norm, b_fal = _baseline(sim, norm, fal)
        series_n = [de_dataframe(df) for df in norm.values()]
        series_f = {k: de_dataframe(df) for k, df in fal.items()}

        def ev(serie, quais=None):
            if quais is None:
                return [a["indice"] for a in m.analisar(*serie)[2]]
            est, z, idx = m.estatisticas_de(*serie)
            (nome,) = quais
            return eventos(est[nome] > m.limiar[nome], m._persist(nome), m.holdoff, idx)
        h = 48 * len(normais)
        linhas = [{"detector": "ATUAL: proprio de monitor.py", **_resumo(b_norm, b_fal, falhas, h)}]
        for rot, quais in [("v2 completo (" + " OU ".join(_rotulo(k, v) for k, v in CONFIG_FINAL.items()) + ")", None)] +                 [(f"  só {_rotulo(k, v)} (limite do combinado)", {k}) for k, v in CONFIG_FINAL.items()]:
            linhas.append({"detector": rot, **_resumo([ev(x, quais) for x in series_n],
                                                      {k: ev(x, quais) for k, x in series_f.items()}, falhas, h)})
        if com_if:
            iff = IFResiduo(m).fit(treino, cal)
            linhas.append({"detector": "Isolation Forest nas features do resíduo v2 (janelas 30 s)",
                           **_resumo([iff.alarmes(x) for x in series_n],
                                     {k: iff.alarmes(x) for k, x in series_f.items()}, falhas, h)})
        lims = "; ".join(f"{k}: {v:.3f} (réplicas {min(m.limiar_reps[k]):.3f}–{max(m.limiar_reps[k]):.3f}; "
                         f"máx na calibração {m.max_calib[k]:.3f})" for k, v in m.limiar.items())
        texto += [f"### {nome_sim}", "",
                  f"ARX p = {m.ordem[0]}, q = {m.ordem[1]}, branqueamento AR({len(m.phi)}), σ = {m.sigma:.4f} m/min. "
                  f"Limites (meta 1 FA/semana no total, metade para cada estatística): {lims}.", "",
                  pd.DataFrame(linhas).to_markdown(index=False), ""]
        print(f"  {nome_sim}: {time.time() - t0:.0f} s", flush=True)
    return texto


# ---------------------------------------------------------------- bases publicas
# Variantes DECLARADAS ANTES de rodar (nao ajustadas nas bases). Mesmo protocolo dos scripts em bases/:
# dados padronizados com o treino; limite = maior valor no treino; mediana movel de 3 amostras.
# Sem setpoint nas bases: o ARX vira AR por sensor (ordem por BIC, ate 10) + branqueamento AR (BIC, ate 10).
def _z_por_sensor(Xtr, Xte):
    from monitor.proximo_passo_v2 import ProximoPassoV2
    Ztr, Zte = np.zeros_like(Xtr), np.zeros_like(Xte)
    for j in range(Xtr.shape[1]):
        m = ProximoPassoV2(fs=1, p_max=10, q_max=0, r_max=10, portao_s=None)
        m.treinar(Xtr[:, j], None, np.ones(len(Xtr), bool))
        Ztr[:, j] = m.z(Xtr[:, j], None, np.ones(len(Xtr), bool))[0]
        Zte[:, j] = m.z(Xte[:, j], None, np.ones(len(Xte), bool))[0]
    return Ztr, Zte


def stat_v2_ewma(Xtr, Xte):        # "v2 + EWMA": media de z^2 entre sensores (EWMA aplicada pelo protocolo)
    a, b = _z_por_sensor(Xtr, Xte)
    return (a ** 2).mean(1), (b ** 2).mean(1)


def _ou(s1, s2):
    """OU de dois criterios 'passou do maximo do treino' num score so (>1 = alarme)."""
    (a1, b1), (a2, b2) = s1, s2
    n1, n2 = max(a1.max(), 1e-12), a2.max() + 0.5
    return np.maximum(a1 / n1, (a2 + 0.5) / n2), np.maximum(b1 / n1, (b2 + 0.5) / n2)


def stat_v2_completo(Xtr, Xte, fs):
    """Mesma regra do modelo final: variancia do residuo em 300 s OU contagem de |z| > 5 em 600 s
    (media/soma entre sensores). Calculado em linha sobre treino seguido do teste (janelas causais)."""
    a, b = _z_por_sensor(Xtr, Xte)
    Z, n = np.vstack([a, b]), len(a)
    sv = np.column_stack([ESTATISTICAS["var_jan"](Z[:, j], W=int(300 * fs)) for j in range(Z.shape[1])]).mean(1)
    sc = sum(ESTATISTICAS["contagem"](Z[:, j], W=int(600 * fs), k_pico=5.0) for j in range(Z.shape[1]))
    # no treino so contam as janelas ja completas (aquecimento devolve 0)
    return _ou((sv[:n], sv[n:]), (sc[:n], sc[n:]))


def etapa_bases():
    sys.path.insert(0, str(Path(__file__).parent / "bases"))
    import testar_skab as sk
    texto = ["## Bases públicas", "",
             "Variantes declaradas antes de rodar; limite = maior valor no treino (regra dos scripts em `bases/`), "
             "porque 400 amostras de treino não permitem calibrar meta de FA em tempo.", ""]
    arquivos = sk.carregar()
    linhas = []
    for rot, fn, suav in [("ATUAL: D1 AR(5) por sensor + EWMA (testar_skab.py)", sk.stat_d1, True),
                          ("v2: AR por BIC + branqueamento + EWMA de z²", stat_v2_ewma, True),
                          ("v2 completo: var. 300 s OU contagem |z|>5 em 600 s", lambda a, b: stat_v2_completo(a, b, 1), False),
                          ("Isolation Forest (leaderboard SKAB)", None, None)]:
        if fn is None:
            linhas.append({"detector": rot, "F1": 0.29, "FAR %": "-", "MAR %": "-"})
            continue
        (f1, far, mar), _ = sk.avaliar(fn, arquivos, suav)
        linhas.append({"detector": rot, "F1": round(f1, 2), "FAR %": round(far, 1), "MAR %": round(mar, 1)})
    texto += [f"### SKAB ({len(arquivos)} experimentos, protocolo oficial: treino = 400 primeiras amostras)", "",
              pd.DataFrame(linhas).to_markdown(index=False), ""]

    # PMSM, como em bases/testar_pmsm.py
    from sklearn.preprocessing import StandardScaler
    import testar_pmsm as pm
    df = pd.read_csv(pm.ARQ)
    normal = df[df.FDD == "F0"][pm.SINAIS].to_numpy(float)
    meio = len(normal) // 2
    tr, val = normal[:meio], normal[meio:]
    sc = StandardScaler().fit(tr)
    linhas = []
    for rot, fn, suav in [("ATUAL: D1 AR(5) por sensor + EWMA", sk.stat_d1, True),
                          ("v2: AR por BIC + branqueamento + EWMA de z²", stat_v2_ewma, True),
                          ("v2 completo: var. 300 s OU contagem |z|>5 em 600 s", lambda a, b: stat_v2_completo(a, b, 10), False)]:
        def pontuar(X):
            s_tr, s = fn(sc.transform(tr), sc.transform(X))
            if suav:
                s_tr = pd.Series(s_tr).ewm(alpha=0.2).mean().to_numpy()
                s = pd.Series(s).ewm(alpha=0.2).mean().to_numpy()
            p = (s > s_tr.max()).astype(int)
            return pd.Series(p).rolling(3, min_periods=1).median().to_numpy().astype(bool)
        linha = {"detector": rot, "FA % (normal não visto)": round(100 * pontuar(val).mean(), 1)}
        det = []
        for c in pm.DESC:
            p = pontuar(df[df.FDD == c][pm.SINAIS].to_numpy(float))
            linha[c] = f"{100 * p.mean():.0f}%"
            det.append(p.any())
        linha["falhas detectadas"] = f"{sum(det)}/8"
        linhas.append(linha)
    texto += ["### PMSM (inversor, 10 Hz; treino = 1ª metade do normal)", "",
              "Célula = % das amostras da falha sinalizadas. " + "; ".join(f"{k} = {v}" for k, v in pm.DESC.items()), "",
              pd.DataFrame(linhas).to_markdown(index=False), ""]
    return texto


def gravar(secao, linhas):
    atual = OUT.read_text(encoding="utf-8") if OUT.exists() else "# Validação do Modelo 3 v2 (prever o próximo passo)\n"
    marca = f"<!-- {secao} -->"
    partes = atual.split(marca)
    novo = marca + "\n" + "\n".join(linhas) + "\n" + marca
    atual = partes[0] + novo + partes[2] if len(partes) == 3 else atual.rstrip() + "\n\n" + novo + "\n"
    OUT.write_text(atual, encoding="utf-8")
    print("\n".join(linhas))


def main():
    etapa = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if etapa == "dev":
        gravar("dev", ["## Desenvolvimento (sementes de falha 11–20, normais 31–35 × 48 h)", "",
                       "Cada estatística sozinha, limite por bootstrap em blocos com meta de 1 FA por semana monitorada.",
                       ""] + etapa_sim(DEV_NORMAL, DEV_FALHA))
    elif etapa == "dev-final":
        gravar("dev-final", ["## Desenvolvimento — configuração escolhida", ""] + etapa_final(DEV_NORMAL, DEV_FALHA))
    elif etapa == "teste":
        gravar("teste", ["## TESTE (rodado uma vez; falhas 101–110, normais 91–95 × 48 h)", ""]
               + etapa_final(TESTE_NORMAL, TESTE_FALHA))
    elif etapa == "bases":
        gravar("bases", etapa_bases())
    else:
        raise SystemExit(f"etapa desconhecida: {etapa}")


if __name__ == "__main__":
    main()
