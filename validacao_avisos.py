"""
Validacao do Modelo 1 (avisos e contadores do drive) -> validacao_avisos.md

Protocolo (separacao feita antes de olhar qualquer resultado):
    desenvolvimento  cenarios de degradacao: sementes 1-30   (cenario = CENARIOS[semente % 3])
                     operacao normal:        sementes 31-40  (4 semanas cada)
    teste (UMA vez)  cenarios: sementes 101-110;  normal: sementes 91-95 (4 semanas cada)
Cada simulacao traz 14 dias de historico normal (treino, Fase I) e o periodo monitorado.
Os parametros do monitor sao escolhidos SO no desenvolvimento; o teste roda com eles fixos.

Uso:  python validacao_avisos.py            (~5 min)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from monitor.avisos import FAMILIAS, MonitorAvisos, RegraAtual  # noqa: E402
from poc.simulador_avisos import CENARIOS, ENCODERS, HIP, contadores, eventos_do_log, simular  # noqa: E402

DEV_CEN, DEV_NORM = range(1, 31), range(31, 41)
TESTE_CEN, TESTE_NORM = range(101, 111), range(91, 96)
META = 1.0                     # falso alarme por semana, linha inteira (10 encoders)
GRADE = [dict(razao=r, fator_seguranca=f) for r in (2, 4, 8) for f in (1, 3)]
SEMANA = 7 * 86400.0
_cache = {}


def dados(seed, normal):
    if seed not in _cache:
        df, info = simular(seed, None if normal else CENARIOS[seed % 3])
        _cache[seed] = (eventos_do_log(df), contadores(df, 60.0, info["t_fim_s"]), info)
    return _cache[seed]


def rodar(metodo, seed, normal):
    ev, cont, info = dados(seed, normal)
    th = info["t_fim_hist_s"]
    ev_tr, cont_tr = ev[ev["t_s"] < th], cont[cont["t_s"] < th]
    if metodo["tipo"] == "cusum":
        m = MonitorAvisos(meta_fa_semana=META, **metodo["param"]).treinar(ev_tr, th, ENCODERS, cont_tr)
        alarmes, _ = m.analisar(ev, th, info["t_fim_s"], cont)
    else:
        m = RegraAtual(familias=metodo["familias"]).treinar(ev_tr, list(ENCODERS), cont_tr)
        alarmes = m.analisar(ev, th, info["t_fim_s"], cont)
    return alarmes, info


def avaliar(metodo, sem_cen, sem_norm):
    linhas, fa_norm, semanas = [], 0, 0.0
    for s in sem_norm:
        alarmes, info = rodar(metodo, s, True)
        fa_norm += len(alarmes)
        semanas += (info["t_fim_s"] - info["t_fim_hist_s"]) / SEMANA
    for s in sem_cen:
        alarmes, info = rodar(metodo, s, False)
        ti, tf, enc = info["t_inicio_s"], info["t_falha_s"], info["encoder"]
        certos = [a for a in alarmes if a["encoder"] == enc and ti <= a["t_s"] < tf]
        fa = [a for a in alarmes if a["t_s"] < tf and a not in certos]
        lead = (tf - certos[0]["t_s"]) / 3600 if certos else np.nan
        linhas.append({"semente": s, "cenario": info["cenario"], "encoder": enc,
                       "duracao_h": (tf - ti) / 3600, "detectou": bool(certos), "antecedencia_h": lead,
                       "fa_no_cenario": len(fa),
                       "mensagem": certos[0]["mensagem"] if certos else ""})
    det = pd.DataFrame(linhas)
    return {"fa_semana": fa_norm / semanas, "fa_total": fa_norm, "semanas": semanas,
            "det": int(det["detectou"].sum()), "n": len(det),
            "lead_med_h": float(det["antecedencia_h"].median()),
            "lead_min_h": float(det["antecedencia_h"].min()),
            "fa_cen": int(det["fa_no_cenario"].sum()), "tabela": det}


def fmt_h(h):
    if np.isnan(h):
        return "-"
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"


def linha_resumo(nome, r):
    return (f"| {nome} | {r['fa_semana']:.2f} ({r['fa_total']} em {r['semanas']:.0f} sem.) | "
            f"{r['det']}/{r['n']} | {fmt_h(r['lead_med_h'])} | {fmt_h(r['lead_min_h'])} | {r['fa_cen']} |")


def main():
    t0 = time.time()
    out = ["# Validação do Modelo 1 — avisos e contadores do drive", "",
           "Gerado por `validacao_avisos.py`. Dados **simulados** por `poc/simulador_avisos.py`; "
           "todos os parâmetros da simulação são hipóteses.", "",
           f"Linha simulada: {sum(d == 'tipo_a' for d in ENCODERS.values())} encoders tipo_a + "
           f"{sum(d == 'tipo_b' for d in ENCODERS.values())} tipo_b. Meta: ≤ {META:g} falso alarme por semana "
           "**na linha inteira**. Treino: 14 dias normais antes do período monitorado.", "",
           "Colunas: falsos alarmes por semana em operação normal (sementes de operação normal); "
           "detecção = 1º alarme no encoder certo entre o início da degradação e a falha final; "
           "antecedência = falha final − 1º alarme; FA nos cenários = alarmes em outro encoder ou antes "
           "do início da degradação.", ""]
    cab = ["| Método | FA/semana (normal) | Detecção | Antecedência mediana | Antecedência mínima | FA nos cenários |",
           "|---|---|---|---|---|---|"]

    # ---------------- desenvolvimento
    out += ["## 1. Desenvolvimento (sementes 1–30 cenários, 31–40 normal)", "", *cab]
    resultados = []
    for p in GRADE:
        r = avaliar({"tipo": "cusum", "param": p}, DEV_CEN, DEV_NORM)
        resultados.append((p, r))
        out.append(linha_resumo(f"CUSUM razão {p['razao']}, fator de segurança {p['fator_seguranca']}", r))
        print(p, r["fa_semana"], r["det"], r["lead_med_h"], f"{time.time() - t0:.0f}s", flush=True)
    atuais = {"Atual: erros de comunicação, 1 h, máx+2": {"tipo": "atual", "familias": ("cabo",)},
              "Atual estendida: todos os avisos, 1 h, máx+2": {"tipo": "atual", "familias": FAMILIAS}}
    for nome, m in atuais.items():
        out.append(linha_resumo(nome, avaliar(m, DEV_CEN, DEV_NORM)))
    ok = [(p, r) for p, r in resultados if r["fa_semana"] <= META] or resultados
    escolhido, _ = max(ok, key=lambda pr: (pr[1]["det"], pr[1]["lead_med_h"], -pr[1]["fa_semana"]))
    out += ["", f"**Escolha (regra fixada antes):** entre as configurações com FA/semana ≤ {META:g} no "
            f"desenvolvimento, a de mais detecções e, no empate, maior antecedência mediana → "
            f"razão {escolhido['razao']}, fator de segurança {escolhido['fator_seguranca']}.", ""]

    # ---------------- teste (uma vez)
    out += ["## 2. Teste — medido uma vez (sementes 101–110 cenários, 91–95 normal)", "", *cab]
    r_teste = avaliar({"tipo": "cusum", "param": escolhido}, TESTE_CEN, TESTE_NORM)
    out.append(linha_resumo("**Modelo 1 (CUSUM de Poisson)**", r_teste))
    r_atual = {}
    for nome, m in atuais.items():
        r_atual[nome] = avaliar(m, TESTE_CEN, TESTE_NORM)
        out.append(linha_resumo(nome, r_atual[nome]))
    out += ["", "### Detalhe por cenário (teste)", "",
            "| Semente | Cenário | Encoder | Duração da degradação | Modelo 1: antecedência | Atual (erros): antecedência | Atual estendida: antecedência |",
            "|---|---|---|---|---|---|---|"]
    ta = list(r_atual.values())
    for i, row in r_teste["tabela"].iterrows():
        a1, a2 = ta[0]["tabela"].iloc[i], ta[1]["tabela"].iloc[i]
        out.append(f"| {row['semente']} | {row['cenario']} | {row['encoder']} | {fmt_h(row['duracao_h'])} | "
                   f"{fmt_h(row['antecedencia_h']) if row['detectou'] else 'não detectou'} | "
                   f"{fmt_h(a1['antecedencia_h']) if a1['detectou'] else 'não detectou'} | "
                   f"{fmt_h(a2['antecedencia_h']) if a2['detectou'] else 'não detectou'} |")
    exemplos = r_teste["tabela"]
    exemplos = exemplos[exemplos["detectou"]].drop_duplicates("cenario")
    out += ["", "### Exemplos de mensagem (teste)", ""]
    out += [f"- *{m}*" for m in exemplos["mensagem"]]
    out += ["", "## 3. Hipóteses da simulação", "", "| Parâmetro | Valor |", "|---|---|"]
    out += [f"| `{k}` | {v} |" for k, v in HIP.items()]
    out += ["", f"Tempo de execução: {time.time() - t0:.0f} s."]
    Path(__file__).with_name("validacao_avisos.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"escrito validacao_avisos.md ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
