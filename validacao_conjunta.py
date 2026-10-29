"""
Validacao CONJUNTA dos tres modelos v2 no MESMO simulador -> validacao_conjunta.md

    Modelo 1  avisos do drive          (monitor/avisos.py; razao 8, fator de seguranca 3)
    Modelo 2  torque v2.1 com isolacao (monitor/torque_v2.py; k = 3, blocos de 2 h, com tensao das pontas)
    Modelo 3  proximo passo v2         (monitor/proximo_passo_v2.py; um por rolo)
reunidos em monitor/conjunto.py (MonitorConjunto), rodando sobre poc/malha_fechada_v2.py.

Avisos: poc.simulador_avisos.simular_acoplado gera, para cada cenario, 14 dias de historico
normal (treino do Modelo 1) e o log do cenario, coerente com a falha (acoplamento documentado
na funcao; todas as taxas sao hipoteses). Linha: festoon e pb1 com drive tipo_a, pb2 tipo_b.

Sementes (escolhidas ANTES de rodar; nenhuma das de teste foi usada pelos agentes dos tres
modelos nem em desenvolvimento - 201-205 ficaram de fora porque outro estudo as usou):
    treino            mf2 1-2 (12 h cada)              (as do treino do Modelo 2)
    calibracao        mf2 3-8 (48 h cada)              (as da calibracao do Modelo 2 v2.1)
    desenvolvimento   falhas 41-45, normais 46-47 (48 h)
    TESTE (uma vez)   falhas 221-230, normais 196-200 (48 h); Modelo 1 sozinho tambem em
                      4 semanas de log normal por semente (simulador_avisos.simular, 196-200)
                      2a rodada (29/10/2026), depois da revisao do codigo. A 1a rodada usou
                      falhas 211-220 e normais 191-195; essas sementes foram reaproveitadas em
                      comparacao_metodos.py e ja foram vistas, por isso foram trocadas.
Cenarios de falha: 6 h, falha comeca em 1 h, rampa de 4 h (simular_mf2). Nos cenarios so de
avisos (erros_com, degradacao_sinal) a velocidade e o torque sao normais e a degradacao vai de
1 h ate a falha final em 6 h.

    python validacao_conjunta.py dev      (sementes de desenvolvimento)
    python validacao_conjunta.py teste    (sementes de teste; rodar UMA vez)
Cada etapa grava a sua secao em validacao_conjunta.md.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from monitor.conjunto import MODELOS, MonitorConjunto
from monitor.torque_v2 import MonitorTorqueV2
from poc.malha_fechada_v2 import simular_mf2
from poc.simulador_avisos import ENCODERS_LINHA, simular, simular_acoplado

FS = 10
INICIO_S = 3600.0
HORAS_NORMAL = 48
SEMANA_H = 168.0
TREINO, CAL = (1, 2), range(3, 9)
ETAPAS = {"dev": dict(falha=range(41, 46), normal=range(46, 48)),
          "teste": dict(falha=range(221, 231), normal=range(196, 201))}
CASOS = [("deriva_ganho", "festoon"), ("deriva_ganho", "pb1"), ("deriva_ganho", "pb2"),
         ("atrito", "festoon"), ("atrito", "pb1"), ("atrito", "pb2"),
         ("perda_pulsos", "pb1"), ("perda_pulsos", "pb2"),
         ("ruido", "pb1"), ("ruido", "pb2"),
         ("erros_com", "pb1"), ("degradacao_sinal", "festoon")]
SO_AVISOS = ("erros_com", "degradacao_sinal")
ROTULO = {"avisos": "M1 avisos", "torque": "M2 torque", "proximo_passo": "M3 próximo passo"}
OUT = Path(__file__).with_name("validacao_conjunta.md")


def cenario(seed, falha, rolo, horas=6):
    """(df no relogio do log, eventos, contadores, info)."""
    df = simular_mf2(horas, seed=seed, falha=None if falha in SO_AVISOS else falha, rolo=rolo)
    ev, cont, info = simular_acoplado(seed, df["t_s"].to_numpy(), df["severidade"].to_numpy(), falha, rolo,
                                      inicio_s=INICIO_S)
    df = df.assign(t_s=df["t_s"] + info["t0_s"])
    return df, ev, cont, info


def rodar(mc, base, seed, falha, rolo, horas=6):
    df, ev, cont, info = cenario(seed, falha, rolo, horas)
    t0 = info["t0_s"]
    mc.avisos = None
    mc.treinar_avisos(ev[ev["t_s"] < t0], t0, ENCODERS_LINHA, cont[cont["t_s"] < t0])
    eventos = mc.analisar(df, ev, cont, t_ini=t0, t_fim=info["t_fim_s"])
    extra = base.analisar(df)[1] if base is not None else []
    return eventos, extra, info


def correto(e, falha, rolo):
    if falha == "atrito":
        return e["modelo"] == "torque" and e["tipo"] == "mecanica" and e["rolo"] == rolo
    if e["modelo"] == "torque":
        return e["tipo"] == "encoder" and e["encoder"] == rolo
    return e["encoder"] == rolo


def limite_95(k, horas):
    """Limite superior unilateral 95 % (Poisson exato) da taxa, em eventos por semana."""
    return chi2.ppf(0.95, 2 * k + 2) / 2 / horas * SEMANA_H


def _cel(atrasos, n):
    return f"{len(atrasos)}/{n}" + (f" · {np.median(atrasos):.0f} min" if atrasos else "")


def avaliar_casos(mc, base, sementes):
    linhas_det, linhas_iso, pre, todos_pre = [], [], {m: 0 for m in MODELOS}, 0
    pre_sev = {"baixa": 0, "media": 0, "alta": 0}
    tot_det, tot_iso = 0, 0
    n = len(sementes)
    for falha, rolo in CASOS:
        atr = {m: [] for m in MODELOS + ("conjunto", "m2_sem_tensao")}
        primeiro = {m: 0 for m in MODELOS}
        iso1, iso_m2, conf_m2, errado, n_m2 = 0, 0, 0, 0, 0
        for s in sementes:
            ev, extra, info = rodar(mc, base, s, falha, rolo)
            t_on = info["t0_s"] + INICIO_S
            t_lim = info["t_falha_s"] or np.inf
            for e in ev:
                if e["t_s"] < t_on:
                    pre[e["modelo"]] += 1
                    pre_sev[e["severidade"]] += 1
                    todos_pre += 1
            dep = [e for e in ev if t_on <= e["t_s"] < t_lim]
            errado += sum(not correto(e, falha, rolo) for e in dep)
            for m in MODELOS:
                d = [e for e in dep if e["modelo"] == m]
                if d:
                    atr[m].append((d[0]["t_s"] - t_on) / 60)
            if dep:
                atr["conjunto"].append((dep[0]["t_s"] - t_on) / 60)
                primeiro[dep[0]["modelo"]] += 1
                iso1 += correto(dep[0], falha, rolo)
            d2 = [e for e in dep if e["modelo"] == "torque"]
            if d2:
                n_m2 += 1
                iso_m2 += correto(d2[0], falha, rolo)
                conf = [e["origem"]["confirmada"] for e in d2 if e["origem"].get("confirmada")]
                certo = ("mecanica" if falha == "atrito" else "encoder", rolo)
                conf_m2 += bool(conf) and conf[0][:2] == certo
            dx = [e for e in extra if t_on <= e["t_s"] < t_lim]
            if dx:
                atr["m2_sem_tensao"].append((dx[0]["t_s"] - t_on) / 60)
        caso = f"{falha} {rolo}"
        linhas_det.append({"caso": caso, "conjunto": _cel(atr["conjunto"], n),
                           **{ROTULO[m]: _cel(atr[m], n) for m in MODELOS},
                           "M2 sem tensão das pontas": _cel(atr["m2_sem_tensao"], n),
                           "1º a disparar": ", ".join(f"{ROTULO[m].split()[0]} {k}" for m, k in primeiro.items() if k)
                           or "-"})
        det = len(atr["conjunto"])
        tot_det, tot_iso = tot_det + det, tot_iso + iso1
        linhas_iso.append({"caso": caso, "isolação no 1º evento do conjunto": f"{iso1}/{det}",
                           "M2: 1º alarme → confirmada": f"{iso_m2}/{n_m2} → {conf_m2}/{n_m2}",
                           "eventos depois do início apontando outro encoder/causa": errado})
        print(f"  {caso}: {linhas_det[-1]['conjunto']}", flush=True)
    return linhas_det, linhas_iso, pre, pre_sev, todos_pre, (tot_det, tot_iso)


def avaliar_normais(mc, base, sementes):
    por_modelo, sev, m2b = {m: 0 for m in MODELOS}, {"baixa": 0, "media": 0, "alta": 0}, 0
    exemplos = []
    for s in sementes:
        ev, extra, _ = rodar(mc, base, s, None, "pb1", HORAS_NORMAL)
        for e in ev:
            por_modelo[e["modelo"]] += 1
            sev[e["severidade"]] += 1
            exemplos.append(e)
        m2b += len(extra)
    return por_modelo, sev, m2b, exemplos


def m1_longo(mc, sementes, dias=28.0):
    """Modelo 1 sozinho em 4 semanas de log normal por semente (mesma linha de 3 encoders)."""
    fa, semanas = 0, 0.0
    for s in sementes:
        df, info = simular(s, None, dias_hist=14.0, dias_monit=dias, encoders=ENCODERS_LINHA)
        th = info["t_fim_hist_s"]
        from poc.simulador_avisos import contadores, eventos_do_log
        ev, cont = eventos_do_log(df), contadores(df, 60.0, info["t_fim_s"])
        mc.avisos = None
        mc.treinar_avisos(ev[ev["t_s"] < th], th, ENCODERS_LINHA, cont[cont["t_s"] < th])
        fa += len(mc.avisos.analisar(ev, th, info["t_fim_s"], cont)[0])
        semanas += dias / 7
    return fa, semanas


def montar():
    treino = [simular_mf2(12, seed=s) for s in TREINO]
    cal = [simular_mf2(48, seed=s) for s in CAL]
    mc = MonitorConjunto().treinar(treino).calibrar(cal)
    base = MonitorTorqueV2(usar_tensao_pontas=False, **mc.param_torque).treinar(treino).calibrar(cal)
    return mc, base


def etapa(nome):
    cfg = ETAPAS[nome]
    t0 = time.time()
    mc, base = montar()
    print(f"treino e calibracao: {time.time() - t0:.0f} s", flush=True)
    por_modelo, sev, m2b, exemplos = avaliar_normais(mc, base, cfg["normal"])
    horas = HORAS_NORMAL * len(cfg["normal"])
    print(f"normais: {por_modelo} ({time.time() - t0:.0f} s)", flush=True)
    det, iso, pre, pre_sev, todos_pre, (tot_det, tot_iso) = avaliar_casos(mc, base, cfg["falha"])
    fa1, sem1 = m1_longo(mc, cfg["normal"])
    n_cen = len(CASOS) * len(cfg["falha"])
    total = sum(por_modelo.values())

    fa = [{"modelo": ROTULO[m], f"FA em {horas} h normais": k, "FA/semana": round(k / horas * SEMANA_H, 2),
           f"FA na 1ª hora dos {n_cen} cenários": pre[m]} for m, k in por_modelo.items()]
    fa.append({"modelo": "**conjunto (total)**", f"FA em {horas} h normais": total,
               "FA/semana": round(total / horas * SEMANA_H, 2), f"FA na 1ª hora dos {n_cen} cenários": todos_pre})
    fa.append({"modelo": "M2 sem tensão das pontas (referência, fora do conjunto)", f"FA em {horas} h normais": m2b,
               "FA/semana": round(m2b / horas * SEMANA_H, 2), f"FA na 1ª hora dos {n_cen} cenários": "-"})
    hp = horas + n_cen
    sev_tot = {k: sev[k] + pre_sev[k] for k in sev}
    texto = [
        f"## {'TESTE (rodado uma vez)' if nome == 'teste' else 'Desenvolvimento (conferência do pipeline)'}",
        "",
        f"Sementes: falhas {cfg['falha'].start}–{cfg['falha'].stop - 1}, normais {cfg['normal'].start}–"
        f"{cfg['normal'].stop - 1} ({len(cfg['normal'])} × {HORAS_NORMAL} h). Treino mf2 {TREINO}, calibração mf2 "
        f"{CAL.start}–{CAL.stop - 1} (48 h). Modelo 2: h = {mc.torque.h:.1f} (sem tensão: {base.h:.1f}); "
        f"Modelo 3: ARX p, q = {mc.proximo['pb1'].ordem}, limites " +
        "; ".join(f"{r}: var {m.limiar['var_jan']:.3f}, contagem {m.limiar['contagem']:.0f}"
                  for r, m in mc.proximo.items()) + ".",
        "",
        "### Falsos alarmes",
        "",
        pd.DataFrame(fa).to_markdown(index=False),
        "",
        f"Severidade dos falsos alarmes ({horas} h normais + 1ª hora dos cenários = {hp} h): "
        + ", ".join(f"{k} {v}" for k, v in sev_tot.items())
        + f" → {sum(sev_tot.values()) / hp * SEMANA_H:.2f} por semana no total, "
        f"{sev_tot['alta'] / hp * SEMANA_H:.2f} por semana com severidade alta. Limite superior 95 % "
        f"(Poisson exato) da taxa do conjunto: {limite_95(total + todos_pre, hp):.2f}/semana com as {hp} h, "
        f"{limite_95(total, horas):.2f}/semana só com as {horas} h normais contínuas (a 1ª hora de cada "
        "cenário começa parada, com o CUSUM zerado: exposição menor que uma hora de regime).",
        f"Modelo 1 sozinho em {sem1:.0f} semanas de log normal (mesmas sementes, 3 encoders): {fa1} FA "
        f"= {fa1 / sem1:.2f} por semana.",
        "",
        "### Detecção (detectados/n · atraso mediano desde o início da falha) e qual modelo disparou primeiro",
        "",
        pd.DataFrame(det).to_markdown(index=False),
        "",
        f"Total: {tot_det}/{n_cen} cenários detectados (qualquer evento depois do início, de qualquer modelo); "
        f"o 1º evento do conjunto apontou o encoder/causa certo em {tot_iso}/{tot_det}.",
        "",
        "### Isolação",
        "",
        "Certo = encoder do rolo com falha (no torque: hipótese 'encoder' desse rolo); no atrito, só a hipótese "
        "'mecânica' do rolo certo conta. Contado sobre os cenários detectados.",
        "",
        pd.DataFrame(iso).to_markdown(index=False),
        "",
    ]
    if exemplos:
        texto += ["Falsos alarmes em operação normal (mensagens):", ""]
        texto += [f"- {e['severidade']}: {e['mensagem'][:220]}" for e in exemplos[:6]] + [""]
    texto.append(f"Tempo: {time.time() - t0:.0f} s.")
    return texto


def gravar(secao, linhas):
    cab = ("# Validação conjunta — Modelos 1, 2 e 3 (v2) no mesmo simulador\n\n"
           "Gerado por `validacao_conjunta.py` (protocolo e sementes no cabeçalho do script). Simulador "
           "`poc/malha_fechada_v2.py` + avisos acoplados (`poc/simulador_avisos.simular_acoplado`); monitor "
           "`monitor/conjunto.py`. Falha: 6 h, começa em 1 h, rampa de 4 h. Nos casos `erros_com` e "
           "`degradacao_sinal` só os avisos mudam (velocidade e torque normais) e a falha final é em 6 h; "
           "alarme depois dela não conta.\n")
    atual = OUT.read_text(encoding="utf-8") if OUT.exists() else cab
    marca = f"<!-- {secao} -->"
    partes = atual.split(marca)
    novo = marca + "\n" + "\n".join(linhas) + "\n" + marca
    atual = partes[0] + novo + partes[2] if len(partes) == 3 else atual.rstrip() + "\n\n" + novo + "\n"
    OUT.write_text(atual, encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    nome = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if nome not in ETAPAS:
        raise SystemExit(f"etapa desconhecida: {nome} (use dev ou teste)")
    linhas = etapa(nome)
    gravar(nome, linhas)
    print("\n".join(linhas))


if __name__ == "__main__":
    main()
