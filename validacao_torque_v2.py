"""
Validacao do Modelo 2 (torque esperado x real) no simulador v2 (poc/malha_fechada_v2.py).

Compara, no MESMO simulador:
  antes  = indicador 'torque' do MonitorEncoder atual (EWMA lambda=0,005, corte 1,1 x maximo),
           um monitor por rolo; a isolacao "ingenua" e o rolo cujo monitor alarma primeiro;
  depois = MonitorTorqueV2 (regressao com dinamica, medias de 1 min, 3 residuos estruturados,
           CUSUM bilateral, h por meta de 1 falso alarme por semana via bootstrap em blocos).

Dois conjuntos de teste, cada um medido uma vez:
  Teste 1 - v2 como estava ANTES de qualquer teste (k=1; calibracao 8 x 12 h, sementes 3-10).
            Normais 91-95 (48 h), falhas 101-110.
  Teste 2 - v2.1: depois do Teste 1 mostrar falso alarme acima da meta, o diagnostico e o ajuste
            (k=3, blocos de 2 h, calibracao 6 x 48 h, sementes 3-8) foram feitos SO com sementes
            de desenvolvimento (11-33). Medido em sementes novas: normais 96-100, falhas 111-120.
  Treino do modelo: sementes 1-2 (12 h). Falhas: 6 h, comeca em 1 h, rampa de 4 h.

    python validacao_torque_v2.py          (os dois testes, gera validacao_torque_v2.md)
    python validacao_torque_v2.py --dev    (v2.1 nas sementes de desenvolvimento 21-23 / 11-20; so imprime)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from monitor import MonitorEncoder
from monitor.torque_v2 import MonitorTorqueV2
from poc.malha_fechada_v2 import ROLOS, simular_mf2

FS = 10
F0 = int(3600 * FS)
DEV = "--dev" in sys.argv
HORAS_NORMAL = 48
CASOS = [("deriva_ganho", "festoon"), ("deriva_ganho", "pb1"), ("deriva_ganho", "pb2"),
         ("atrito", "pb1"), ("atrito", "pb2"), ("perda_pulsos", "pb1"), ("ruido", "pb1")]
V2 = dict(k=1.0, media_s=60, ordem_ar=0)                      # registrado antes do Teste 1
V21 = dict(k=3.0, media_s=60, ordem_ar=0, bloco_h=2.0)        # ajustado so com sementes 1-40
CONJUNTOS = [
    dict(nome="Teste 1 — v2 como registrado antes do teste", normal=range(91, 96), falha=range(101, 111),
         params=V2, cal=lambda m: [simular_mf2(12, seed=s, modo_tensao=m) for s in range(3, 11)]),
    dict(nome="Teste 2 — v2.1 (ajuste feito só com sementes de desenvolvimento), sementes novas",
         normal=range(96, 101), falha=range(111, 121), params=V21,
         cal=lambda m: [simular_mf2(48, seed=s, modo_tensao=m) for s in range(3, 9)]),
]
if DEV:
    CONJUNTOS = [dict(CONJUNTOS[1], nome="DESENVOLVIMENTO (v2.1, sementes 11-23)",
                      normal=range(21, 24), falha=range(11, 21))]


class Baseline:
    """Tres MonitorEncoder (um por rolo), so o indicador 'torque'."""

    def __init__(self, treino, cal):
        self.m = {r: MonitorEncoder(r, [v for v in ROLOS if v != r], fs=FS).treinar(treino).calibrar(cal)
                  for r in ROLOS}

    def eventos(self, df):
        ev = []
        for r, m in self.m.items():
            ev += [dict(e, rolo=r) for e in m.analisar(df)[1] if e["indicador"] == "torque"]
        return sorted(ev, key=lambda e: e["indice"])


def avaliar(eventos_fn, cenarios, normais, tipo_v2):
    linhas = {}
    fa = [len(eventos_fn(d)) for d in normais]
    for (falha, rolo), dfs in cenarios.items():
        det, atr, iso1, iso_rolo, isoc, pre = 0, [], 0, 0, 0, 0
        for df in dfs:
            ev = eventos_fn(df)
            pre += sum(e["indice"] < F0 for e in ev)
            dep = [e for e in ev if e["indice"] >= F0]
            if not dep:
                continue
            det += 1
            atr.append((dep[0]["indice"] - F0) / FS / 60)
            iso_rolo += dep[0]["rolo"] == rolo
            if tipo_v2:
                certo = ("encoder" if falha in ("deriva_ganho", "perda_pulsos", "ruido") else "mecanica", rolo)
                iso1 += (dep[0]["tipo"], dep[0]["rolo"]) == certo
                conf = [e["confirmada"] for e in dep if e["confirmada"]]
                isoc += bool(conf) and conf[0][:2] == certo
        linhas[(falha, rolo)] = dict(det=det, atr=np.median(atr) if atr else np.nan, iso1=iso1,
                                     iso_rolo=iso_rolo, isoc=isoc, pre=pre, n=len(dfs))
    return sum(fa), linhas


def fmt(l, v2):
    s = f"{l['det']}/{l['n']}"
    if l["det"]:
        s += f" · {l['atr']:.0f} min"
    iso = f"{l['iso_rolo']}/{l['det']}" if not v2 else f"{l['iso1']}/{l['det']} → {l['isoc']}/{l['det']}"
    return s, iso


def rodar(cj, t0, extras):
    params = cj["params"]
    treino = [simular_mf2(12, seed=s) for s in (1, 2)]
    cal = cj["cal"](())
    normais = [simular_mf2(HORAS_NORMAL, seed=s) for s in cj["normal"]]
    cenarios = {c: [simular_mf2(6, seed=s, falha=c[0], rolo=c[1]) for s in cj["falha"]] for c in CASOS}
    print(f"simulacao {time.time() - t0:.0f} s", flush=True)

    base = Baseline(treino[0], treino[1])
    v2 = MonitorTorqueV2(**params).treinar(treino).calibrar(cal)
    v2t = MonitorTorqueV2(usar_tensao_pontas=True, **params).treinar(treino).calibrar(cal)
    metodos = [("Antes: 'torque' atual (3 monitores)", base.eventos, False),
               ("Depois: torque v2 (só torque + setpoint)", lambda d: v2.analisar(d)[1], True),
               ("Depois: torque v2 + tensão das pontas", lambda d: v2t.analisar(d)[1], True)]
    res = {nome: avaliar(fn, cenarios, normais, tv2) for nome, fn, tv2 in metodos}
    fa_pb1 = sum(sum(e["indicador"] == "torque" for e in base.m["pb1"].analisar(d)[1]) for d in normais)
    print(f"avaliacao {time.time() - t0:.0f} s", flush=True)

    horas = HORAS_NORMAL * len(normais)
    tab, tab_iso = [], []
    for nome, _, tv2 in metodos:
        fa, linhas = res[nome]
        linha = {"método": nome, f"FA em {horas} h": fa}
        linha_i = {"método": nome}
        for c in CASOS:
            linha[f"{c[0]} {c[1]}"], linha_i[f"{c[0]} {c[1]}"] = fmt(linhas[c], tv2)
        tab.append(linha)
        tab_iso.append(linha_i)
    pre = {nome: sum(l["pre"] for l in res[nome][1].values()) for nome, _, _ in metodos}
    texto = [
        f"## {cj['nome']}",
        "",
        f"Normais: sementes {cj['normal'].start}–{cj['normal'].stop - 1} ({len(normais)} × {HORAS_NORMAL} h = "
        f"{horas} h; meta de 1 FA/semana ⇒ esperado ≈ {horas / 168:.1f}). "
        f"Falhas: sementes {cj['falha'].start}–{cj['falha'].stop - 1}.",
        f"Parâmetros: {params}; h = {v2.h:.1f} (com tensão das pontas: {v2t.h:.1f}); "
        f"taxa de FA na reamostragem da calibração: {v2.fa_boot_semana:.2f}/semana.",
        "",
        "### Detecção e falsos alarmes",
        pd.DataFrame(tab).to_markdown(index=False),
        "",
        "Falsos alarmes antes do início da falha (70 cenários × 1 h): "
        + ", ".join(f"{k}: {v}" for k, v in pre.items()) + ".",
        f"Só o monitor do P&B1 do método antigo (como em `validacao_malha_fechada.md`): {fa_pb1} FA em {horas} h.",
        "",
        "### Isolação (qual encoder / se é mecânica)",
        "Antes: rolo do primeiro monitor que alarmou (o método antigo não distingue encoder de mecânica).",
        "Depois: tipo + rolo certos no 1º alarme → na isolação confirmada (maioria de 3 alarmes em 90 min), "
        "sobre os detectados.",
        "",
        pd.DataFrame(tab_iso).to_markdown(index=False),
        "",
    ]
    if not extras:
        return texto

    # ---------------- modo de controle: P&B2 com malha de tensao (trim)
    tr_t = [simular_mf2(12, seed=s, modo_tensao=("pb2",)) for s in (1, 2)]
    base_t = Baseline(tr_t[0], tr_t[1])
    v2_t = MonitorTorqueV2(**params).treinar(tr_t).calibrar(cj["cal"](("pb2",)))
    cen_t = {("deriva_ganho", r): [simular_mf2(6, seed=s, falha="deriva_ganho", rolo=r, modo_tensao=("pb2",))
                                   for s in cj["falha"]] for r in ("pb1", "pb2")}
    _, lt_b = avaliar(base_t.eventos, cen_t, [], False)
    _, lt_v = avaliar(lambda d: v2_t.analisar(d)[1], cen_t, [], True)
    trim = []
    for df in cen_t[("deriva_ganho", "pb2")]:
        rod = df["v_setpoint"] > 300
        a, b = df[(df.t_s < 3600) & (df.t_s > 1800) & rod], df[(df.severidade > 0.9) & rod]
        trim.append(b.trim_pb2.mean() - a.trim_pb2.mean())
    print(f"modo tensao {time.time() - t0:.0f} s", flush=True)
    tab_t = []
    for nome, lin, tv2 in (("Antes: 'torque' atual", lt_b, False), ("Depois: torque v2", lt_v, True)):
        linha = {"método": nome}
        for c in lin:
            linha[f"deriva {c[1]} (detecção)"], linha[f"deriva {c[1]} (isolação)"] = fmt(lin[c], tv2)
        tab_t.append(linha)

    # assinatura medida (semente de desenvolvimento 11, fim da rampa)
    assin = []
    for falha, rolo in [("deriva_ganho", r) for r in ROLOS] + [("atrito", r) for r in ROLOS]:
        df = simular_mf2(6, seed=11, falha=falha, rolo=rolo)
        rod = df["v_setpoint"] > 300
        a, b = df[(df.t_s < 3600) & (df.t_s > 1800) & rod], df[(df.severidade > 0.9) & rod]
        linha = {"falha": f"{falha} {rolo}"}
        for r in ROLOS:
            linha[f"Δ torque {r} (N.m)"] = f"{b[f'tq_{r}'].mean() - a[f'tq_{r}'].mean():+.1f}"
        linha["Δ tensão vão 1 / vão 2 (N)"] = (f"{b.tensao_vao1.mean() - a.tensao_vao1.mean():+.0f} / "
                                              f"{b.tensao_vao2.mean() - a.tensao_vao2.mean():+.0f}")
        linha["velocidade lida − real do rolo com falha (m/min)"] = \
            f"{(b[f'v_{rolo}'] - b[f'v_real_{rolo}']).mean():+.3f}"
        assin.append(linha)
    return texto + [
        "### Dependência do modo de controle: P&B2 com malha de tensão (trim de velocidade)",
        "Modelos treinados e calibrados com a linha nesse modo.",
        "",
        pd.DataFrame(tab_t).to_markdown(index=False),
        "",
        f"Deriva do P&B2 nesse modo: o trim do P&B2 muda {np.mean(trim):+.4f} m/min (média de 10 cenários) — "
        "é ali que a falha aparece, não no torque.",
        "",
        "## Assinatura medida no simulador (semente de desenvolvimento 11, fim da rampa × 1ª hora)",
        pd.DataFrame(assin).to_markdown(index=False),
        "",
    ]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    texto = [
        "# Validação do Modelo 2 — torque esperado × real (simulador v2)",
        "",
        "Simulador `poc/malha_fechada_v2.py`; script `validacao_torque_v2.py`.",
        "Falhas: 10 cenários de 6 h por linha (falha começa em 1 h, rampa de 4 h; deriva = 0,005 % no fim;",
        "atrito = +6 N.m no fim). Célula = detectados/10 · atraso mediano desde o início da falha.",
        "",
    ]
    for n, cj in enumerate(CONJUNTOS):
        texto += rodar(cj, t0, extras=(n == len(CONJUNTOS) - 1))
    texto.append(f"Tempo total: {time.time() - t0:.0f} s.")
    if not DEV:                                     # o modo --dev so imprime
        Path(__file__).with_name("validacao_torque_v2.md").write_text("\n".join(texto), encoding="utf-8")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
