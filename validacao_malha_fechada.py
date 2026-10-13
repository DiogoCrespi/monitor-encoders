"""
Validacao do monitor na simulacao em MALHA FECHADA (poc/malha_fechada.py).

Pergunta: quando o drive corrige a velocidade que o encoder mede, os indicadores
baseados em velocidade ainda enxergam a falha? E o torque?

Treino: 8 h normal (semente 1). Calibracao: 8 h normal (semente 2).
Teste nao usado no desenvolvimento: 5 x 48 h normais (sementes 91-95) e
10 cenarios por falha (sementes 101-110).

    python validacao_malha_fechada.py      (gera validacao_malha_fechada.md)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from monitor import MonitorEncoder
from poc.malha_fechada import simular_mf

FS = 10
F0 = int(3600 * FS)
FALHAS = ["perda_pulsos", "ruido", "deriva_ganho", "erros_com"]
TESTE = range(101, 111)
NORMAL = [91, 92, 93, 94, 95]

VARIANTES = {
    "Só vizinhos (velocidade lida)": {"vizinhos"},
    "Só próprio sinal (prever o próximo passo)": {"proprio"},
    "Só torque (vs estimativa cinemática)": {"torque"},
    "Só contador de erros": {"erros"},
    "Monitor sem torque (vizinhos + próprio + erros)": {"vizinhos", "proprio", "erros"},
    "Monitor com torque (os quatro)": {"vizinhos", "proprio", "torque", "erros"},
}


def main():
    treino, val = simular_mf(8, seed=1), simular_mf(8, seed=2)
    mon = MonitorEncoder("pb1", ["festoon", "pb2"], fs=FS).treinar(treino).calibrar(val)

    ev_normal = [mon.analisar(simular_mf(48, seed=s))[1] for s in NORMAL]
    cenarios = {(f, s): simular_mf(6, seed=s, falha=f) for f in FALHAS for s in TESTE}
    ev_falha = {k: mon.analisar(df)[1] for k, df in cenarios.items()}

    linhas = []
    for nome, inds in VARIANTES.items():
        linha = {"detector": nome,
                 "FA em 240 h": sum(sum(e["indicador"] in inds for e in ev) for ev in ev_normal)}
        fa_antes = 0
        for f in FALHAS:
            atrasos = []
            for s in TESTE:
                ev = [e for e in ev_falha[(f, s)] if e["indicador"] in inds]
                fa_antes += sum(e["indice"] < F0 for e in ev)
                depois = [e["indice"] for e in ev if e["indice"] >= F0]
                if depois:
                    atrasos.append((depois[0] - F0) / FS / 60)
            linha[f] = f"{len(atrasos)}/10" + (f" · {np.median(atrasos):.0f} min" if atrasos else "")
        linha["FA antes da falha (40 cen.)"] = fa_antes
        linhas.append(linha)

    # quanto a falha aparece em cada sinal no fim da rampa (severidade > 0,9), linha a > 300 m/min
    efeito = []
    for f in FALHAS:
        df = cenarios[(f, TESTE[0])]
        rod = df["v_setpoint"] > 300
        a, b = df[(df.t_s < 3600) & rod], df[(df.severidade > 0.9) & rod]
        efeito.append({
            "falha": f,
            "velocidade lida P&B1 − setpoint (m/min)": f"{(a.v_pb1 - a.v_setpoint).mean():+.3f} → {(b.v_pb1 - b.v_setpoint).mean():+.3f}",
            "velocidade REAL P&B1 − setpoint (m/min)": f"{(a.v_real_pb1 - a.v_setpoint).mean():+.3f} → {(b.v_real_pb1 - b.v_setpoint).mean():+.3f}",
            "tensão vão 1 (N)": f"{a.tensao_vao1.mean():.0f} ± {a.tensao_vao1.std():.0f} → {b.tensao_vao1.mean():.0f} ± {b.tensao_vao1.std():.0f}",
            "torque P&B1 (N.m)": f"{a.tq_pb1.mean():+.1f} ± {a.tq_pb1.std():.1f} → {b.tq_pb1.mean():+.1f} ± {b.tq_pb1.std():.1f}",
        })

    texto = [
        "# Validação em malha fechada",
        "",
        "Simulação `poc/malha_fechada.py`: cada drive controla a velocidade que o próprio encoder mede;",
        "tensão no papel e torque de cada motor calculados a partir das velocidades reais.",
        "Teste: 240 h normais e 10 cenários por falha, nenhum usado no desenvolvimento.",
        "Célula = cenários detectados / 10 · atraso mediano desde o início da falha.",
        "",
        "## Detecção",
        pd.DataFrame(linhas).to_markdown(index=False),
        "",
        "## Onde a falha aparece (1ª hora normal → severidade > 90%, semente 101)",
        pd.DataFrame(efeito).to_markdown(index=False),
    ]
    out = Path(__file__).with_name("validacao_malha_fechada.md")
    out.write_text("\n".join(texto), encoding="utf-8")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
