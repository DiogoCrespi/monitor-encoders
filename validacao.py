"""
Validacao do monitor:
  1. comparacao com referencias simples
  2. teste em cenarios nao usados no desenvolvimento (sementes 101-110),
     com treino, calibracao e teste separados

Detectores comparados (todos treinados/calibrados do mesmo jeito):
  R0  referencia nula: nunca alarma
  R1  referencia ingenua: P&B1 comparado so com o Festoon (razao fixa), sem dinamica
  M1  so o indicador 'proprio sinal' (prever o proximo passo)
  M2  so o indicador 'vizinhos'
  M   monitor completo (vizinhos + proprio + contador de erros)

    python validacao.py      (gera validacao.md)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from monitor import MonitorEncoder, de_simulacao
from monitor.monitor import _atraso
from poc.poc_deteccao import simular

FS = 10
F0 = int(3600 * FS)
FALHAS = ["perda_pulsos", "ruido", "deriva_ganho", "erros_com"]
SEMENTES = range(101, 111)


class SoFestoon(MonitorEncoder):
    """Referencia ingenua: estima o P&B1 so pelo Festoon no mesmo instante."""
    def _x_viz(self, df):
        return df[["v_festoon"]].to_numpy()


def filtrar(eventos, indicadores):
    return [e for e in eventos if e["indicador"] in indicadores]


def main():
    treino = de_simulacao(simular(8, seed=1))
    val = de_simulacao(simular(8, seed=2))
    normal = de_simulacao(simular(48, seed=99))
    cen = {(f, s): simular(6, seed=s, falha=f) for f in FALHAS for s in SEMENTES}

    completo = MonitorEncoder("pb1", ["festoon", "pb2"], fs=FS).treinar(treino).calibrar(val)
    ingenuo = SoFestoon("pb1", ["festoon", "pb2"], fs=FS).treinar(treino).calibrar(val)
    variantes = {
        "R0 referência nula (nunca alarma)": (None, set()),
        "R1 referência ingênua (só Festoon, razão fixa)": (ingenuo, {"vizinhos"}),
        "M1 só próprio sinal (prever próximo passo)": (completo, {"proprio"}),
        "M2 só vizinhos (redundância)": (completo, {"vizinhos"}),
        "M  monitor completo": (completo, {"vizinhos", "proprio", "erros"}),
    }

    linhas = []
    for nome, (mon, inds) in variantes.items():
        fa = len(filtrar(mon.analisar(normal)[1], inds)) if mon else 0
        linha = {"detector": nome, "FA em 48 h normal": fa}
        fa_antes = 0
        for f in FALHAS:
            atrasos = []
            for s in SEMENTES:
                if mon is None:
                    continue
                df = cen[(f, s)]
                ev = filtrar(mon.analisar(de_simulacao(df))[1], inds)
                fa_antes += sum(e["indice"] < F0 for e in ev)
                depois = [e for e in ev if e["indice"] >= F0]
                if depois:
                    atrasos.append((depois[0]["indice"] - F0) / FS / 60)
            linha[f] = (f"{len(atrasos)}/10 · {np.median(atrasos):.0f} min" if atrasos else "0/10")
        linha["FA antes da falha (40 cen.)"] = fa_antes
        linhas.append(linha)

    res = pd.DataFrame(linhas)
    texto = [
        "# Validação: referências e teste em cenários não usados no desenvolvimento",
        "",
        "Treino: 8 h normal (semente 1). Calibração da linha de corte: 8 h normal (semente 2).",
        "Teste NUNCA usado no desenvolvimento: 48 h normal (semente 99) e 10 cenários por falha",
        "(sementes 101-110). Célula = cenários detectados / 10 · atraso mediano desde o início da falha.",
        "",
        res.to_markdown(index=False),
    ]
    out = Path(__file__).with_name("validacao.md")
    out.write_text("\n".join(texto), encoding="utf-8")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
