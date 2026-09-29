"""
Testa os detectores na base publica 'Comprehensive Dataset for Fault Detection and
Diagnosis in Inverter-Driven PMSM Systems' (bancada real, 10 Hz, CC-BY-4.0).
  Artigo: https://pmc.ncbi.nlm.nih.gov/articles/PMC11791249/
  Dados:  https://zenodo.org/records/14482932
Sinais: Ia, Ib, VDC, IDC, T1, T2, T3, VD. Classes: F0 normal + 8 falhas de inversor
(circuito aberto, curto, sobretemperatura). Nao ha sinal de encoder.

Protocolo (deteccao de novidade, treino so com normal):
  treino = 1a metade de F0; falsos alarmes = 2a metade de F0;
  para cada falha: % de amostras sinalizadas e tempo ate o 1o alarme.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from testar_skab import DETECTORES

ARQ = Path(__file__).parent / "PMSM" / "PMSM-inverter-fault-diagnosis-3.0" / "processed_data" / "converted_dataset.csv"
SINAIS = ["Ia", "Ib", "VDC", "IDC", "T1", "T2", "T3", "VD"]
DESC = {"F1": "S3 circuito aberto", "F2": "S6 circuito aberto", "F3": "S2 curto",
        "F4": "S3 curto", "F5": "S5 curto", "F6": "HB1 sobretemp.",
        "F7": "HB1+HB2 sobretemp.", "F8": "HB3 sobretemp."}


def main():
    df = pd.read_csv(ARQ)
    normal = df[df.FDD == "F0"][SINAIS].to_numpy(float)
    meio = len(normal) // 2
    tr, val = normal[:meio], normal[meio:]
    sc = StandardScaler().fit(tr)

    linhas = []
    for suav in (False, True):
        for nome, fn in DETECTORES.items():
            linha = {"detector": nome + (" + EWMA" if suav else "")}

            def pontuar(X):
                s_tr, s = fn(sc.transform(tr), sc.transform(X))
                if suav:
                    s_tr = pd.Series(s_tr).ewm(alpha=0.2).mean().to_numpy()
                    s = pd.Series(s).ewm(alpha=0.2).mean().to_numpy()
                p = (s > s_tr.max()).astype(int)
                return pd.Series(p).rolling(3, min_periods=1).median().to_numpy().astype(bool)

            linha["FA % (normal nao visto)"] = round(100 * pontuar(val).mean(), 1)
            detect = []
            for c in DESC:
                p = pontuar(df[df.FDD == c][SINAIS].to_numpy(float))
                primeiro = f"{np.argmax(p) / 10:.1f}s" if p.any() else "-"
                linha[f"{c} {DESC[c]}"] = f"{100 * p.mean():.0f}% ({primeiro})"
                detect.append(p.any())
            linha["falhas detectadas"] = f"{sum(detect)}/8"
            linhas.append(linha)
    res = pd.DataFrame(linhas)
    texto = [
        "# Teste na base publica PMSM (inversor, bancada real 10 Hz)",
        "",
        f"Treino: {meio} amostras normais. Falso alarme medido em {len(val)} amostras normais nao vistas.",
        "Celula de falha = % das amostras da falha sinalizadas (tempo ate o 1o alarme).",
        "Linha de corte = maior valor no treino; mediana movel de 3 amostras.",
        "",
        res.to_markdown(index=False),
    ]
    out = Path(__file__).parent / "resultados_pmsm.md"
    out.write_text("\n".join(texto), encoding="utf-8")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
