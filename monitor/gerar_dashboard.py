"""
Gera dashboard/monitor-encoders.html (arquivo unico, abre offline no navegador).

    python -m monitor.gerar_dashboard        (na raiz do repositorio)
"""
import json
import re
from html import escape
from pathlib import Path

import numpy as np

from monitor import MonitorEncoder, de_simulacao
from poc.poc_deteccao import simular

RAIZ = Path(__file__).resolve().parent.parent
FS, BLOCO = 10, 100                # agrega 10 s por ponto do grafico
F0_MIN = 60.0
CENARIOS = [
    ("normal", "Operação normal", None, 40),
    ("perda_pulsos", "Perda de pulsos (canal intermitente)", "perda_pulsos", 10),
    ("ruido", "Ruído crescente (disco/EMI)", "ruido", 10),
    ("deriva_ganho", "Deriva lenta (acoplamento folgando)", "deriva_ganho", 10),
    ("erros_com", "Erros de comunicação (cabo/conector)", "erros_com", 10),
]


def agregar(x, fn):
    n = len(x) // BLOCO * BLOCO
    return fn(x[:n].reshape(-1, BLOCO), axis=1)


def tabela_md_para_html(caminho):
    """Converte as tabelas markdown de um arquivo de resultados em HTML."""
    linhas = [l.strip() for l in Path(caminho).read_text(encoding="utf-8").splitlines()]
    html, bloco = [], []
    for l in linhas + [""]:
        if l.startswith("|"):
            bloco.append(l)
            continue
        if bloco:
            cab = [c.strip() for c in bloco[0].strip("|").split("|")]
            corpo = [[c.strip() for c in b.strip("|").split("|")] for b in bloco[2:]]
            html.append("<div class='tw'><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cab)
                        + "</tr></thead><tbody>" + "".join(
                            "<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>" for r in corpo)
                        + "</tbody></table></div>")
            bloco = []
        elif l.startswith("## "):
            html.append(f"<h4>{escape(l[3:])}</h4>")
    return "\n".join(html)


def main():
    monitor = MonitorEncoder("pb1", ["festoon", "pb2"], fs=FS)
    monitor.treinar(de_simulacao(simular(8, seed=1))).calibrar(de_simulacao(simular(8, seed=2)))

    dados = {"limiar": monitor.limiar, "cenarios": []}
    for chave, titulo, falha, seed in CENARIOS:
        sim = simular(6, seed=seed, falha=falha)
        ind, eventos = monitor.analisar(de_simulacao(sim))
        sev = sim["severidade"].to_numpy()
        t = agregar(sim["t_s"].to_numpy() / 60, np.min)
        for e in eventos:
            e["t_min"] = round(e["t_s"] / 60, 1)
            e["severidade"] = round(float(sev[e["indice"]]), 2)
            e["fase"] = "antes da falha" if (falha and e["t_min"] < F0_MIN) or not falha else "durante a falha"
        dados["cenarios"].append({
            "chave": chave, "titulo": titulo, "falha": falha is not None,
            "t": np.round(t, 2).tolist(),
            "velocidade": np.round(agregar(ind["medida"], np.mean), 1).tolist(),
            "vizinhos": np.round(agregar(ind["vizinhos"], np.max), 3).tolist(),
            "proprio": np.round(agregar(ind["proprio"], np.max), 3).tolist(),
            "erros": agregar(ind["erros"], np.max).astype(int).tolist(),
            "severidade": np.round(agregar(sev, np.max), 3).tolist(),
            "eventos": [{k: e[k] for k in ("t_min", "indicador", "mensagem", "severidade", "fase")}
                        for e in eventos],
        })

    # tabelas geradas pelos scripts de validacao; as que ainda nao foram rodadas ficam de fora
    fontes = [("Simulacao dos encoders (3 sementes por falha)", RAIZ / "poc" / "resultados.md"),
              ("Base publica real: SKAB (bomba/circulacao de agua)", RAIZ / "bases" / "resultados_skab.md"),
              ("Base publica real: PMSM com inversor (10 Hz)", RAIZ / "bases" / "resultados_pmsm.md")]
    validacao = "".join(f"<h3>{t}</h3>" + tabela_md_para_html(p) for t, p in fontes if p.exists()) \
        or "<p>Rode os scripts de validacao (ver README) para preencher esta secao.</p>"
    modelo = json.dumps(monitor.exportar(), indent=1)

    modelo_html = Path(__file__).with_name("dashboard_modelo.html").read_text(encoding="utf-8")
    html = (modelo_html
            .replace("/*__DADOS__*/null", json.dumps(dados, separators=(",", ":")))
            .replace("<!--__VALIDACAO__-->", validacao)
            .replace("<!--__MODELO__-->", escape(modelo)))
    assert not re.search(r"__DADOS__|__VALIDACAO__|__MODELO__", html), "marcador nao substituido"
    saida = RAIZ / "dashboard" / "monitor-encoders.html"
    saida.parent.mkdir(exist_ok=True)
    saida.write_text(html, encoding="utf-8")
    print(f"gerado {saida} ({saida.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
