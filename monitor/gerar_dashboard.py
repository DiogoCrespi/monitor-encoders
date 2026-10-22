"""
Gera dashboard/monitor-encoders.html (arquivo unico, abre offline no navegador).

    python -m monitor.gerar_dashboard        (na raiz do repositorio; ~1-2 min)

Mostra o monitor conjunto (monitor/conjunto.py, os tres modelos v2) rodando sobre o simulador
v2 (poc/malha_fechada_v2.simular_mf2) com o log de avisos acoplado
(poc/simulador_avisos.simular_acoplado), como em validacao_conjunta.py:
    treino   mf2 sementes 1-2 (12 h)      calibracao  mf2 sementes 3-5 (24 h)
    cenarios sementes de desenvolvimento 41-46 (6 h; falha a partir de 1 h, rampa de 4 h)
Para caber em ~1-2 min, a ordem do ARX do Modelo 3 fica fixa em (16, 3) (a que o BIC escolhe
nesses dados) e a calibracao usa 3 x 24 h em vez de 6 x 48 h; os numeros do painel sao de
demonstracao - os resultados validados ficam em validacao_conjunta.md.
"""
import json
import re
import time
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from monitor import MonitorConjunto
from poc.malha_fechada_v2 import simular_mf2
from poc.simulador_avisos import ENCODERS_LINHA, simular_acoplado

RAIZ = Path(__file__).resolve().parent.parent
FS, BLOCO = 10, 600                # agrega 1 min por ponto do grafico
INICIO_S = 3600.0                  # a falha comeca em 1 h
TREINO, CAL = (1, 2), (3, 4, 5)
SO_AVISOS = ("erros_com", "degradacao_sinal")
PARAM_PROXIMO = {"ordem": (16, 3)}
# (chave, titulo, falha, rolo, semente)
CENARIOS = [
    ("normal", "Operação normal", None, "pb1", 46),
    ("deriva_pb1", "Deriva de escala no P&B1 (raio efetivo/escorregamento)", "deriva_ganho", "pb1", 41),
    ("atrito_pb2", "Atrito no P&B2 (mecânica, não encoder)", "atrito", "pb2", 42),
    ("perda_pulsos_pb1", "Perda de pulsos no P&B1", "perda_pulsos", "pb1", 43),
    ("ruido_pb2", "Ruído crescente no P&B2", "ruido", "pb2", 44),
    ("erros_com_pb1", "Cabo/conector degradando no P&B1 (só avisos)", "erros_com", "pb1", 45),
    ("sinal_festoon", "Sinal do encoder degradando no Festoon (só avisos)", "degradacao_sinal", "festoon", 41),
]


def agregar(x, fn):
    n = len(x) // BLOCO * BLOCO
    return fn(np.asarray(x, float)[:n].reshape(-1, BLOCO), axis=1)


def tabela_md_para_html(texto):
    """Converte as tabelas e titulos (##, ###) de um texto markdown em HTML."""
    linhas = [l.strip() for l in texto.splitlines()]
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
        if l.startswith("## ") or l.startswith("### "):
            html.append(f"<h4>{escape(l.lstrip('#').strip())}</h4>")
    return "\n".join(html)


def secao_md(caminho, marca):
    """Trecho de um .md entre os marcadores <!-- marca --> (como gravam os scripts de validacao)."""
    partes = Path(caminho).read_text(encoding="utf-8").split(f"<!-- {marca} -->")
    return partes[1] if len(partes) == 3 else ""


def cenario(seed, falha, rolo, horas=6):
    """(df no relogio do log, eventos, contadores, info) - igual a validacao_conjunta.cenario."""
    df = simular_mf2(horas, seed=seed, falha=None if falha in SO_AVISOS else falha, rolo=rolo)
    ev, cont, info = simular_acoplado(seed, df["t_s"].to_numpy(), df["severidade"].to_numpy(), falha, rolo,
                                      inicio_s=INICIO_S)
    return df.assign(t_s=df["t_s"] + info["t0_s"]), ev, cont, info


def correto(e, falha, rolo):
    """Isolacao certa: o encoder do rolo com falha (no atrito, a hipotese 'mecanica' desse rolo)."""
    if falha == "atrito":
        return e["modelo"] == "torque" and e["tipo"] == "mecanica" and e["rolo"] == rolo
    if e["modelo"] == "torque":
        return e["tipo"] == "encoder" and e["encoder"] == rolo
    return e["encoder"] == rolo


def _expandir(n, idx, valores):
    """Estatistica comprimida (so amostras no portao) de volta a 10 Hz, congelada fora do portao."""
    full = np.full(n, np.nan)
    full[idx] = valores
    return pd.Series(full).ffill().fillna(0.0).to_numpy()


def indicadores(mc, df, ev, cont, info, rolo):
    """Series de cada modelo para o rolo mostrado, a 10 Hz, e os limites."""
    n = len(df)
    st, _ = mc.torque.analisar(df)
    m3 = mc.proximo[rolo]
    est, _, idx = m3.estatisticas_de(*mc._serie(df, rolo))
    # Modelo 1: maior S/h entre as familias do encoder do rolo (1 = limite)
    alarmes, traj = mc.avisos.analisar(ev, info["t0_s"], info["t_fim_s"], cont)
    t = df["t_s"].to_numpy()
    s_av = np.zeros(n)
    for (enc, fam), tr in traj.items():
        if mc.rolo_do_encoder(enc) != rolo or not len(tr):
            continue
        j = np.searchsorted(tr["t_s"].to_numpy(), t, side="left")      # bin que contem t
        s = tr["S"].to_numpy() / mc.avisos.canais[(enc, fam)]["h"]
        s_av = np.maximum(s_av, s[np.minimum(j, len(s) - 1)])
    for a in alarmes:                         # a trajetoria ja vem zerada depois do alarme: marca o pico
        if mc.rolo_do_encoder(a["encoder"]) == rolo:
            i = min(n - 1, max(0, int(round((a["t_s"] - info["t0_s"]) * FS))))
            s_av[i] = max(s_av[i], a["S"] / a["h"])
    return ({"velocidade": df[f"v_{rolo}"].to_numpy(float), "torque": st["estatistica"],
             "var_jan": _expandir(n, idx, est["var_jan"]), "contagem": _expandir(n, idx, est["contagem"]),
             "avisos": s_av},
            {"torque": st["limiar"], "var_jan": m3.limiar["var_jan"], "contagem": m3.limiar["contagem"],
             "avisos": 1.0})


def _grafico_do_evento(e, rolo):
    if e["modelo"] == "torque":
        return "torque"
    if e["rolo"] != rolo:
        return None                                            # outro rolo: so na tabela
    return e["origem"].get("estatistica", "var_jan") if e["modelo"] == "proximo_passo" else "avisos"


def main():
    t_ini = time.time()
    treino = [simular_mf2(12, seed=s) for s in TREINO]
    cal = [simular_mf2(24, seed=s) for s in CAL]
    mc = MonitorConjunto(param_proximo=PARAM_PROXIMO).treinar(treino).calibrar(cal)
    print(f"treino e calibracao: {time.time() - t_ini:.0f} s", flush=True)

    dados, modelo = {"cenarios": []}, None
    for chave, titulo, falha, rolo, seed in CENARIOS:
        df, ev, cont, info = cenario(seed, falha, rolo)
        t0 = info["t0_s"]
        mc.avisos = None                                       # Modelo 1: historico normal do cenario
        mc.treinar_avisos(ev[ev["t_s"] < t0], t0, ENCODERS_LINHA, cont[cont["t_s"] < t0])
        if modelo is None:
            modelo = json.dumps(mc.exportar(), indent=1)
        eventos = mc.analisar(df, ev, cont, t_ini=t0, t_fim=info["t_fim_s"])
        ind, lim = indicadores(mc, df, ev, cont, info, rolo)
        sev = df["severidade"].to_numpy()
        t_falha = info["t_falha_s"] or np.inf
        saida = []
        for e in eventos:
            t_min = (e["t_s"] - t0) / 60
            i = min(len(df) - 1, max(0, int(round((e["t_s"] - t0) * FS))))
            durante = falha is not None and t0 + INICIO_S <= e["t_s"] < t_falha
            saida.append({
                "t_min": round(t_min, 1), "modelo": e["modelo"], "rolo": e["rolo"], "encoder": e["encoder"],
                "tipo": e["tipo"], "severidade": e["severidade"], "mensagem": e["mensagem"],
                "corroborado": e["corroborado_por"], "grafico": _grafico_do_evento(e, rolo),
                "falha_pct": None if falha in SO_AVISOS or falha is None else round(float(sev[i]), 2),
                "fase": "durante a falha" if durante else "falso alarme",
                "isolacao": (None if not durante else ("certa" if correto(e, falha, rolo) else "errada")),
            })
        tt = agregar((df["t_s"].to_numpy() - t0) / 60, np.min)
        dados["cenarios"].append({
            "chave": chave, "titulo": titulo, "falha": falha, "rolo": rolo, "so_avisos": falha in SO_AVISOS,
            "t": np.round(tt, 2).tolist(), "limiar": {k: round(float(v), 4) for k, v in lim.items()},
            "velocidade": np.round(agregar(ind["velocidade"], np.mean), 1).tolist(),
            **{k: np.round(agregar(ind[k], np.max), 3).tolist() for k in ("torque", "var_jan", "contagem", "avisos")},
            "eventos": saida,
        })
        print(f"  {chave}: {len(saida)} evento(s) ({time.time() - t_ini:.0f} s)", flush=True)

    # resultados validados (validacao_conjunta.py); sem o .md, a secao mostra como gerar
    md = RAIZ / "validacao_conjunta.md"
    texto = secao_md(md, "teste") if md.exists() else ""
    validacao = ("<h3>Validação conjunta — teste (sementes novas, rodado uma vez)</h3>" + tabela_md_para_html(texto)
                 if texto else "<p>Rode <code>python validacao_conjunta.py teste</code> (ver README) para "
                               "preencher esta seção.</p>")

    modelo_html = Path(__file__).with_name("dashboard_modelo.html").read_text(encoding="utf-8")
    html = (modelo_html
            .replace("/*__DADOS__*/null", json.dumps(dados, separators=(",", ":"), ensure_ascii=False)
                     .replace("</", "<\\/"))
            .replace("<!--__VALIDACAO__-->", validacao)
            .replace("<!--__MODELO__-->", escape(modelo)))
    assert not re.search(r"__DADOS__|__VALIDACAO__|__MODELO__", html), "marcador nao substituido"
    saida = RAIZ / "dashboard" / "monitor-encoders.html"
    saida.parent.mkdir(exist_ok=True)
    saida.write_text(html, encoding="utf-8")
    print(f"gerado {saida} ({saida.stat().st_size / 1024:.0f} KB, {time.time() - t_ini:.0f} s)")


if __name__ == "__main__":
    main()
