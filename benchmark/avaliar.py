"""
Avaliador comum das arquiteturas (interface em arquiteturas/base.py) nas familias de benchmark/cenarios.py.

    python -m benchmark.avaliar --seeds dev --familias nominal avisos \
        --arquiteturas arquiteturas.atual:Conjunto arquiteturas.atual:Modelo2 --tag dev_atual
    -> benchmark/resultados_<tag>.md, resultados_<tag>.csv (resumo) e resultados_<tag>_cenarios.csv (bruto)

Sementes: 'dev' = 500-599 (uso livre); 'final' = 800-999, so com --confirmar-final (rodar UMA vez; ja usada);
'confirmacao' = 1000-1199 (mesmo tamanho da final), so com --confirmar-confirmacao - reservada ao
orquestrador para a rodada de confirmacao (rodar UMA vez).
Treino mf2 1-2 (12 h), calibracao mf2 3-8 (48 h; --cal longa = 3-20), na planta de cada familia.

CONTAGEM POR INCIDENTE (ver incidentes()): alarmes com a mesma chave 'incidente' sao um so; senao,
alarmes ate 10 min depois do 1o alarme de um incidente entram nele (janela fixa, sem encadear). A
(rolo, causa) de um incidente e a do 1o alarme dele que aponta um rolo.

Metricas por (familia, variante, arquitetura) e por caso:
    deteccao      cenarios de falha com algum incidente comecando em [inicio da falha, t_lim) / N;
                  t_lim = falha final nos casos so de avisos (fim do arquivo), senao fim do arquivo
    atraso        mediana (min) do 1o incidente depois do inicio da falha
    iso 1o        o 1o incidente depois do inicio aponta (rolo, causa) certos; causa certa = 'mecanica' no
                  atrito e 'encoder' nas demais (contado sobre os detectados)
    iso confirm.  1o par (rolo, causa) que aparece em 2 incidentes distintos depois do inicio (regra unica
                  para todas as arquiteturas) e certo / cenarios com par confirmado
    errados       incidentes depois do inicio com (rolo, causa) errados
    FA            incidentes nos arquivos normais (todos contam) por semana (168 h) e limite superior 95 %
                  unilateral de Poisson exato; FA_pre = incidentes antes do inicio nos arquivos de falha
    tempo         segundos de `analisar` somados (sem simulacao) e por 48 h de dado
"""
import argparse
import copy
import importlib
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from . import cenarios as C

SEMANA_H = 168.0
AQUI = Path(__file__).resolve().parent


def limite_95(k, horas):
    """Limite superior unilateral 95 % (Poisson exato) da taxa, em eventos por semana."""
    if horas <= 0:
        return np.nan
    return chi2.ppf(0.95, 2 * k + 2) / 2 / horas * SEMANA_H


def carregar_classe(spec):
    mod, _, cls = spec.partition(":")
    if not cls:
        raise ValueError(f"use modulo:Classe (recebido {spec!r})")
    return getattr(importlib.import_module(mod), cls)


# ------------------------------------------------------------------ metricas de um cenario
def _correto(e, rolo, causa):
    return e.get("rolo") == rolo and e.get("causa") == causa


HOLDOFF_S = 600.0


def incidentes(alarmes, holdoff_s=HOLDOFF_S):
    """Agrupa alarmes em incidentes (a unidade de contagem de deteccao e de falso alarme).

    Regra (ordem de tempo):
      1. alarmes com o mesmo valor (nao None) na chave opcional 'incidente' sao o mesmo incidente,
         a qualquer distancia no tempo (ligacao explicita: ex. a CVA dispara e o Modelo 2 confirma depois);
      2. senao, um alarme ate holdoff_s (10 min) depois do PRIMEIRO alarme do incidente aberto mais
         recente entra nele (janela fixa, sem encadear: um alarme continuo conta 1 vez a cada 10 min);
      3. senao, abre um incidente novo.
    Devolve lista de listas de alarmes (cada lista em ordem de tempo)."""
    grupos, por_id = [], {}
    for e in sorted(alarmes, key=lambda e: e["t_s"]):
        ident = e.get("incidente")
        if ident is not None and ident in por_id:
            por_id[ident].append(e)
            continue
        if grupos and e["t_s"] - grupos[-1][0]["t_s"] <= holdoff_s:
            g = grupos[-1]
        else:
            g = []
            grupos.append(g)
        g.append(e)
        if ident is not None:
            por_id[ident] = g
    return grupos


def _isolacao(inc):
    """(rolo, causa) do incidente = 1o alarme dele que aponta um rolo (senao o 1o alarme)."""
    e = next((x for x in inc if x.get("rolo") is not None), inc[0])
    return e.get("rolo"), e.get("causa")


def medir(alarmes, g, holdoff_s=HOLDOFF_S):
    """Metricas de um cenario (g = saida de cenarios.gerar sem o df), contadas por incidente."""
    al = sorted(alarmes, key=lambda e: e["t_s"])
    incs = incidentes(al, holdoff_s)
    r = {"n_alarmes": len(al), "n_incidentes": len(incs), "horas": g["horas"]}
    if g["falha"] is None:
        r.update(fa=len(incs))
        return r
    t_on, t_lim = g["t_on"], g["t_lim"]
    certo = (g["rolo"], g["causa"])
    pre = [x for x in incs if x[0]["t_s"] < t_on]
    # incidente "depois do inicio": o 1o alarme dele cai em [t_on, t_lim)
    dep = [x for x in incs if t_on <= x[0]["t_s"] < t_lim]
    r.update(fa_pre=len(pre), horas_pre=(t_on - g["t0"]) / 3600, detectado=bool(dep),
             atraso_min=(dep[0][0]["t_s"] - t_on) / 60 if dep else np.nan,
             iso1=_isolacao(dep[0]) == certo if dep else np.nan,
             errados=sum(_isolacao(x) != certo for x in dep))
    vistos, conf = {}, None
    for x in dep:                                  # 1o (rolo, causa) repetido em 2 incidentes distintos
        k = _isolacao(x)
        vistos[k] = vistos.get(k, 0) + 1
        if vistos[k] >= 2:
            conf = k
            break
    r["iso_conf"] = np.nan if conf is None else conf == certo
    return r


# ------------------------------------------------------------------ trabalho nos processos
_ARQ = {}


def _init(pacote):
    global _ARQ
    _ARQ = pickle.loads(pacote)


CACHE = AQUI / "_cache"


def _arquivo_cache(planta, cal):
    return CACHE / f"treino_{planta}_{cal}_{C.HORAS_TREINO}h_{C.HORAS_CAL}h_{len(C.CAL)}.pkl"


def _gerar_treino(args):
    """Gera (uma vez) o treino/calibracao de uma planta e guarda em benchmark/_cache (deterministico)."""
    planta, cal = args
    arq = _arquivo_cache(planta, cal)
    if not arq.exists():
        CACHE.mkdir(exist_ok=True)
        (CACHE / ".gitignore").write_text("*\n")
        tmp = arq.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(C.dados_treino(planta, cal), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, arq)
    return planta


def _hash_fontes(spec):
    """Hash do codigo que define o modelo treinado: modulo da classe, arquiteturas/, monitor/, simuladores."""
    import hashlib
    raiz = AQUI.parent
    mod = importlib.import_module(spec.partition(":")[0])
    arqs = sorted({Path(mod.__file__).resolve(), *(raiz / "arquiteturas").glob("*.py"), *(raiz / "monitor").glob("*.py"),
                   *(raiz / "poc").glob("*.py"), AQUI / "sim_ext.py", AQUI / "cenarios.py"})
    h = hashlib.sha256(spec.encode())
    for a in arqs:
        h.update(a.read_bytes())
    return h.hexdigest()[:16]


def _treinar(args):
    spec, planta, cal, meta = args[:4]
    usar_cache = len(args) > 4 and args[4]
    arq_m = CACHE / f"modelo_{spec.replace(':', '.')}_{planta}_{cal}_{meta:g}_{_hash_fontes(spec)}.pkl"
    if usar_cache and arq_m.exists():
        return spec, planta, arq_m.read_bytes(), 0.0, 0.0
    t = time.time()
    _gerar_treino((planta, cal))
    with open(_arquivo_cache(planta, cal), "rb") as f:
        treino, calib, ev, dur = pickle.load(f)
    t_sim = time.time() - t
    t = time.time()
    a = carregar_classe(spec)()
    a.treinar(treino, ev, dur)
    a.calibrar(calib, meta_fa_semana=meta)
    b = pickle.dumps(a)
    if usar_cache:
        CACHE.mkdir(exist_ok=True)
        tmp = arq_m.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(b)
        os.replace(tmp, arq_m)
    return spec, planta, b, t_sim, time.time() - t


def _rodar(c):
    t = time.time()
    g = C.gerar(c)
    t_sim = time.time() - t
    df, ev = g.pop("df"), g.pop("eventos")
    linhas = []
    for (spec, planta), arq in _ARQ.items():
        if planta != c.planta:
            continue
        a = copy.deepcopy(arq)
        t = time.time()
        try:
            alarmes, erro = a.analisar(df.copy(), ev.copy() if ev is not None else None), ""
        except Exception as exc:                          # registra e segue (arquitetura com bug)
            alarmes, erro = [], f"{type(exc).__name__}: {exc}"
        dt = time.time() - t
        r = medir(alarmes, g)
        linhas.append({"arquitetura": getattr(a, "nome", spec), "spec": spec, "familia": c.familia,
                       "variante": c.variante, "caso": c.caso, "seed": c.seed, "normal": c.falha is None,
                       **r, "t_analisar_s": dt, "t_sim_s": t_sim, "erro": erro})
    return linhas


# ------------------------------------------------------------------ agregacao
def _frac(k, n):
    return f"{int(k)}/{int(n)}"


def agregar(raw, por):
    linhas = []
    for chave, g in raw.groupby(por, sort=False):
        f = g[~g["normal"]]
        n = g[g["normal"]]
        det = f[f["detectado"] == True]  # noqa: E712
        conf = f["iso_conf"].dropna()
        k_fa, h_fa = int(n["fa"].sum()) if len(n) else 0, float(n["horas"].sum()) if len(n) else 0.0
        h_tot = float(g["horas"].sum())
        linha = dict(zip(por, chave if isinstance(chave, tuple) else (chave,)))
        linha.update({
            "deteccao": _frac(len(det), len(f)) if len(f) else "-",
            "atraso_med_min": round(float(det["atraso_min"].median()), 1) if len(det) else np.nan,
            "iso_1o": _frac(det["iso1"].astype(bool).sum(), len(det)) if len(det) else "-",
            "iso_confirmada": _frac(conf.astype(bool).sum(), len(conf)) if len(conf) else "-",
            "incid_errados_pos": int(f["errados"].sum()) if len(f) else 0,
            "alarmes_brutos": int(g["n_alarmes"].sum()),
            "FA_pre_falha": int(f["fa_pre"].sum()) if len(f) else 0,
            "FA": k_fa if len(n) else "-", "horas_normais": round(h_fa, 1),
            "FA_semana": round(k_fa / h_fa * SEMANA_H, 2) if h_fa else np.nan,
            "FA_semana_LS95": round(limite_95(k_fa, h_fa), 2) if h_fa else np.nan,
            "t_analisar_s": round(float(g["t_analisar_s"].sum()), 1),
            "s_por_48h": round(float(g["t_analisar_s"].sum()) / h_tot * 48, 2) if h_tot else np.nan,
            "erros": int((g["erro"] != "").sum()),
        })
        linhas.append(linha)
    return pd.DataFrame(linhas)


def relatorio(raw, treinos, args, t_total):
    resumo = agregar(raw, ["familia", "variante", "arquitetura"])
    casos = agregar(raw, ["familia", "variante", "caso", "arquitetura"])
    fam = agregar(raw, ["familia", "arquitetura"])
    txt = [f"# Benchmark de arquiteturas — {args.tag}", "",
           f"Gerado por `python -m benchmark.avaliar` (sementes **{args.seeds}**, calibração {args.cal}, meta "
           f"{args.meta} FA/semana). Arquiteturas: {', '.join(args.arquiteturas)}. Famílias: "
           f"{', '.join(args.familias)}. {len(raw['seed'].unique())} sementes, {len(raw)} execuções; "
           f"tempo total {t_total / 60:.1f} min.", "",
           "Tudo é contado por **incidente**: alarmes com a mesma chave `incidente` são um só; senão, alarmes até "
           "10 min depois do 1º alarme de um incidente entram nele (janela fixa). Detecção = cenários de falha com "
           "incidente depois do início; atraso mediano em min; isolação = (rolo, causa) certos no 1º incidente e "
           "no 1º par repetido em 2 incidentes; FA = incidentes em arquivos normais (por semana e limite superior "
           "95 % de Poisson exato); FA_pre_falha = incidentes antes do início nos cenários de falha; "
           "alarmes_brutos = alarmes antes do agrupamento.", "",
           "## Por família", "", fam.to_markdown(index=False), "",
           "## Por família e variante", "", resumo.to_markdown(index=False), "",
           "## Treino e calibração", "", pd.DataFrame(treinos).to_markdown(index=False), "",
           "## Por caso", "", casos.to_markdown(index=False), ""]
    erros = raw[raw["erro"] != ""]
    if len(erros):
        txt += ["## Erros", ""] + [f"- {r.arquitetura} {r.familia}/{r.variante}/{r.caso}/{r.seed}: {r.erro}"
                                   for r in erros.head(30).itertuples()] + [""]
    return "\n".join(txt), pd.concat([fam.assign(variante="(todas)"), resumo], ignore_index=True)


# ------------------------------------------------------------------ principal
def avaliar(specs, familias, seeds="dev", n_falha=None, n_normal=None, cal="padrao", meta=1.0, processos=4,
            log=print, cache_modelos=False):
    """Roda e devolve (raw DataFrame, lista de treinos). Usavel sem a CLI (testes)."""
    cens = C.montar(familias, seeds, n_falha, n_normal)
    plantas = sorted({c.planta for c in cens})
    tarefas_tr = [(s, p, cal, meta, cache_modelos) for s in specs for p in plantas]
    ctx = mp.get_context("spawn")
    t0 = time.time()
    if processos <= 1:
        tr = [_treinar(t) for t in tarefas_tr]
    else:
        with ctx.Pool(min(processos, len(tarefas_tr))) as pool:
            pool.map(_gerar_treino, [(p, cal) for p in plantas], chunksize=1)
            tr = pool.map(_treinar, tarefas_tr, chunksize=1)
    arqs = {(s, p): pickle.loads(b) for s, p, b, _, _ in tr}
    treinos = [{"spec": s, "planta": p, "t_simulacao_s": round(ts, 1), "t_treino_calibracao_s": round(tt, 1)}
               for s, p, _, ts, tt in tr]
    log(f"treino/calibracao: {time.time() - t0:.0f} s ({len(tr)} modelos)")
    cens.sort(key=lambda c: -c.horas * (3 if c.planta == "tensao_pb2" else 1))   # longos primeiro
    raw = []
    if processos <= 1:
        _init(pickle.dumps(arqs))
        it, pool = map(_rodar, cens), None
    else:
        pool = ctx.Pool(processos, initializer=_init, initargs=(pickle.dumps(arqs),))
        it = pool.imap_unordered(_rodar, cens, chunksize=1)
    try:
        for i, linhas in enumerate(it, 1):
            raw += linhas
            if i % max(1, len(cens) // 20) == 0 or i == len(cens):
                log(f"  {i}/{len(cens)} cenarios ({time.time() - t0:.0f} s)")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    raw = pd.DataFrame(raw).sort_values(["familia", "variante", "caso", "seed", "arquitetura"],
                                        kind="stable").reset_index(drop=True)
    return raw, treinos


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", choices=("dev", "final", "confirmacao"), default="dev")
    p.add_argument("--confirmar-final", action="store_true", help="obrigatorio com --seeds final")
    p.add_argument("--confirmar-confirmacao", action="store_true", help="obrigatorio com --seeds confirmacao")
    p.add_argument("--familias", nargs="+", default=list(C.FAMILIAS), choices=list(C.FAMILIAS))
    p.add_argument("--arquiteturas", nargs="+", default=["arquiteturas.atual:Conjunto", "arquiteturas.atual:Modelo2"])
    p.add_argument("--n-falha", type=int, help="repeticoes por caso de falha (padrao: cenarios.TAMANHO)")
    p.add_argument("--n-normal", type=int, help="arquivos normais de 48 h por variante (padrao: cenarios.TAMANHO)")
    p.add_argument("--cal", choices=("padrao", "longa"), default="padrao")
    p.add_argument("--cache-modelos", action="store_true",
                   help="reaproveita modelos treinados em benchmark/_cache (chave = hash do codigo); nao usar na final")
    p.add_argument("--meta", type=float, default=1.0, help="meta de FA por semana passada a calibrar()")
    p.add_argument("--processos", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--tag", default=None)
    a = p.parse_args(argv)
    if a.seeds == "final" and not a.confirmar_final:
        raise SystemExit("--seeds final e reservado ao orquestrador (rodar UMA vez): passe --confirmar-final")
    if a.seeds == "confirmacao" and not a.confirmar_confirmacao:
        raise SystemExit("--seeds confirmacao e reservado ao orquestrador (rodar UMA vez): passe "
                         "--confirmar-confirmacao")
    if a.seeds in C.RESERVADAS and a.cache_modelos:
        raise SystemExit(f"--cache-modelos nao vale em --seeds {a.seeds} (treinar do zero)")
    if a.seeds in C.RESERVADAS and (a.n_falha is not None or a.n_normal is not None):
        raise SystemExit(f"--seeds {a.seeds} usa o tamanho fixo de cenarios.TAMANHO['{a.seeds}']")
    a.tag = a.tag or a.seeds
    t0 = time.time()
    raw, treinos = avaliar(a.arquiteturas, a.familias, a.seeds, a.n_falha, a.n_normal, a.cal, a.meta, a.processos,
                           cache_modelos=a.cache_modelos)
    md, resumo = relatorio(raw, treinos, a, time.time() - t0)
    (AQUI / f"resultados_{a.tag}.md").write_text(md, encoding="utf-8")
    resumo.to_csv(AQUI / f"resultados_{a.tag}.csv", index=False)
    raw.to_csv(AQUI / f"resultados_{a.tag}_cenarios.csv", index=False)
    print(md)


if __name__ == "__main__":
    main()
