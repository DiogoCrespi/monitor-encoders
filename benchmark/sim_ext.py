"""
Variantes dos simuladores para o benchmark (sem editar poc/malha_fechada_v2.py nem poc/simulador_avisos.py).

simular_ext  = copia de poc.malha_fechada_v2.simular_mf2 com parametros OPCIONAIS novos. Com todos os
               parametros no padrao, o resultado e IDENTICO ao de simular_mf2 (mesma sequencia do
               gerador; conferido em tests/test_benchmark.py). Os sorteios novos usam um gerador
               separado (default_rng([seed, 11])), para nao deslocar a sequencia original.

    ea_fator        EA da planta x fator (elasticidade do papel mudou; invisivel nos dados)
    draw_fator      DRAW de pb1 e pb2 x fator (o setpoint de draw aparece nas colunas draw_pb1/draw_pb2)
    atrito_fator    atrito (Coulomb, viscoso, extra frio) x fator; escalar ou {rolo: fator}
    receitas        troca de receita no meio do arquivo: lista de (t_s, id, {draw, tensao, vel}) em ordem
                    de tempo; a partir de t_s valem draw x draw, tensao das pontas x tensao, setpoint de
                    velocidade x vel. Coluna 'receita' = id em vigor (0 antes da 1a troca).
    pre_h           arquivo "quente": simula pre_h horas a mais ANTES e corta o arquivo no primeiro
                    instante de producao estavel depois disso (redutor quente, linha rodando; t_s
                    recomeca em 0). inicio_h da falha conta a partir do corte.
    ruido_hf        desvio (m/s) de um jitter de banda larga ACIMA da banda da malha: a malha nao reage
                    (a velocidade real nao muda), mas ele aparece na velocidade gravada. So com a linha
                    rodando. Representado ja amostrado a 10 Hz (branco por aliasing).
    frac_hf_ruido   na falha 'ruido', fracao da VARIANCIA do ruido da falha que fica acima da banda
                    (branca, visivel na velocidade gravada); o resto continua como no original.
    raio, relacao   {rolo: raio do rolo (m)}, {rolo: relacao de reducao motor/rolo}. O torque do motor
                    passa a ter braco r_ef = raio / relacao em vez de R (inercia refletida J / r_ef).
                    Com rolos diferentes as assinaturas de torque deixam de ser +-1.

simular_acoplado_ext = copia de poc.simulador_avisos.simular_acoplado com:
    taxa_acoplada_h  incidentes/h com severidade 1 nas falhas perda_pulsos/ruido (original: 2.0)
    fator_pos        taxa NORMAL de avisos x fator depois de t0 (taxa mudou depois do treino)
    aglomerado       {'media': 3, 'intervalo_s': 900}: avisos normais em aglomerados (mesma taxa media;
                     media de avisos por aglomerado, intervalo medio entre eles) - nao Poisson
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from poc import malha_fechada_v2 as mf2
from poc import simulador_avisos as sa
from poc.malha_fechada_v2 import (B_VISC, COULOMB, DELTA_DERIVA, DRAW, EA, FALHAS, FRIO_EXTRA, J_R, L_VAO, R,
                                  ROLOS, T_ENTRADA, T_SAIDA, TAU_AQUECE, TAU_ESFRIA, TAU_MALHA, TAU_REF,
                                  TAU_TENSAO_PI, V_MIN, ATRITO_FALHA, _alfa_vao, _ar, _pb, _pb_variavel)
from poc.poc_deteccao import FS, perfil_linha

COLUNAS_VERDADE = ("v_real_festoon", "v_real_pb1", "v_real_pb2", "tensao_vao1", "tensao_vao2", "severidade")


def _por_rolo(x, padrao=1.0):
    if isinstance(x, dict):
        return {r: float(x.get(r, padrao)) for r in ROLOS}
    return {r: float(x) for r in ROLOS}


def _corte_quente(sp, pre_n):
    """1o indice >= pre_n com setpoint de producao estavel ha 60 s."""
    jan = 60 * FS
    s = pd.Series(sp)
    estavel = ((s.rolling(jan).max() - s.rolling(jan).min()) < 1e-9) & (s > 400 / 60.0)
    idx = np.flatnonzero(estavel.to_numpy()[pre_n:])
    return pre_n + (int(idx[0]) if len(idx) else 0)


def simular_ext(horas, seed, falha=None, rolo="pb1", inicio_h=1.0, rampa_h=4.0, modo_tensao=(),
                ea_fator=1.0, draw_fator=1.0, atrito_fator=1.0, receitas=(), pre_h=0.0,
                ruido_hf=0.0, frac_hf_ruido=0.0, raio=None, relacao=None):
    """Mesma saida de simular_mf2 (+ colunas receita, draw_pb1, draw_pb2). Ver o docstring do modulo."""
    if falha is not None and falha not in FALHAS:
        raise ValueError(f"falha desconhecida: {falha}")
    if rolo not in ROLOS:
        raise ValueError(f"rolo desconhecido: {rolo}")
    if set(modo_tensao) - {"pb2"}:
        raise ValueError("modo_tensao so implementado para 'pb2'")
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng([int(seed), 11])               # sorteios novos (nao mexe na sequencia)
    sp = perfil_linha(horas + pre_h, rng) / 60.0                # m/s
    n = len(sp)
    corte = _corte_quente(sp, int(pre_h * 3600 * FS)) if pre_h > 0 else 0

    # receitas: fatores por amostra (no relogio do arquivo final, isto e, depois do corte)
    f_draw, f_tens, f_vel = np.full(n, float(draw_fator)), np.ones(n), np.ones(n)
    rid = np.zeros(n, int)
    for t_r, ident, fat in receitas:
        i0 = corte + int(round(t_r * FS))
        f_draw[i0:] = float(draw_fator) * fat.get("draw", 1.0)
        f_tens[i0:] = fat.get("tensao", 1.0)
        f_vel[i0:] = fat.get("vel", 1.0)
        rid[i0:] = ident
    if receitas:
        sp = sp * f_vel

    sev = np.zeros(n)
    if falha:
        f0 = corte + int(inicio_h * 3600 * FS)
        sev = np.clip((np.arange(n) - f0) / (rampa_h * 3600 * FS), 0, 1)
        sev[:f0] = 0.0
    rodando = sp > 0.5
    comum = _ar(rng, n, 0.99, 2e-4) * rodando

    ea = EA * ea_fator
    draw = {k: DRAW[k] * f_draw for k in ROLOS}
    ref, d = {}, {}
    hf_falha = np.zeros(n)
    for nome in ROLOS:
        ref[nome] = _pb(sp * (1 + draw[nome]), TAU_REF) + comum + _ar(rng, n, 0.9, 2e-6) * rodando
        d[nome] = _ar(rng, n, 0.7, 7e-6) * rodando + _ar(rng, n, 0.7, 1e-6)
        if nome != rolo or not falha:
            continue
        if falha == "deriva_ganho":
            d[nome] = d[nome] - DELTA_DERIVA * sev * ref[nome]
        elif falha == "perda_pulsos":
            p = np.where(sev > 0, 2e-4 + 2e-3 * sev, 0.0)
            for i in np.flatnonzero((rng.random(n) < p) & rodando):
                L = rng.integers(2, 10)
                d[nome][i:i + L] -= rng.uniform(1e-4, 4e-4) * ref[nome][i:i + L]
        elif falha == "ruido":
            if frac_hf_ruido > 0:
                # mesma variancia total: parte dentro da banda (como no original), parte acima
                d[nome] = d[nome] + _ar(rng, n, 0.7, 3e-5 * np.sqrt(1 - frac_hf_ruido)) * sev * rodando
                hf_falha = rng2.normal(0, 3e-5 / np.sqrt(1 - 0.49) * np.sqrt(frac_hf_ruido), n) * sev * rodando
            else:
                d[nome] = d[nome] + _ar(rng, n, 0.7, 3e-5) * sev * rodando

    alfa_malha = 1 - np.exp(-1 / (FS * TAU_MALHA))
    v = {k: ref[k] - lfilter([alfa_malha], [1, -(1 - alfa_malha)], d[k]) for k in ROLOS}

    vl = np.maximum(sp, V_MIN)
    T_ent, T_sai = T_ENTRADA * f_tens, T_SAIDA * f_tens
    T0 = T_ent * (1 + _ar(rng, n, 1 - 1 / (FS * 300), 0.01 * np.sqrt(2 / (FS * 300)))
                  + _ar(rng, n, 0.9, 0.005 * np.sqrt(0.19)))
    T3 = T_sai * (1 + _ar(rng, n, 1 - 1 / (FS * 300), 0.01 * np.sqrt(2 / (FS * 300)))
                  + _ar(rng, n, 0.9, 0.005 * np.sqrt(0.19)))
    a1, a2 = _alfa_vao(L_VAO[0], vl), _alfa_vao(L_VAO[1], vl)
    T1 = _pb_variavel(T0 + ea * (v["pb1"] - v["festoon"]) / vl, a1, T_ENTRADA + ea * DRAW["pb1"] * f_draw[0])
    trim = np.zeros(n)
    if "pb2" in modo_tensao:
        T2_sp = T_ent + ea * draw["pb2"]
        T2, trim = _vao2_malha_tensao(T1, v, vl, a2, rng, ea, T2_sp)
        v["pb2"] = v["pb2"] + trim
    else:
        T2 = _pb_variavel(T1 + ea * (v["pb2"] - v["pb1"]) / vl, a2, T_ENTRADA + ea * DRAW["pb2"] * f_draw[0])
    vizinhas = {"festoon": (T0, T1), "pb1": (T1, T2), "pb2": (T2, T3)}

    ta = np.where(rodando, 1 - np.exp(-1 / (FS * TAU_AQUECE)), 0.0)
    te = np.where(rodando, 0.0, 1 - np.exp(-1 / (FS * TAU_ESFRIA)))
    aq = _pb_variavel(np.where(rodando, 1.0, 0.0), 1 - ta - te, 0.0)

    fat = _por_rolo(atrito_fator)
    r_ef = {k: R for k in ROLOS}
    if raio is not None or relacao is not None:
        rr, ii = _por_rolo(raio or R, R), _por_rolo(relacao or 1.0)
        r_ef = {k: rr[k] / ii[k] for k in ROLOS}

    df = pd.DataFrame({"t_s": np.arange(n) / FS, "v_setpoint": np.round(sp * 60, 2)})
    hf = {k: np.zeros(n) for k in ROLOS}
    if ruido_hf > 0:
        for k in ROLOS:
            hf[k] = rng2.normal(0, ruido_hf, n) * rodando
    if falha == "ruido":
        hf[rolo] = hf[rolo] + hf_falha
    for nome in ROLOS:
        antes, depois = vizinhas[nome]
        atrito = (rodando * (COULOMB + FRIO_EXTRA * (1 - aq)) + B_VISC * v[nome])
        if fat[nome] != 1.0:
            atrito = atrito * fat[nome]
        atrito = atrito + _ar(rng, n, 1 - 1 / (FS * 3600), 0.3 * np.sqrt(2 / (FS * 3600)))
        if falha == "atrito" and nome == rolo:
            atrito = atrito + ATRITO_FALHA * sev * rodando
        jr = J_R if r_ef[nome] == R else J_R * R / r_ef[nome]
        torque = (jr * np.gradient(v[nome]) * FS + atrito + r_ef[nome] * (antes - depois)
                  + _ar(rng, n, 0.5, 0.5 * np.sqrt(0.75)))
        df[f"v_{nome}"] = np.round((v[nome] + d[nome] + hf[nome]) * 60, 3)
        df[f"tq_{nome}"] = np.round(torque, 2)
    for nome in ROLOS:
        df[f"v_real_{nome}"] = np.round(v[nome] * 60, 4)
    if "pb2" in modo_tensao:
        df["trim_pb2"] = np.round(trim * 60, 4)
    df["tensao_entrada"] = np.round(T0 + _ar(rng, n, 0.5, 2.0), 1)
    df["tensao_saida"] = np.round(T3 + _ar(rng, n, 0.5, 2.0), 1)
    df["tensao_vao1"], df["tensao_vao2"] = np.round(T1, 1), np.round(T2, 1)
    df["severidade"] = sev
    df["receita"] = rid
    df["draw_pb1"], df["draw_pb2"] = draw["pb1"], draw["pb2"]
    if corte:
        df = df.iloc[corte:].reset_index(drop=True)
        df["t_s"] = np.arange(len(df)) / FS
    return df


def _vao2_malha_tensao(T1, v, vl, a2, rng, ea, T2_sp):
    """Copia de mf2._vao2_com_malha_tensao com EA e setpoint (vetor) parametrizados."""
    n = len(T1)
    medida = _ar(rng, n, 0.5, 2.0)
    T2, trim = np.empty(n), np.empty(n)
    t2, integ = T2_sp[0], 0.0
    v1, v2 = v["pb1"], v["pb2"]
    for i in range(n):
        u = v2[i] + integ
        t2 = a2[i] * t2 + (1 - a2[i]) * (T1[i] + ea * (u - v1[i]) / vl[i])
        erro = T2_sp[i] - (t2 + medida[i])
        integ += vl[i] / (ea * TAU_TENSAO_PI) * erro / FS
        T2[i], trim[i] = t2, integ
    return T2, trim


def tempo_troca(df_ou_sp, t_alvo_s):
    """Inicio da 1a parada (setpoint 0) em t >= t_alvo_s - onde a troca de receita e feita."""
    sp = np.asarray(df_ou_sp)
    i0 = int(t_alvo_s * FS)
    idx = np.flatnonzero(sp[i0:] <= 0)
    return (i0 + int(idx[0])) / FS if len(idx) else t_alvo_s


def perfil(horas, seed):
    """O mesmo setpoint (m/s) que simular_ext/simular_mf2 sorteiam para esta semente (sem receitas)."""
    return perfil_linha(horas, np.random.default_rng(seed)) / 60.0


# ------------------------------------------------------------------ avisos
def _pontos_aglomerados(rng, taxa_h, t0, t1, aglomerado):
    if not aglomerado:
        return sa._pontos(rng, taxa_h, t0, t1)
    m, dt = float(aglomerado.get("media", 3.0)), float(aglomerado.get("intervalo_s", 900.0))
    centros = sa._pontos(rng, lambda t: taxa_h(t) / m, t0, t1)
    t = []
    for c in centros:
        k = rng.geometric(1 / m)
        t.extend(c + np.concatenate([[0.0], np.cumsum(rng.exponential(dt, k - 1))]))
    t = np.sort(np.asarray(t, float))
    return t[t < t1]


def _normal_ext(rng, t0, t1, encoders, t_pos, fator_pos, aglomerado):
    HIP = sa.HIP
    linhas = []
    pos = (lambda t: np.where(np.asarray(t) >= t_pos, fator_pos, 1.0)) if fator_pos != 1.0 else (lambda t: 1.0)
    for k, (enc, drive) in enumerate(encoders.items()):
        fd = sa._fator_diario(rng, t0, t1)
        ln = lambda: rng.lognormal(0, HIP["sigma_lognormal"])
        if drive == "tipo_a":
            tc = HIP["taxa_cabo_a_h"] * ln()
            ts = HIP["taxa_sinal_a_h"] * ln()
            linhas += sa._incrementos_cabo(rng, _pontos_aglomerados(rng, lambda t: tc * fd(t) * pos(t), t0, t1,
                                                                    aglomerado), drive, enc, k + 1)
        else:
            ts = HIP["taxa_sinal_b_h"] * ln()
        linhas += sa._avisos_sinal(rng, _pontos_aglomerados(rng, lambda t: ts * fd(t) * pos(t), t0, t1, aglomerado),
                                   drive, enc)
    tipo_a = [e for e, d in encoders.items() if d == "tipo_a"]
    rx = [e for e, d in encoders.items() if d == "tipo_b"]
    for t in sa._pontos(rng, lambda t: np.full(len(t), HIP["taxa_emc_h"]) * pos(t), t0, t1):
        if tipo_a:
            for enc in rng.choice(tipo_a, rng.integers(1, len(tipo_a) + 1), replace=False):
                k = list(encoders).index(enc)
                tt = t + rng.uniform(0, 20, rng.integers(1, 7))
                linhas += sa._incrementos_cabo(rng, np.sort(tt), "tipo_a", enc, k + 1)
        for enc in rx:
            if rng.random() < HIP["emc_prob_b"]:
                linhas.append((t + rng.uniform(0, 20), "tipo_b", enc, "aviso_sinal_b", "aviso", np.nan))
    return linhas


def simular_acoplado_ext(seed, t_s, severidade, falha=None, rolo="pb1", inicio_s=3600.0, dias_hist=14.0,
                         encoders=None, taxa_acoplada_h=sa.TAXA_ACOPLADA_H, fator_pos=1.0, aglomerado=None):
    """simular_acoplado com taxa acoplada, fator de taxa normal depois do treino e aglomerados (padrao = original)."""
    encoders = sa.ENCODERS_LINHA if encoders is None else encoders
    rng = np.random.default_rng([int(seed), 7])
    t_s = np.asarray(t_s, float)
    t0 = dias_hist * sa.DIA
    t_fim = t0 + float(t_s[-1]) + (t_s[1] - t_s[0] if len(t_s) > 1 else 0.0)
    linhas = _normal_ext(rng, 0.0, t_fim, encoders, t0, fator_pos, aglomerado)
    drive = encoders.get(rolo)
    comp = list(encoders).index(rolo) + 1 if rolo in encoders else 0
    info = {"seed": seed, "falha": falha, "encoder": rolo, "t0_s": t0, "t_fim_hist_s": t0,
            "t_inicio_s": t0 + inicio_s, "t_fim_s": t_fim, "t_falha_s": None}
    if falha in ("perda_pulsos", "ruido"):
        sev = np.asarray(severidade, float)
        taxa = lambda t: taxa_acoplada_h * np.interp(np.asarray(t) - t0, t_s, sev)
        tempos = sa._pontos(rng, taxa, t0, t_fim)
        if falha == "perda_pulsos" and drive == "tipo_a":
            linhas += sa._incrementos_cabo(rng, tempos, drive, rolo, comp)
        else:
            linhas += sa._avisos_sinal(rng, tempos, drive, rolo)
    elif falha in sa.CENARIOS_SO_AVISOS:
        cen = sa.CENARIOS_SO_AVISOS[falha]
        if cen == "cabo" and drive != "tipo_a":
            raise ValueError("erros_com so existe em encoder tipo_a (contador de erros)")
        t_falha = t0 + float(t_s[-1])
        linhas += sa._degradacao(rng, cen, rolo, drive, comp, t0 + inicio_s, t_falha)
        info["t_falha_s"] = t_falha
    elif falha not in (None, "deriva_ganho", "atrito"):
        raise ValueError(f"falha desconhecida: {falha}")
    df = sa._tabela(linhas)
    df = df[df["t_s"] < t_fim].reset_index(drop=True)
    return sa.eventos_do_log(df), sa.contadores(df, 60.0, t_fim), info


__all__ = ["simular_ext", "simular_acoplado_ext", "tempo_troca", "perfil", "COLUNAS_VERDADE", "mf2"]
