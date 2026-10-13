"""
Simulacao em MALHA FECHADA de 3 rolos de uma linha de papel (Festoon, P&B1, P&B2).

Diferenca para poc_deteccao.simular: la o erro do encoder aparece direto na
velocidade lida. Aqui cada drive controla a velocidade que o SEU encoder mede:

    erro de medicao do encoder:   d(t)
    velocidade real do rolo:      v = v_ref - PB(d)          PB = passa-baixa da malha
    velocidade lida pelo drive:   m = v + d = v_ref + (d - PB(d))

Logo, um erro lento (deriva de ganho) some da velocidade lida: o drive "corrige"
e o rolo passa a girar errado. O erro reaparece na tensao do papel e no torque:

    tensao no vao k:   T_k = T_(k-1) + EA * (v_saida - v_entrada) / v_linha   (com atraso de 1a ordem)
    torque do motor:   tau = (J/r) * dv/dt + b * v - r * (T_depois - T_antes)

Pontas: desbobinador e rebobinador controlam a tensao de entrada (150 N) e de saida (400 N).
Todos os P&B sao controlados por velocidade com draw fixo (sem malha de tensao local).
Todos os parametros sao hipoteses de ordem de grandeza, nao medicoes.
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from poc.poc_deteccao import FS, perfil_linha

EA = 1.2e6            # N   rigidez do papel (~0,3 mm x 1 m, E ~ 4 GPa)
R = 0.1               # m   raio do rolo
J_R = 15.0            # N.s inercia refletida dividida pelo raio (J/r)
B = 0.2               # N.m por m/s, atrito viscoso
DRAW = {"festoon": 0.0, "pb1": 1e-4, "pb2": 2e-4}     # ~120 N de tensao por vao
T_ENTRADA, T_SAIDA = 150.0, 400.0                    # N, controladas nas pontas
TAU_REF, TAU_MALHA, TAU_TENSAO = 1.0, 0.15, 0.3      # s
V_MIN = 1.0           # m/s, piso da velocidade de linha na formula da tensao
RUIDO_MED = 0.02 / 60  # m/s, inovacao do ruido de medicao de cada encoder (0,02 m/min)


def _pb(x, tau):
    a = 1 - np.exp(-1 / (FS * tau))
    return lfilter([a], [1, -(1 - a)], x)


def _ar(rng, n, phi, sigma):
    return lfilter([1], [1, -phi], rng.normal(0, sigma, n))


def simular_mf(horas, seed, falha=None, inicio_h=1.0, rampa_h=3.0):
    rng = np.random.default_rng(seed)
    sp = perfil_linha(horas, rng) / 60.0                       # m/s
    n = len(sp)
    sev = np.zeros(n)
    if falha:
        f0 = int(inicio_h * 3600 * FS)
        sev = np.clip((np.arange(n) - f0) / (rampa_h * 3600 * FS), 0, 1)
        sev[:f0] = 0.0
    rodando = sp > 0.5
    comum = _ar(rng, n, 0.99, 2e-4) * rodando                  # perturbacao comum da linha

    v, med = {}, {}
    for nome, draw in DRAW.items():
        ref = _pb(sp * (1 + draw), TAU_REF) + comum + _ar(rng, n, 0.9, 1e-5)
        d = _ar(rng, n, 0.7, RUIDO_MED)                        # erro normal de medicao
        if nome == "pb1" and falha == "deriva_ganho":          # acoplamento folgando: le a menos
            d = d - 3e-4 * sev * ref
        elif nome == "pb1" and falha == "perda_pulsos":        # quedas curtas na leitura
            p = np.where(sev > 0, 2e-4 + 3e-3 * sev, 0.0)
            for i in np.flatnonzero((rng.random(n) < p) & rodando):
                L = rng.integers(2, 10)
                d[i:i + L] -= rng.uniform(5e-4, 2e-3) * ref[i:i + L]
        elif nome == "pb1" and falha == "ruido":               # disco sujo / EMI
            d = d + _ar(rng, n, 0.7, 0.15 / 60) * sev
        v[nome] = ref - _pb(d, TAU_MALHA)                      # a malha corrige o que o encoder mede
        med[nome] = v[nome] + d

    vl = np.maximum(v["festoon"], V_MIN)
    T0 = T_ENTRADA + _ar(rng, n, 0.999, 0.3)
    T3 = T_SAIDA + _ar(rng, n, 0.999, 0.3)
    T1 = _pb(T0 + EA * (v["pb1"] - v["festoon"]) / vl, TAU_TENSAO)
    T2 = _pb(T1 + EA * (v["pb2"] - v["pb1"]) / vl, TAU_TENSAO)
    vizinhas = {"festoon": (T0, T1), "pb1": (T1, T2), "pb2": (T2, T3)}

    lam = np.full(n, 0.2 / 3600 / FS)
    if falha == "erros_com":
        lam = lam + 30 * sev / 3600 / FS

    df = pd.DataFrame({"t_s": np.arange(n) / FS, "v_setpoint": np.round(sp * 60, 2)})
    for nome in DRAW:
        antes, depois = vizinhas[nome]
        torque = (J_R * np.gradient(v[nome]) * FS + B * v[nome] - R * (depois - antes)
                  + _ar(rng, n, 0.5, 0.05))
        df[f"v_{nome}"] = np.round(med[nome] * 60, 2)          # m/min, como o drive mostra
        df[f"tq_{nome}"] = np.round(torque, 2)                 # N.m
    df["erros_pb1"] = np.cumsum(rng.poisson(lam))
    df["tensao_vao1"], df["tensao_vao2"] = np.round(T1, 1), np.round(T2, 1)
    df["v_real_pb1"] = np.round(v["pb1"] * 60, 3)
    df["severidade"] = sev
    return df
