"""
Simulador v2 em MALHA FECHADA de 3 rolos de uma linha de papel (Festoon, P&B1, P&B2).

Revisao do poc/malha_fechada.py com parametros de ordem de grandeza da fisica da tensao:

  * malha de velocidade rapida (TAU_MALHA = 10 ms, ~16 Hz de banda; antes 0,15 s);
  * atraso da tensao no vao tau = L / v, variavel com a velocidade (antes 0,3 s fixo);
  * EA = 2,0e6 N por metro de largura (cartao ~0,35 mm, E ~ 5,7 GPa; antes 1,2e6);
  * tensao de operacao 700-900 N/m (cartao 12-18 pt; antes 150-400);
  * deriva de encoder pequena e plausivel: 0,005 % no fim da rampa (antes 0,03 %),
    que corresponde a ~4 K de dilatacao de um rolo de aco (erro de raio efetivo);
  * falha injetavel em QUALQUER rolo (festoon, pb1, pb2);
  * modo opcional de rolo com malha de tensao (celula de carga + trim de velocidade PI);
  * perturbacoes "normais" que um monitor real enfrenta: tensao das pontas (bailarinos)
    vagueando, atrito variando devagar por rolo e aquecimento do redutor depois da partida.

Modelo (desvios pequenos, sem escorregamento, papel linear elastico):

    velocidade real    v_k = ref_k - PB_malha(d_k)          d_k = erro de medicao do encoder k
    velocidade lida    m_k = v_k + d_k
    tensao no vao j    L_j dT_j/dt = v (T_(j-1) - T_j) + EA (v_saida - v_entrada)
                       -> T_j = PB_{tau = L_j/v}( T_(j-1) + EA (v_saida - v_entrada) / v )
    torque do motor    M_k = (J/r) dv_k/dt + atrito_k(v, temperatura) + r (T_antes - T_depois)

Em regime, T1 = T0 + EA (v_pb1 - v_fest)/v e T2 = T0 + EA (v_pb2 - v_fest)/v, entao um rolo
que gira rapido demais (encoder lendo a menos) por uma fracao delta muda os torques assim
(k = EA * r * delta):

    rolo com erro    festoon   P&B1   P&B2
    festoon            +k        0     -k
    P&B1               -k       +k      0
    P&B2                0       -k     +k

As colunas somam zero: a soma dos tres torques = r (T0 - T3) nao depende dos encoders.
Esta tabela e a base da isolacao em monitor/torque_v2.py.

PREMISSAS (hipoteses de ordem de grandeza, nao medicoes):
  - largura 1 m, entao "N" de tensao = "N/m";
  - pontas (desbobinador / rebobinador) controlam T0 e T3 com bailarino: valor medio fixo,
    variacao lenta de ~1 % e rapida de ~0,5 %;
  - rolos sem escorregamento (o escorregamento reduziria o sinal no torque);
  - encoder de alta resolucao: ruido de medicao ~1e-6 da velocidade;
  - atrito = Coulomb 3 N.m + viscoso 0,5 N.m/(m/s) + extra de 1,5 N.m com redutor frio, que some
    com tau de 20 min de operacao (e volta em parada, tau 60 min) + passeio lento de 0,3 N.m;
  - ruido do torque lido ~0,5 N.m, AR(0,5) a 10 Hz;
  - a amostragem de saida e 10 Hz (coletor/CLP); a malha de velocidade, mais rapida que isso,
    e representada pelo seu efeito na amostra (filtro discreto exato de 1a ordem).
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from poc.poc_deteccao import FS, perfil_linha

ROLOS = ("festoon", "pb1", "pb2")
FALHAS = ("deriva_ganho", "perda_pulsos", "ruido", "atrito")

EA = 2.0e6               # N    rigidez do papel por metro de largura (E*t)
R = 0.15                 # m    raio dos rolos
J_R = 2.0 / R            # N.s  inercia (2 kg.m2) dividida pelo raio
B_VISC = 0.5             # N.m por m/s
COULOMB = 3.0            # N.m  atrito seco com a linha rodando
FRIO_EXTRA = 1.5         # N.m  atrito a mais com o redutor frio
TAU_AQUECE, TAU_ESFRIA = 20 * 60.0, 60 * 60.0   # s
L_VAO = (2.0, 1.5)       # m    vao festoon->P&B1 e P&B1->P&B2
T_ENTRADA, T_SAIDA = 700.0, 900.0               # N, controladas nas pontas
DRAW = {"festoon": 0.0, "pb1": 5e-5, "pb2": 1e-4}  # T1 = 800 N, T2 = 900 N
TAU_REF = 1.0            # s    rampa de referencia
TAU_MALHA = 0.01         # s    malha de velocidade (~16 Hz)
TAU_TENSAO_PI = 2.0      # s    malha de tensao (trim), quando ligada
V_MIN = 1.0              # m/s  piso de velocidade para tau = L/v e para o ganho EA/v
DELTA_DERIVA = 5e-5      # fracao da velocidade no fim da rampa (0,005 %)
ATRITO_FALHA = 6.0       # N.m  aumento de atrito (rolamento) no fim da rampa


def _pb(x, tau):
    a = 1 - np.exp(-1 / (FS * tau))
    return lfilter([a], [1, -(1 - a)], x)


def _ar(rng, n, phi, sigma):
    return lfilter([1], [1, -phi], rng.normal(0, sigma, n))


def _pb_variavel(u, a, y0, bloco=128):
    """y[i] = a[i] y[i-1] + (1 - a[i]) u[i] com a[i] variavel, vetorizado por blocos."""
    y = np.empty_like(u)
    prev = y0
    for s in range(0, len(u), bloco):
        e = min(s + bloco, len(u))
        la = np.cumsum(np.log(a[s:e]))
        y[s:e] = np.exp(la) * (prev + np.cumsum((1 - a[s:e]) * u[s:e] * np.exp(-la)))
        prev = y[e - 1]
    return y


def _alfa_vao(L, vl):
    return np.exp(-vl / (FS * L))          # tau = L / v


def simular_mf2(horas, seed, falha=None, rolo="pb1", inicio_h=1.0, rampa_h=4.0,
                modo_tensao=()):
    """Gera um DataFrame no formato do monitor (+ colunas de verdade de campo).

    falha: None ou um de FALHAS, aplicada ao `rolo` (festoon, pb1 ou pb2).
    modo_tensao: rolos com malha de tensao local; hoje so 'pb2' (celula de carga no vao 2,
                 PI corrigindo a referencia de velocidade -> coluna trim_pb2).
    """
    if falha is not None and falha not in FALHAS:
        raise ValueError(f"falha desconhecida: {falha}")
    if rolo not in ROLOS:
        raise ValueError(f"rolo desconhecido: {rolo}")
    if set(modo_tensao) - {"pb2"}:
        raise ValueError("modo_tensao so implementado para 'pb2'")
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

    ref, d = {}, {}
    for nome, draw in DRAW.items():
        ref[nome] = _pb(sp * (1 + draw), TAU_REF) + comum + _ar(rng, n, 0.9, 2e-6) * rodando
        d[nome] = _ar(rng, n, 0.7, 7e-6) * rodando + _ar(rng, n, 0.7, 1e-6)
        if nome != rolo or not falha:
            continue
        if falha == "deriva_ganho":                           # le a menos: rolo gira rapido
            d[nome] = d[nome] - DELTA_DERIVA * sev * ref[nome]
        elif falha == "perda_pulsos":                         # quedas curtas na leitura
            p = np.where(sev > 0, 2e-4 + 2e-3 * sev, 0.0)
            for i in np.flatnonzero((rng.random(n) < p) & rodando):
                L = rng.integers(2, 10)
                d[nome][i:i + L] -= rng.uniform(1e-4, 4e-4) * ref[nome][i:i + L]
        elif falha == "ruido":                                # disco sujo / EMI
            d[nome] = d[nome] + _ar(rng, n, 0.7, 3e-5) * sev * rodando

    alfa_malha = 1 - np.exp(-1 / (FS * TAU_MALHA))
    v = {k: ref[k] - lfilter([alfa_malha], [1, -(1 - alfa_malha)], d[k]) for k in ROLOS}

    vl = np.maximum(sp, V_MIN)
    T0 = T_ENTRADA * (1 + _ar(rng, n, 1 - 1 / (FS * 300), 0.01 * np.sqrt(2 / (FS * 300)))
                      + _ar(rng, n, 0.9, 0.005 * np.sqrt(0.19)))
    T3 = T_SAIDA * (1 + _ar(rng, n, 1 - 1 / (FS * 300), 0.01 * np.sqrt(2 / (FS * 300)))
                    + _ar(rng, n, 0.9, 0.005 * np.sqrt(0.19)))
    a1, a2 = _alfa_vao(L_VAO[0], vl), _alfa_vao(L_VAO[1], vl)
    T1 = _pb_variavel(T0 + EA * (v["pb1"] - v["festoon"]) / vl, a1, T_ENTRADA + EA * DRAW["pb1"])
    trim = np.zeros(n)
    if "pb2" in modo_tensao:
        T2, trim = _vao2_com_malha_tensao(T1, v, vl, a2, rng)
        v["pb2"] = v["pb2"] + trim
    else:
        T2 = _pb_variavel(T1 + EA * (v["pb2"] - v["pb1"]) / vl, a2, T_ENTRADA + EA * DRAW["pb2"])
    vizinhas = {"festoon": (T0, T1), "pb1": (T1, T2), "pb2": (T2, T3)}

    # temperatura do redutor (0 = frio, 1 = quente): aquece rodando, esfria parado
    ta = np.where(rodando, 1 - np.exp(-1 / (FS * TAU_AQUECE)), 0.0)
    te = np.where(rodando, 0.0, 1 - np.exp(-1 / (FS * TAU_ESFRIA)))
    aq = _pb_variavel(np.where(rodando, 1.0, 0.0), 1 - ta - te, 0.0)

    df = pd.DataFrame({"t_s": np.arange(n) / FS, "v_setpoint": np.round(sp * 60, 2)})
    for nome in ROLOS:
        antes, depois = vizinhas[nome]
        atrito = (rodando * (COULOMB + FRIO_EXTRA * (1 - aq)) + B_VISC * v[nome]
                  + _ar(rng, n, 1 - 1 / (FS * 3600), 0.3 * np.sqrt(2 / (FS * 3600))))
        if falha == "atrito" and nome == rolo:
            atrito = atrito + ATRITO_FALHA * sev * rodando
        torque = (J_R * np.gradient(v[nome]) * FS + atrito + R * (antes - depois)
                  + _ar(rng, n, 0.5, 0.5 * np.sqrt(0.75)))
        df[f"v_{nome}"] = np.round((v[nome] + d[nome]) * 60, 3)   # m/min, lida pelo drive
        df[f"tq_{nome}"] = np.round(torque, 2)                    # N.m
    for nome in ROLOS:
        df[f"v_real_{nome}"] = np.round(v[nome] * 60, 4)
    if "pb2" in modo_tensao:
        df["trim_pb2"] = np.round(trim * 60, 4)                   # m/min
    # tensao das pontas como o CLP/drive enxerga (celula de carga ou bailarino), com ruido
    df["tensao_entrada"] = np.round(T0 + _ar(rng, n, 0.5, 2.0), 1)
    df["tensao_saida"] = np.round(T3 + _ar(rng, n, 0.5, 2.0), 1)
    df["tensao_vao1"], df["tensao_vao2"] = np.round(T1, 1), np.round(T2, 1)
    df["severidade"] = sev
    return df


def _vao2_com_malha_tensao(T1, v, vl, a2, rng):
    """Vao 2 com celula de carga e PI corrigindo a referencia do P&B2 (laco por amostra)."""
    n = len(T1)
    T2_sp = T_ENTRADA + EA * DRAW["pb2"]
    medida = _ar(rng, n, 0.5, 2.0)                             # ruido da celula de carga (N)
    T2, trim = np.empty(n), np.empty(n)
    t2, integ = T2_sp, 0.0
    v1, v2 = v["pb1"], v["pb2"]
    for i in range(n):
        u = v2[i] + integ
        t2 = a2[i] * t2 + (1 - a2[i]) * (T1[i] + EA * (u - v1[i]) / vl[i])
        erro = T2_sp - (t2 + medida[i])
        integ += vl[i] / (EA * TAU_TENSAO_PI) * erro / FS     # ganho integral ~ v/(EA tau)
        T2[i], trim[i] = t2, integ
    return T2, trim


ASSINATURA = {           # efeito no torque de cada rolo de um encoder LENDO A MENOS (rolo rapido)
    "festoon": np.array([+1.0, 0.0, -1.0]),
    "pb1": np.array([-1.0, +1.0, 0.0]),
    "pb2": np.array([0.0, -1.0, +1.0]),
}
