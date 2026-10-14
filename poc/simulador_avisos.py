"""
Simulador de LOG DE EVENTOS (avisos e falhas) dos drives de uma linha de papel.

Gera o que se leria do historico de alarmes/falhas do drive e o contador de erros
de comunicacao do encoder amostrado. Codigos com nomes genericos (ver monitor/avisos.py).

    eventos   : t_s, drive, encoder, codigo, tipo ('aviso'|'falha'), valor
    contadores: t_s, encoder, contador ('contador_erros'), valor (ACUMULADO)

TODOS OS PARAMETROS ABAIXO SAO HIPOTESES (ordem de grandeza), nao medidas da planta.
Nao existe fonte publica com taxa normal de avisos nem com a antecedencia entre o
primeiro aviso e a falha. Eles servem para testar
o metodo, nao para prever o numero real de alarmes.

Cenarios de degradacao (sempre num unico encoder, terminando na falha final):
    cabo         cabo/conector de comunicacao piorando: contador_erros incrementa cada vez mais
                 (aviso_cabo a cada incremento) ate falha_cabo.               (so tipo_a)
    amplitude    disco sujo / LED envelhecendo: amplitude do sinal cai e os avisos de
                 nivel aparecem cada vez mais (tipo_a: aviso_sinal/aviso_encoder, valor = amplitude
                 em mV, ate falha_sinal; tipo_b: aviso_sinal_b ate falha_sinal_b)
    intermitente incidentes cada vez mais proximos ("horas a dias entre incidentes"),
                 , cada um com rajada de avisos e as vezes uma falha
                 transitoria que se reconhece e a linha volta, ate a falha final
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

# 4 encoders tipo_a (com contador de erros de comunicacao) e 6 tipo_b (so avisos de sinal)
ENCODERS = {**{f"a_{i}": "tipo_a" for i in range(1, 5)},
            **{f"b_{i}": "tipo_b" for i in range(1, 7)}}
CENARIOS = ("cabo", "amplitude", "intermitente")
DIA = 86400.0
PASSO = 60.0                       # grade de geracao (s)

# ---------------------------------------------------------------- HIPOTESES
HIP = {
    # taxa normal de "incidentes" por encoder (por hora), mediana e dispersao lognormal entre encoders
    "taxa_cabo_a_h": 0.010,      # ~1 incidente de cabo a cada 4 dias
    "taxa_sinal_a_h": 0.002,     # aviso_sinal/aviso_encoder esporadico: ~1 a cada 3 semanas
    "taxa_sinal_b_h": 0.004,       # aviso_sinal_b esporadico: ~1 a cada 10 dias
    "sigma_lognormal": 0.7,         # encoders diferentes tem taxas diferentes
    "fator_diario_forma": 3.0,      # variacao dia a dia (gama, media 1): umidade, mix de producao
    "incrementos_por_incidente": 1.5,   # media (geometrica) de incrementos de contador_erros por incidente
    # rajada de EMC na linha inteira (partida de motor grande, solda, contator)
    "taxa_emc_h": 0.01,             # ~1 a cada 4 dias
    "emc_prob_b": 0.3,             # chance de cada tipo_b gerar aviso_sinal_b numa rajada
    # degradacao
    "cabo_dur_dias": (0.5, 7.0),    # duracao uniforme
    "cabo_taxa_final_h": 20.0,      # incidentes/h no fim (x fator lognormal)
    "cabo_expoente": 3.0,           # comeco lento: taxa ~ u^3
    "amp_dur_dias": (1.0, 10.0),
    "amp_inicial_mv": 450.0, "amp_final_mv": 130.0, "amp_aviso_mv": 230.0, "amp_ruido_mv": 40.0,
    "amp_taxa_max_h": 6.0,          # taxa de avisos quando a amplitude media esta bem abaixo do limite
    "int_dur_dias": (0.25, 5.0),
    "int_taxa_ini_h": 1 / 48,       # 1 incidente a cada 2 dias ...
    "int_taxa_fim_h": 0.5,          # ... ate 1 a cada 2 h
    "int_eventos_por_incidente": (2, 8),
    "int_prob_falha_transitoria": 0.2,
    "fator_severidade_sigma": 0.6,  # cenarios mais fracos ou mais fortes (lognormal)
    "pre_dias": (3.0, 10.0),        # operacao normal monitorada antes de a degradacao comecar
}

CODIGOS = {
    ("tipo_a", "cabo"): ("aviso_cabo", "falha_cabo"),
    ("tipo_a", "sinal"): ("aviso_sinal", "falha_sinal"),
    ("tipo_b", "sinal"): ("aviso_sinal_b", "falha_sinal_b"),
}


def _pontos(rng, taxa_h, t0, t1):
    """Tempos de um processo de Poisson nao homogeneo; taxa_h(t_s) em eventos/h (vetorizada)."""
    grade = np.arange(t0, t1, PASSO)
    if len(grade) == 0:
        return np.array([])
    lam = np.clip(taxa_h(grade), 0, None) * PASSO / 3600
    n = rng.poisson(lam)
    t = np.repeat(grade, n) + rng.uniform(0, PASSO, n.sum())
    return np.sort(t[t < t1])


def _fator_diario(rng, t0, t1):
    dias = int(np.ceil((t1 - t0) / DIA)) + 1
    f = rng.gamma(HIP["fator_diario_forma"], 1 / HIP["fator_diario_forma"], dias)
    return lambda t: f[((np.asarray(t) - t0) // DIA).astype(int)]


def _incrementos_cabo(rng, tempos, drive, enc, comp):
    """Cada incidente de cabo gera >=1 incremento de contador_erros; aviso_cabo so e registrado se o
    anterior ja sumiu (o alarme some sozinho em 5 s), como no historico do drive."""
    linhas = []
    ultimo_alarme = -np.inf
    p = 1 / HIP["incrementos_por_incidente"]
    for t in tempos:
        for j in range(rng.geometric(p)):
            ti = t + j * rng.uniform(0.2, 3.0)
            linhas.append((ti, drive, enc, "contador_erros+", "contagem", 1.0))
            if ti - ultimo_alarme > 5.0:
                linhas.append((ti, drive, enc, "aviso_cabo", "aviso", float(comp)))
                ultimo_alarme = ti
    return linhas


def _avisos_sinal(rng, tempos, drive, enc, amp=None):
    linhas = []
    for t in tempos:
        if drive == "tipo_a":
            a = amp(t) if amp else HIP["amp_aviso_mv"]
            valor = float(np.clip(a + rng.normal(0, HIP["amp_ruido_mv"]), 60, HIP["amp_aviso_mv"]))
            codigo = "aviso_sinal" if rng.random() < 0.8 else "aviso_encoder"
            linhas.append((t, drive, enc, codigo, "aviso", round(valor)))
        else:
            linhas.append((t, drive, enc, "aviso_sinal_b", "aviso", np.nan))
    return linhas


def _normal(rng, t0, t1, encoders):
    linhas = []
    for k, (enc, drive) in enumerate(encoders.items()):
        fd = _fator_diario(rng, t0, t1)
        ln = lambda: rng.lognormal(0, HIP["sigma_lognormal"])
        if drive == "tipo_a":
            tc = HIP["taxa_cabo_a_h"] * ln()
            ts = HIP["taxa_sinal_a_h"] * ln()
            linhas += _incrementos_cabo(rng, _pontos(rng, lambda t: tc * fd(t), t0, t1), drive, enc, k + 1)
        else:
            ts = HIP["taxa_sinal_b_h"] * ln()
        linhas += _avisos_sinal(rng, _pontos(rng, lambda t: ts * fd(t), t0, t1), drive, enc)
    # rajadas de EMC (causa comum, varios encoders ao mesmo tempo)
    tipo_a = [e for e, d in encoders.items() if d == "tipo_a"]
    tipo_b_lista = [e for e, d in encoders.items() if d == "tipo_b"]
    for t in _pontos(rng, lambda t: np.full(len(t), HIP["taxa_emc_h"]), t0, t1):
        if tipo_a:
            for enc in rng.choice(tipo_a, rng.integers(1, len(tipo_a) + 1), replace=False):
                k = list(encoders).index(enc)
                tt = t + rng.uniform(0, 20, rng.integers(1, 7))
                linhas += _incrementos_cabo(rng, np.sort(tt), "tipo_a", enc, k + 1)
        for enc in tipo_b_lista:
            if rng.random() < HIP["emc_prob_b"]:
                linhas.append((t + rng.uniform(0, 20), "tipo_b", enc, "aviso_sinal_b", "aviso", np.nan))
    return linhas


def _degradacao(rng, cenario, enc, drive, comp, t_ini, t_fim):
    dur = t_fim - t_ini
    u = lambda t: np.clip((np.asarray(t) - t_ini) / dur, 0, 1)
    fs = rng.lognormal(0, HIP["fator_severidade_sigma"])
    linhas = []
    if cenario == "cabo":
        taxa = lambda t: HIP["cabo_taxa_final_h"] * fs * u(t) ** HIP["cabo_expoente"]
        linhas += _incrementos_cabo(rng, _pontos(rng, taxa, t_ini, t_fim), drive, enc, comp)
    elif cenario == "amplitude":
        a0, a1 = HIP["amp_inicial_mv"], HIP["amp_final_mv"]
        amp = lambda t: a0 - (a0 - a1) * u(t) ** 1.5
        taxa = lambda t: HIP["amp_taxa_max_h"] * fs * norm.cdf((HIP["amp_aviso_mv"] - amp(t)) / HIP["amp_ruido_mv"])
        linhas += _avisos_sinal(rng, _pontos(rng, taxa, t_ini, t_fim), drive, enc, amp)
    elif cenario == "intermitente":
        r0, r1 = HIP["int_taxa_ini_h"], HIP["int_taxa_fim_h"] * fs
        taxa = lambda t: r0 + (r1 - r0) * u(t)
        aviso, falha = CODIGOS[(drive, "cabo" if drive == "tipo_a" else "sinal")]
        for t in _pontos(rng, taxa, t_ini, t_fim):
            n = rng.integers(*HIP["int_eventos_por_incidente"], endpoint=True)
            tt = np.sort(t + rng.uniform(0, 120, n))
            if drive == "tipo_a":
                linhas += _incrementos_cabo(rng, tt, drive, enc, comp)
            else:
                linhas += _avisos_sinal(rng, tt, drive, enc)
            if rng.random() < HIP["int_prob_falha_transitoria"]:
                linhas.append((tt[-1] + 1.0, drive, enc, falha, "falha", np.nan))
    else:
        raise ValueError(f"cenario desconhecido: {cenario}")
    # falha final (para a maquina; exige troca)
    fam = "cabo" if cenario == "cabo" or (cenario == "intermitente" and drive == "tipo_a") else "sinal"
    _, falha = CODIGOS[(drive, fam)]
    valor = float(HIP["amp_final_mv"]) if (cenario == "amplitude" and drive == "tipo_a") else np.nan
    linhas.append((t_fim, drive, enc, falha, "falha", valor))
    return linhas


def _tabela(linhas):
    df = pd.DataFrame(linhas, columns=["t_s", "drive", "encoder", "codigo", "tipo", "valor"])
    return df.sort_values("t_s", kind="stable").reset_index(drop=True)


def simular(seed, cenario=None, dias_hist=14.0, dias_monit=28.0, encoders=ENCODERS):
    """Simula um historico normal (para treino) seguido de um periodo monitorado.

    cenario=None: periodo monitorado todo normal (dias_monit dias).
    cenario em CENARIOS: normal por HIP['pre_dias'], depois degradacao num encoder ate a
    falha final; o periodo monitorado termina na falha.

    Devolve (eventos, info). Em eventos, as linhas codigo='contador_erros+' sao incrementos do
    contador (use contadores() para amostra-lo e eventos_do_log() para o log puro).
    """
    rng = np.random.default_rng(seed)
    t_hist = dias_hist * DIA
    info = {"seed": seed, "cenario": cenario, "t_fim_hist_s": t_hist, "encoder": None,
            "drive": None, "t_inicio_s": None, "t_falha_s": None}
    if cenario is None:
        t_fim = t_hist + dias_monit * DIA
        linhas = _normal(rng, 0.0, t_fim, encoders)
    else:
        cand = [e for e, d in encoders.items() if cenario != "cabo" or d == "tipo_a"]
        enc = str(rng.choice(cand))
        drive = encoders[enc]
        t_ini = t_hist + rng.uniform(*HIP["pre_dias"]) * DIA
        chave = {"cabo": "cabo_dur_dias", "amplitude": "amp_dur_dias", "intermitente": "int_dur_dias"}[cenario]
        t_fim = t_ini + rng.uniform(*HIP[chave]) * DIA
        linhas = _normal(rng, 0.0, t_fim, encoders)
        linhas += _degradacao(rng, cenario, enc, drive, list(encoders).index(enc) + 1, t_ini, t_fim)
        info.update(encoder=enc, drive=drive, t_inicio_s=t_ini, t_falha_s=t_fim)
    info["t_fim_s"] = t_fim
    df = _tabela(linhas)
    return df[df["t_s"] <= t_fim].reset_index(drop=True), info


def eventos_do_log(df):
    """So o que aparece no historico de alarmes/falhas (sem os incrementos do contador)."""
    return df[df["codigo"] != "contador_erros+"].reset_index(drop=True)


def contadores(df, periodo_s=60.0, t_fim=None):
    """Amostra o contador_erros acumulado de cada encoder tipo_a a cada periodo_s (formato CONTADORES)."""
    inc = df[df["codigo"] == "contador_erros+"]
    t_fim = float(df["t_s"].max()) if t_fim is None else t_fim
    grade = np.arange(0.0, t_fim + periodo_s, periodo_s)
    saida = []
    for enc in sorted(df.loc[df["drive"] == "tipo_a", "encoder"].unique()):
        t = np.sort(inc.loc[inc["encoder"] == enc, "t_s"].to_numpy())
        saida.append(pd.DataFrame({"t_s": grade, "encoder": enc, "contador": "contador_erros",
                                   "valor": np.searchsorted(t, grade, side="right")}))
    return pd.concat(saida, ignore_index=True) if saida else pd.DataFrame(
        columns=["t_s", "encoder", "contador", "valor"])


# ---------------------------------------------------------------- acoplamento com a malha fechada
# Usado na validacao conjunta (validacao_conjunta.py): o log de avisos de uma linha com os tres
# rolos de poc/malha_fechada_v2.py, coerente com a falha injetada no cenario.
# TUDO AQUI E HIPOTESE (nao ha dado da planta sobre quais avisos acompanham cada falha).
ENCODERS_LINHA = {"festoon": "tipo_a", "pb1": "tipo_a", "pb2": "tipo_b"}
TAXA_ACOPLADA_H = 2.0            # incidentes/h com severidade 1 (perda de pulsos, ruido)
CENARIOS_SO_AVISOS = {"erros_com": "cabo", "degradacao_sinal": "amplitude"}


def simular_acoplado(seed, t_s, severidade, falha=None, rolo="pb1", inicio_s=3600.0,
                     dias_hist=14.0, encoders=None):
    """Log de avisos coerente com um cenario de poc.malha_fechada_v2.simular_mf2.

    t_s, severidade: colunas do DataFrame do cenario (tempo do cenario, severidade 0-1).
    O log tem dias_hist dias de historico normal (treino do Modelo 1) e depois o cenario:
    tempo do log = t0 + t_s, com t0 = dias_hist dias. Avisos normais (taxas de HIP, rajadas
    de EMC) correm o tempo todo, em todos os encoders.

    Falha no cenario -> avisos acoplados no `rolo`:
      None, 'deriva_ganho', 'atrito'  nenhum (encoder digital nao avisa deriva de raio/escala;
                                      atrito e mecanico)
      'perda_pulsos'   incidentes de comunicacao (tipo_a: incremento de contador_erros + aviso_cabo;
                       tipo_b: aviso_sinal_b) com taxa TAXA_ACOPLADA_H x severidade(t)
      'ruido'          avisos de sinal (tipo_a: aviso_sinal/aviso_encoder; tipo_b: aviso_sinal_b), mesma taxa
      'erros_com'      degradacao 'cabo' de _degradacao (so tipo_a) de inicio_s ate o fim do
                       cenario, onde fica a falha final; a velocidade do cenario e normal
      'degradacao_sinal'  degradacao 'amplitude' de _degradacao, idem

    Devolve (eventos do log, contadores amostrados a cada 60 s, info).
    """
    encoders = ENCODERS_LINHA if encoders is None else encoders
    rng = np.random.default_rng([int(seed), 7])
    t_s = np.asarray(t_s, float)
    t0 = dias_hist * DIA
    t_fim = t0 + float(t_s[-1]) + (t_s[1] - t_s[0] if len(t_s) > 1 else 0.0)
    linhas = _normal(rng, 0.0, t_fim, encoders)
    drive = encoders.get(rolo)
    comp = list(encoders).index(rolo) + 1 if rolo in encoders else 0
    info = {"seed": seed, "falha": falha, "encoder": rolo, "t0_s": t0, "t_fim_hist_s": t0,
            "t_inicio_s": t0 + inicio_s, "t_fim_s": t_fim, "t_falha_s": None}
    if falha in ("perda_pulsos", "ruido"):
        sev = np.asarray(severidade, float)
        taxa = lambda t: TAXA_ACOPLADA_H * np.interp(np.asarray(t) - t0, t_s, sev)
        tempos = _pontos(rng, taxa, t0, t_fim)
        if falha == "perda_pulsos" and drive == "tipo_a":
            linhas += _incrementos_cabo(rng, tempos, drive, rolo, comp)
        else:
            linhas += _avisos_sinal(rng, tempos, drive, rolo)
    elif falha in CENARIOS_SO_AVISOS:
        cen = CENARIOS_SO_AVISOS[falha]
        if cen == "cabo" and drive != "tipo_a":
            raise ValueError("erros_com so existe em encoder tipo_a (contador contador_erros)")
        t_falha = t0 + float(t_s[-1])
        linhas += _degradacao(rng, cen, rolo, drive, comp, t0 + inicio_s, t_falha)
        info["t_falha_s"] = t_falha
    elif falha not in (None, "deriva_ganho", "atrito"):
        raise ValueError(f"falha desconhecida: {falha}")
    df = _tabela(linhas)
    df = df[df["t_s"] < t_fim].reset_index(drop=True)
    return eventos_do_log(df), contadores(df, 60.0, t_fim), info
