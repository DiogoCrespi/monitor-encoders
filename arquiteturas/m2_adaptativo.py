"""
Arquitetura candidata "Modelo 2 adaptativo a receita e material" (m2_adaptativo).

Problema que resolve (docs/MODELO-2-torque.md, limitacao 7): o Modelo 2 (regressao do torque por rolo
+ residuos estruturados + CUSUM) e muito sensivel (sigma do residuo de 1 min ~0,2-0,3 N.m no festoon e
no P&B1) e qualquer mudanca de EA (papel) ou de draw DEPOIS do treino desloca os tres torques por um
vetor de soma zero DENTRO do subespaco das assinaturas de encoder. Resultado: dezenas de falsos alarmes
"encoder" por EA -2 % ou draw +5 %.

Fisica usada (malha_fechada_v2.py, regime, sem escorregamento): T_j = T0 + EA * eps_j, com eps_j a
deformacao REAL do vao j. Com draw comandado D1 (P&B1) e D2 (P&B2) e erro relativo de leitura delta_k:
    eps_1 = D1 + delta_pb1 - delta_fest           eps_2 = D2 + delta_pb2 - delta_fest
    torque de tensao = r * (T_antes - T_depois) = r*EA*( -eps_1 , eps_1 - eps_2 , eps_2 ) (+ pontas)
Logo:
  * mudanca de EA por um fator (1+x)  -> desloca os torques por x * N, com N = r*EA*(-D1, D1-D2, D2):
    direcao CONHECIDA (o CLP da D1 e D2), so a amplitude e desconhecida;
  * mudanca de draw comandada         -> conhecida pelo CLP: entra como REGRESSOR (sem adaptacao);
  * deriva de ganho do encoder k      -> assinatura de encoder (soma zero), em geral NAO paralela a N.
NAO identificavel (fundamental): uma mudanca de draw NAO informada no P&B1 e a mesma coisa fisica que
uma deriva de ganho do encoder do P&B1 (eps_1 muda igual). Uma mudanca proporcional de D1 e D2 nao
informada e identica a uma mudanca de EA. Se D2 = D1 (vao 2 sem draw), N e paralelo a assinatura do
encoder do festoon, e EA vira indistinguivel do festoon. O setpoint de draw do CLP (colunas draw_pb1,
draw_pb2) resolve o primeiro caso; a velocidade lida pelos encoders (que a malha faz seguir a referencia)
da o MESMO draw comandado quando o CLP nao exporta o setpoint (fonte_draw='encoders') - ela nao contem
informacao sobre o erro do encoder, so sobre o comando.

Componentes (sobre MonitorTorqueV2, mesmos blocos de 60 s, portao, CUSUM k = 3, meta de FA):
  1. Regressor fisico de receita: n_k(t) = r*EA_nom*(-D1, D1-D2, D2)_k (draw do CLP ou dos encoders).
     Coeficiente FISICO (c_k = 1; `raio` pode ser {rolo: raio/relacao}), nao ajustado: com trocas de
     receita que multiplicam D1 e D2 pelo mesmo fator e EA diferente por material, um c_k livre se
     confunde com o EA da receita (no desenvolvimento deu c ~ 0,5 em vez de 1).
  2. Parametro de material theta (multiplicador de EA) por receita: estimado no treino (receita mais
     frequente = 1) e, na analise, REESTIMADO numa janela de quarentena de `holdout_min` minutos em regime
     depois de cada troca de receita (estatistica congelada nesse periodo). Sem tensao das pontas, a
     quarentena reestima tambem um offset do P&B2 (as pontas mudam com a receita e so aparecem no P&B2).
     A variacao de theta na reestimacao e limitada a +-delta_ea_max (receita conhecida) ou
     +-delta_ea_max_nova (receita nova).
  3. Rastreamento lento de theta dentro da receita (`tau_ea_h`), SO quando o residuo e consistente com
     uma mudanca de material: a componente do residuo ORTOGONAL a N (no espaco branqueado), filtrada
     com tau de `tau_porta_min`, tem que ficar abaixo de `porta_perp` (None = percentil 95 medido no
     treino). Uma deriva de encoder tem sempre
     componente ortogonal (o P&B1 e todo ortogonal; festoon e P&B2 ~metade no metrico euclidiano), entao
     o rastreamento para quando ela cresce; o que foi absorvido ate ali e so a parte da deriva ao longo
     de N. Limitado a +-delta_ea_rastreio em torno do valor ancorado da receita.
  4. Zona morta ao longo de N (`banda_ea`, fracao de EA): a componente do residuo na direcao N e
     encolhida por soft-threshold de banda_ea*|N|; dentro da banda (umidade/lote de papel), nenhum efeito;
     fora dela, conta normalmente. As outras direcoes nao sao tocadas. PADRAO 0 (desligada): com o
     detector de degrau (4b) ela nao reduz mais o FA e piora a isolacao de derivas lentas no dev.
  4b. Detector de DEGRAU de material (lote novo nao anunciado; `detectar_degrau`): CUSUM bilateral do
     desvio ao longo de N (em unidades de theta, escala sigma_d medida no treino; k_degrau, h_degrau).
     Quando dispara e a componente ortogonal media da janela e < razao_degrau x a componente ao longo
     de N (so material explica), os blocos desde o inicio do degrau ficam SUSPENSOS, abre-se uma
     quarentena de holdout_min, theta e reestimado e testa-se degrau x rampa: se o desvio ainda tem
     inclinacao > rampa_frac x o salto dentro da janela, e rampa (desgaste/deriva) -> theta NAO muda e
     os blocos suspensos voltam a contar; senao e degrau -> theta novo, blocos descartados.
  5. CUSUM bilateral nas 6 hipoteses (encoder/mecanica x 3 rolos), h por meta de FA (reamostragem em
     blocos de 2 h), isolacao confirmada por 2 de 3 alarmes em 90 min - igual ao Modelo 2 v2.1.

Interface: arquiteturas/base.py (treinar -> calibrar -> analisar). O argumento `eventos` (log de avisos)
nao e usado: as trocas de receita vem da coluna 'receita' (sem a coluna: receita unica).

ENTRADA (validada por `preparar_entrada`, em treinar, calibrar e analisar):
    t_s (s, crescente; amostragem de fs = 10 Hz - outra taxa da erro, ou e reamostrada com reamostrar=True),
    v_setpoint e v_<rolo> em m/min (lado do rolo, NAO rpm do motor), tq_<rolo> em N.m no lado do rolo
    (torque do motor x relacao), rolos 'festoon', 'pb1', 'pb2' (nomes fixos);
    opcionais: tensao_entrada/tensao_saida (N; sem elas, modo degradado sem tensao das pontas, decidido no
    treino), draw_pb1/draw_pb2 (fracao; sem elas, draw lido pelos encoders), receita (id; sem ela, receita
    unica). NaN em qualquer coluna usada: preenchido so para os filtros, e os blocos afetados (mais 30 s
    depois) ficam fora do regime. Lacuna em t_s > `lacuna_s` (2 s) = PARADA (setpoint 0 inserido): os filtros,
    os atrasos e o passado da CVA nao atravessam a lacuna; a CUSUM fica congelada, como em qualquer parada.

CAUSALIDADE (`causal`, padrao True): um degrau de material suspeito nunca suspende blocos ja emitidos; os
blocos da quarentena do degrau ficam RETIDOS ate a decisao (degrau -> descartados; rampa -> emitidos com
t_s = instante da decisao, o instante do bloco vai em 't_bloco_s'). Fim do arquivo com decisao pendente:
os blocos retidos nao sao emitidos. causal=False reproduz a decisao em lote antiga (so para comparacao).
"""
import warnings

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from monitor import SP_AQUECIMENTO, V_MIN_REGIME
from monitor.torque_v2 import COL_FORA, HIPOTESES, ROLOS, MonitorTorqueV2, _atraso, _ewma, matriz_assinaturas

try:                                                    # interface comum (pode estar ausente)
    from .base import Arquitetura as _Base
except Exception:                                       # pragma: no cover
    class _Base:
        nome = "base"

# Padroes = CONSTANTES DO SIMULADOR (poc/malha_fechada_v2.R e .EA). Na planta, passe `raio` ({rolo: raio do
# rolo / relacao}, m) e `ea_nominal` (N, E x espessura por metro de largura) da configuracao; um erro de
# escala em EA_nom x raio e absorvido no treino (theta0) quando o draw varia dentro da receita de referencia.
R_PADRAO = 0.15                                         # m, raio (poc/malha_fechada_v2.R)
EA_PADRAO = 2.0e6                                       # N, rigidez nominal do papel (malha_fechada_v2.EA)
FS_PADRAO = 10
COLS_TENSAO = ("tensao_entrada", "tensao_saida")
COLS_DRAW = ("draw_pb1", "draw_pb2")
COL_LACUNA = "lacuna"                                  # implantacao/importar_trace.py: 1 = amostra de lacuna
LACUNA_S = 2.0                                          # dt > LACUNA_S em t_s = parada
PARADA_MAX_S = 4 * 3600.0                               # lacunas maiores sao encurtadas (redutor ja frio)


def _dilatar(m, antes, depois):
    """m[i] True -> True em [i - antes, i + depois]."""
    if not m.any():
        return m
    c = np.concatenate([[0], np.cumsum(m.astype(np.int64))])
    n = len(m)
    i = np.arange(n)
    lo, hi = np.maximum(i - depois, 0), np.minimum(i + antes, n - 1)
    return (c[hi + 1] - c[lo]) > 0


def colunas_obrigatorias():
    return ["t_s", "v_setpoint"] + [f"{p}_{r}" for r in ROLOS for p in ("v", "tq")]


def _checar_colunas(df, usar_tensao):
    falt = [c for c in colunas_obrigatorias() if c not in df.columns]
    if usar_tensao:
        falt += [c for c in COLS_TENSAO if c not in df.columns]
    if falt:
        achadas = sorted(c for c in df.columns if str(c).startswith(("v_", "tq_", "tensao")))
        raise ValueError(
            f"colunas ausentes no arquivo: {falt}. O modelo espera t_s (s), v_setpoint e v_<rolo> (m/min), "
            f"tq_<rolo> (N.m, lado do rolo) com rolos {list(ROLOS)}"
            + (" e tensao_entrada/tensao_saida (N; o modelo foi treinado COM tensao das pontas - para arquivos "
               "sem elas, treine sem as colunas: modo degradado)" if usar_tensao else "")
            + f". Colunas parecidas encontradas: {achadas}. Renomeie os rolos/colunas antes de analisar.")


def _reamostrar(df, fs, lacuna_s):
    """Grade uniforme de fs Hz (cada ponto k representa [t_k, t_k + 1/fs)). Fonte MAIS RAPIDA que fs: media
    das amostras do intervalo (sem aliasing); fonte mais lenta: interpolacao linear. receita/draw e colunas
    nao numericas: ultimo valor. Intervalos sem amostra dentro de lacunas > lacuna_s ficam de fora (viram
    parada no passo seguinte)."""
    t = df["t_s"].to_numpy(float)
    dt_src = float(np.median(np.diff(t)))
    k = np.floor((t - t[0]) * fs + 1e-6).astype(np.int64)
    nk = int(k[-1]) + 1
    g = t[0] + np.arange(nk) / fs
    j = np.clip(np.searchsorted(t, g + 1e-9, side="right") - 1, 0, len(t) - 1)
    jn = np.minimum(j + 1, len(t) - 1)
    rapida = dt_src < 1.0 / fs
    cont = np.bincount(k, minlength=nk)
    if rapida:
        manter = cont > 0
    else:
        manter = ~(((t[jn] - t[j]) > lacuna_s) & (np.abs(g - t[j]) > 1e-9))
    out = {"t_s": g[manter]}
    for c in df.columns:
        if c == "t_s":
            continue
        x = df[c].to_numpy()
        if c in ("receita", COL_LACUNA) + COLS_DRAW or not np.issubdtype(x.dtype, np.number) or x.dtype == bool:
            out[c] = x[j][manter]
        elif rapida:
            xf = x.astype(float)
            fin = np.isfinite(xf)
            sm = np.bincount(k[fin], weights=xf[fin], minlength=nk)
            nn = np.bincount(k[fin], minlength=nk)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[c] = np.where(nn > 0, sm / np.maximum(nn, 1), np.nan)[manter]
        else:
            out[c] = np.interp(g, t, x.astype(float))[manter]
    return pd.DataFrame(out), rapida


def preparar_entrada(df, fs=FS_PADRAO, usar_tensao=False, reamostrar=False, lacuna_s=LACUNA_S, janela_s=30.0,
                     avisar=True):
    """Valida e prepara um arquivo para o modelo (sem mudar o arquivo do chamador). Devolve (df, info).

    1. colunas obrigatorias (nomes dos rolos) e, se `usar_tensao`, tensao das pontas -> ValueError claro;
       draw_pb1/draw_pb2 ou receita sem nenhum valor finito -> tratadas como ausentes (aviso).
    2. t_s: linhas sem t_s finito saem; t_s repetido sai; t_s decrescente -> ValueError.
    3. amostragem: mediana de dt diferente de 1/fs em mais de 2 % -> ValueError (ou reamostra se
       `reamostrar=True`).
    4. lacunas: dt > 1,5/fs -> amostras inseridas; dt <= lacuna_s: NaN (falta curta); dt > lacuna_s:
       PARADA (setpoint 0; lacunas > PARADA_MAX_S encurtadas para PARADA_MAX_S).
    5. NaN (inclusive os inseridos): ffill/bfill so para os filtros; a amostra e marcada fora do regime
       (coluna COL_FORA, lida por MonitorTorqueV2.portao), dilatada 1 s antes e janela_s depois
       (filtros, atrasos de aceleracao, np.gradient). Coluna 'lacuna' = 1 (implantacao/importar_trace.py:
       lacuna ate 15 min com sinais congelados) recebe o mesmo tratamento de parada (fora do regime)
       - consistente com o importador; sem a coluna, as lacunas vem de t_s."""
    info = {"reamostrado": False, "n_lacunas": 0, "s_lacunas": 0.0, "n_paradas_inseridas": 0,
            "amostras_nan": 0, "frac_fora": 0.0, "avisos": []}
    _checar_colunas(df, usar_tensao)
    orig = df
    msgs = info["avisos"]
    soltar = []
    for grupo, nome in ((COLS_DRAW, "draw do CLP"), (("receita",), "receita")):
        pres = [c for c in grupo if c in df.columns]
        if not pres:
            msgs.append("sem draw_pb1/draw_pb2: draw lido pelos encoders" if grupo == COLS_DRAW else
                        "sem coluna 'receita': receita unica (trocas de receita nao sao vistas como tal)")
            continue
        if not len(df):
            continue
        vazio = any(df[c].isna().all() for c in pres) or len(pres) < len(grupo)
        if vazio:
            soltar += pres
            msgs.append(f"{nome} ({pres}) sem valores validos: tratado como ausente")
    if soltar:
        df = df.drop(columns=soltar)
    if usar_tensao:
        for c in COLS_TENSAO:
            if len(df) and not np.isfinite(df[c].to_numpy(float)).any():
                raise ValueError(f"coluna {c} sem nenhum valor finito, mas o modelo foi treinado com a tensao "
                                 "das pontas: treine sem as colunas (modo degradado) ou corrija o arquivo")
    t = df["t_s"].to_numpy(float)
    ok_t = np.isfinite(t)
    if not ok_t.all():
        msgs.append(f"{int((~ok_t).sum())} linhas sem t_s finito removidas")
        df = df[ok_t]
        t = t[ok_t]
    if len(t) > 1:
        dt = np.diff(t)
        if (dt < 0).any():
            raise ValueError(f"t_s nao e crescente (1a volta em t_s = {t[1:][dt < 0][0]:.3f}): ordene o arquivo")
        if (dt == 0).any():
            keep = np.r_[True, dt > 0]
            msgs.append(f"{int((~keep).sum())} linhas com t_s repetido removidas")
            df, t = df[keep], t[keep]
            dt = np.diff(t)
        if len(dt):
            dtm = float(np.median(dt))
            if abs(dtm * fs - 1) > 0.02:
                if not reamostrar:
                    raise ValueError(
                        f"amostragem de {1 / dtm:.4g} Hz em t_s (mediana de dt = {dtm:.4g} s); o modelo espera "
                        f"{fs} Hz (blocos de 60 s = {60 * fs} amostras, filtros e limites em amostras). Reamostre "
                        f"numa grade uniforme de {fs} Hz (ex.: implantacao/importar_trace.reamostrar) ou passe "
                        "reamostrar=True.")
                df, rapida = _reamostrar(df.reset_index(drop=True), fs, lacuna_s)
                t = df["t_s"].to_numpy(float)
                dt = np.diff(t)
                info["reamostrado"] = True
                msgs.append(f"reamostrado de {1 / dtm:.4g} Hz para {fs} Hz ("
                            + ("media por intervalo)" if rapida else
                               "interpolacao: fonte MAIS LENTA que o treino - a aspereza da CVA (RMS da diferenca a "
                               "10 Hz) nao e comparavel ao treino; espere alarmes falsos da CVA)"))
    if not (isinstance(df.index, pd.RangeIndex) and df.index.start == 0 and df.index.step == 1):
        df = df.reset_index(drop=True)
    n = len(df)
    inserido = np.zeros(n, bool)
    parada = np.zeros(n, bool)
    if n > 1:
        dt = np.diff(t)
        gaps = np.flatnonzero(dt > 1.5 / fs)
        if len(gaps):
            dur = dt[gaps]
            longa = dur > lacuna_s
            n_ins = np.minimum(np.round(dur * fs).astype(np.int64) - 1, int(PARADA_MAX_S * fs))
            info.update(n_lacunas=int(len(gaps)), s_lacunas=float((dur - 1 / fs).sum()),
                        n_paradas_inseridas=int(longa.sum()))
            desl = np.zeros(n, np.int64)
            desl[gaps + 1] = n_ins
            pos = np.arange(n) + np.cumsum(desl)
            m = n + int(n_ins.sum())
            novo = {}
            for c in df.columns:
                x = df[c].to_numpy()
                if np.issubdtype(x.dtype, np.number) and x.dtype != bool:
                    y = np.full(m, np.nan)
                else:
                    y = np.full(m, None, dtype=object)
                y[pos] = x
                novo[c] = y
            inserido = np.ones(m, bool)
            inserido[pos] = False
            parada = np.zeros(m, bool)
            tt = novo["t_s"]
            for g_, k, lg in zip(gaps, n_ins, longa):
                a = pos[g_] + 1
                tt[a:a + k] = np.linspace(t[g_], t[g_ + 1], k + 2)[1:-1]
                if lg:
                    parada[a:a + k] = True
            novo["v_setpoint"][parada] = 0.0
            df = pd.DataFrame(novo)
            msgs.append(f"{len(gaps)} lacunas em t_s ({info['s_lacunas']:.0f} s; {int(longa.sum())} > {lacuna_s:g} s "
                        "tratadas como parada)")
            n = m
    usadas = [c for c in colunas_obrigatorias()[1:] + (list(COLS_TENSAO) if usar_tensao else [])
              + [c for c in COLS_DRAW if c in df.columns]]
    bad = inserido & ~parada
    if n:
        X = df[usadas].to_numpy(float)
        bad = bad | ~np.isfinite(X).all(axis=1)
        if "receita" in df.columns:
            bad |= df["receita"].isna().to_numpy()
        if COL_LACUNA in df.columns:                            # lacuna marcada pelo importador da planta
            lac = pd.to_numeric(df[COL_LACUNA], errors="coerce").fillna(0).to_numpy(float) > 0.5
            if lac.any():
                parada = parada | lac
                msgs.append(f"{int(lac.sum())} amostras com {COL_LACUNA}=1 (importador): fora do regime")
    fora_ant = df[COL_FORA].to_numpy(bool) if COL_FORA in df.columns else np.zeros(n, bool)
    if bad.any() or parada.any():
        info["amostras_nan"] = int((bad & ~inserido).sum())
        if df is orig:
            df = df.copy()
        cols_f = usadas + (["receita"] if "receita" in df.columns else [])
        df[cols_f] = df[cols_f].ffill().bfill()
        if info["amostras_nan"]:
            msgs.append(f"{info['amostras_nan']} amostras com NaN: preenchidas so para os filtros, blocos fora do regime")
        fora = _dilatar(bad | parada, int(round(1.0 * fs)), int(round(janela_s * fs))) | fora_ant
        df[COL_FORA] = fora
        info["frac_fora"] = float(fora.mean())
    if n and len(df) >= 60 * fs and np.nanmax(df["v_setpoint"].to_numpy(float)) < V_MIN_REGIME:
        msgs.append(f"v_setpoint nunca chega a {V_MIN_REGIME:.0f} (m/min esperado): sem regime neste arquivo "
                    "- conferir a unidade se a linha estava produzindo")
    if avisar and msgs:
        warnings.warn("entrada: " + "; ".join(msgs), stacklevel=3)
    return df, info


class _MonitorBase(MonitorTorqueV2):
    """MonitorTorqueV2 com estado inicial do aquecimento do redutor por arquivo ('auto': quente se o
    arquivo comeca com a linha rodando), como em arquiteturas/cva_m2.py."""

    def __init__(self, *a, estado_inicial="auto", **kw):
        super().__init__(*a, **kw)
        self.estado_inicial = estado_inicial

    def _x(self, df, rolo):
        sp = df["v_setpoint"].to_numpy(float)
        v = df[f"v_{rolo}"].to_numpy(float) / 60.0
        a = np.gradient(v) * self.fs
        rod = (sp > SP_AQUECIMENTO).astype(float)
        quente = self.estado_inicial == "quente" or (self.estado_inicial == "auto" and len(sp) > 0
                                                     and sp[0] > SP_AQUECIMENTO)
        cols = [sp / 600, v, v ** 2, a] + [_atraso(a, i) for i in range(1, self.lags + 1)]
        for tau_min in (15, 45):
            lam = 1 - np.exp(-1 / (self.fs * 60 * tau_min))
            cols.append(lfilter([lam], [1, -(1 - lam)], rod, zi=[1 - lam])[0] if quente else _ewma(rod, lam))
        if self.usar_tensao:
            for c in ("tensao_entrada", "tensao_saida"):
                x = df[c].to_numpy(float) / 1000
                cols += [x, _ewma(x, 1 - np.exp(-1 / (self.fs * 2.0)))]
        return np.column_stack(cols + [np.ones(len(df))])


class ArquiteturaM2Adaptativo(_Base):
    nome = "m2_adaptativo"

    def __init__(self, k=3.0, media_s=60, bloco_h=2.0, holdoff_min=10.0, usar_tensao_pontas="auto",
                 raio=R_PADRAO, ea_nominal=EA_PADRAO, usar_draw=True, fonte_draw="auto",
                 reestimar_receita=True, holdout_min=20.0, delta_ea_max=0.15, delta_ea_max_nova=0.30,
                 tau_ea_h=12.0, porta_perp=None, tau_porta_min=30.0, delta_ea_rastreio=0.10,
                 banda_ea=0.0, detectar_degrau=True, k_degrau=1.0, h_degrau=6.0, razao_degrau=0.35,
                 rampa_frac=0.5, estado_inicial="auto", semanas_boot=40, seed_boot=0, nome=None,
                 causal=True, receita_ausente="unica", estimar_theta0=True, var_min_theta0=0.05,
                 reamostrar=False, lacuna_s=LACUNA_S):
        """raio: m ({rolo: raio/relacao} ou escalar) e ea_nominal: N - CONFIGURACAO da planta (padroes =
        constantes do simulador). usar_tensao_pontas: 'auto' = usa se todos os arquivos de treino tem
        tensao_entrada/tensao_saida (senao modo degradado, com aviso). causal: ver o docstring do modulo.
        receita_ausente: sem coluna 'receita', 'unica' (padrao) ou 'draw' (nova receita a cada degrau de
        draw > 3 %, heuristica antiga). estimar_theta0: estima no treino o fator de escala de EA_nom x raio
        (theta0) quando o draw varia > var_min_theta0 dentro da receita de referencia; senao theta0 = 1."""
        self.k, self.media_s, self.bloco_h, self.holdoff_min = k, int(media_s), bloco_h, holdoff_min
        if usar_tensao_pontas not in (True, False, "auto"):
            raise ValueError("usar_tensao_pontas: True, False ou 'auto'")
        if receita_ausente not in ("unica", "draw"):
            raise ValueError("receita_ausente: 'unica' ou 'draw'")
        self.usar_tensao_cfg = usar_tensao_pontas
        self.usar_tensao = usar_tensao_pontas is True
        self.raio, self.ea_nominal = raio, ea_nominal
        self.causal, self.receita_ausente = bool(causal), receita_ausente
        self.estimar_theta0, self.var_min_theta0 = estimar_theta0, var_min_theta0
        self.reamostrar, self.lacuna_s = reamostrar, lacuna_s
        self.theta0, self.theta0_identificavel = 1.0, False
        self.usar_draw, self.fonte_draw = usar_draw, fonte_draw
        self.reestimar, self.holdout_min = reestimar_receita, holdout_min
        self.delta_ea_max, self.delta_ea_max_nova = delta_ea_max, delta_ea_max_nova
        self.tau_ea_h, self.porta_perp, self.tau_porta_min = tau_ea_h, porta_perp, tau_porta_min
        self.delta_ea_rastreio, self.banda_ea = delta_ea_rastreio, banda_ea
        self.detectar_degrau, self.k_degrau, self.h_degrau = detectar_degrau, k_degrau, h_degrau
        self.razao_degrau, self.rampa_frac = razao_degrau, rampa_frac
        self.estado_inicial, self.semanas_boot, self.seed_boot = estado_inicial, semanas_boot, seed_boot
        if nome:
            self.nome = nome
        self.m = _MonitorBase(k=k, media_s=media_s, ordem_ar=0, bloco_h=bloco_h, holdoff_min=holdoff_min,
                              usar_tensao_pontas=self.usar_tensao, estado_inicial=estado_inicial,
                              semanas_boot=semanas_boot, seed_boot=seed_boot)
        self.h = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("ultimo_log", None)                               # modelos antigos: log do ultimo arquivo
        return d

    def _preparar(self, df, avisar=True):
        """preparar_entrada com a escolha de tensao do treino (ver o docstring do modulo)."""
        return preparar_entrada(df, self.m.fs, self.usar_tensao, getattr(self, "reamostrar", False),
                                getattr(self, "lacuna_s", LACUNA_S), self.m.janela_s, avisar)[0]

    # ================================================================ sinais de receita
    def _draw(self, df, fonte=None):
        """(D1, D2) a 10 Hz: setpoint do CLP (draw_pb1/draw_pb2) ou deformacao comandada lida pelos
        encoders (a malha faz a leitura seguir a referencia: (v_pb - v_fest)/v_fest = draw comandado).
        `fonte` sobrepoe self.fonte_draw nesta chamada (sem mudar o objeto). Sem as colunas do CLP, 'clp'
        cai para 'encoders'."""
        fonte = fonte or self.fonte_draw
        tem_clp = set(COLS_DRAW) <= set(df.columns)
        if fonte == "auto":
            fonte = "clp" if tem_clp else "encoders"
        if fonte == "clp" and tem_clp:
            return df["draw_pb1"].to_numpy(float), df["draw_pb2"].to_numpy(float), fonte
        fonte = "encoders"
        vf = df["v_festoon"].to_numpy(float)
        ok = vf > 100.0
        out = []
        jan = 60 * self.m.fs
        for r in ("pb1", "pb2"):
            e = pd.Series(np.where(ok, (df[f"v_{r}"].to_numpy(float) - vf) / np.maximum(vf, 1.0), np.nan))
            m = e.rolling(jan, min_periods=jan // 2).mean()
            # media movel de 60 s (600 amostras a 10 Hz), sem nan
            out.append(m.ffill().bfill().fillna(0.0).to_numpy())
        return out[0], out[1], fonte

    def _receita(self, df, D1, D2):
        if "receita" in df.columns:
            return df["receita"].to_numpy()
        if self.receita_ausente == "unica":                     # sem id: receita unica (a de referencia)
            ref = getattr(self, "receita_ref", 0)
            return np.full(len(df), ref)
        # 'draw': nova receita quando o draw muda mais de 3 % (degrau sustentado)
        rid = np.zeros(len(df), int)
        ref, atual = (D1[0], D2[0]), 0
        passo = self.m.fs * 60
        for i in range(0, len(df), passo):
            d1, d2 = D1[i], D2[i]
            if abs(d1 - ref[0]) > 0.03 * max(abs(ref[0]), 1e-9) or abs(d2 - ref[1]) > 0.03 * max(abs(ref[1]), 1e-9):
                atual, ref = atual + 1, (d1, d2)
            rid[i:i + passo] = atual
        return rid

    def _fisico(self, df):
        """n_k a 10 Hz (N.m por unidade de theta): r*EA_nom*(-D1, D1-D2, D2)."""
        D1, D2, fonte = self._draw(df)
        if isinstance(self.raio, dict):
            desc = set(self.raio) - set(ROLOS)
            if desc:
                raise ValueError(f"raio: rolos desconhecidos {sorted(desc)} (esperado {list(ROLOS)})")
            r_ef = np.array([self.raio.get(r, R_PADRAO) for r in ROLOS], float)
        else:
            r_ef = np.full(3, float(self.raio))
        n = np.column_stack([-D1, D1 - D2, D2]) * (r_ef * self.ea_nominal)[None, :]
        if not self.usar_draw:
            n = n * 0.0
        return n, self._receita(df, D1, D2)

    # ================================================================ blocos
    def _blocos(self, df):
        """Por bloco de 60 s com >= 80 % em regime: R0 (residuo SEM o termo de receita), N (c_k*n_k),
        indice 10 Hz do fim do bloco, receita do bloco, inicio_de_trecho."""
        m = self.m
        nb = self.media_s * m.fs
        g = m.portao(df)
        nfis, rid = self._fisico(df)
        r0 = np.column_stack([df[f"tq_{r}"].to_numpy(float) - m._x(df, r) @ self.coef[r] for r in ROLOS])
        nn = nfis * self.c[None, :]
        n = len(df) // nb
        gg = g[: n * nb].reshape(n, nb)
        cont = np.maximum(gg.sum(axis=1), 1)[:, None]
        R0 = (r0[: n * nb] * g[: n * nb, None]).reshape(n, nb, 3).sum(axis=1) / cont
        N = (nn[: n * nb] * g[: n * nb, None]).reshape(n, nb, 3).sum(axis=1) / cont
        sel = np.flatnonzero(gg.sum(axis=1) >= 0.8 * nb)
        rec = rid[np.minimum((sel + 1) * nb - 1, len(df) - 1)]
        return R0[sel], N[sel], (sel + 1) * nb - 1, rec

    # ================================================================ treino
    def treinar(self, normais, eventos_normais=None, duracao_eventos_s=None, alpha=1e-3):
        """MQ CONJUNTO dos tres rolos: regressores de cada rolo (MonitorTorqueV2._x) + termo de receita
        com coeficiente fisico (n_k) + (theta_r - 1)*n_k por receita r != referencia (theta compartilhado
        pelos tres rolos) [+ offset do P&B2 por receita, sem tensao das pontas]. Duas passadas: a 2a pesa
        cada rolo por 1/sigma_k^2 (o P&B2 carrega o passeio das pontas)."""
        dfs = [normais] if isinstance(normais, pd.DataFrame) else list(normais)
        m = self.m
        tem = all(set(COLS_TENSAO) <= set(d.columns) and all(np.isfinite(d[c].to_numpy(float)).any()
                                                              for c in COLS_TENSAO) for d in dfs)
        if self.usar_tensao_cfg == "auto":
            usar = tem
            if not tem:
                warnings.warn("treino sem tensao_entrada/tensao_saida em todos os arquivos: MODO DEGRADADO "
                              "(usar_tensao_pontas=False; a quarentena reestima tambem um offset do P&B2)",
                              stacklevel=2)
        else:
            usar = bool(self.usar_tensao_cfg)
        self.usar_tensao = m.usar_tensao = usar
        dfs = [self._preparar(d) for d in dfs]
        fis = [self._fisico(d) for d in dfs]
        gates = [m.portao(d) for d in dfs]
        rk = np.concatenate([f[1][g] for f, g in zip(fis, gates)])
        vals, cnt = np.unique(rk, return_counts=True)
        vals = [v.item() for v in vals]
        self.receita_ref = vals[int(cnt.argmax())]
        outras = [v for v in vals if v != self.receita_ref and (rk == v).sum() > 600 * m.fs]
        self.c = np.ones(3)                                     # coeficiente fisico (ver docstring)
        nk = np.vstack([f[0][g] for f, g in zip(fis, gates)])
        Xs = [np.vstack([m._x(d, r)[g] for d, g in zip(dfs, gates)]) for r in ROLOS]
        ys = [np.concatenate([d[f"tq_{r}"].to_numpy(float)[g] for d, g in zip(dfs, gates)]) - nk[:, j]
              for j, r in enumerate(ROLOS)]
        p = Xs[0].shape[1]
        # theta0 (escala de EA_nom x raio da configuracao): identificavel so com o draw variando DENTRO da
        # receita de referencia (com draw constante o intercepto absorve theta0 * N)
        ref = rk == self.receita_ref
        nref = nk[ref]
        var0 = (float(np.sqrt(((nref - nref.mean(0)) ** 2).sum(1).mean()) / max(np.linalg.norm(nref.mean(0)), 1e-12))
                if len(nref) else 0.0)
        self.theta0_var = var0
        self.theta0_identificavel = bool(self.estimar_theta0 and self.usar_draw and var0 > self.var_min_theta0)
        if self.theta0_identificavel:
            outras = [self.receita_ref] + outras                 # coluna (theta_ref - 1) * n_k na referencia
        nb_ = 1 if self.usar_tensao else 2                      # colunas por receita: theta (+ beta)
        P = 3 * p + nb_ * len(outras)
        OH = np.column_stack([(rk == v).astype(float) for v in outras]) if outras else np.zeros((len(rk), 0))
        if self.theta0_identificavel and nb_ == 2:
            OH_b = OH.copy()
            OH_b[:, 0] = 0.0                                    # offset do P&B2 da referencia = intercepto
        else:
            OH_b = OH
        w = np.ones(3)
        for passada in range(2):
            A, y = np.zeros((P, P)), np.zeros(P)
            for j in range(3):
                for s0 in range(0, len(rk), 200_000):            # em pedacos (memoria)
                    sl = slice(s0, s0 + 200_000)
                    G = np.zeros((len(rk[sl]), P))
                    G[:, j * p:(j + 1) * p] = Xs[j][sl]
                    G[:, 3 * p:3 * p + len(outras)] = OH[sl] * nk[sl, [j]]
                    if nb_ == 2 and j == 2:
                        G[:, 3 * p + len(outras):] = OH_b[sl]
                    A += w[j] * G.T @ G
                    y += w[j] * G.T @ ys[j][sl]
            b = np.linalg.solve(A + alpha * np.eye(P), y)
            self.coef = {r: b[j * p:(j + 1) * p] for j, r in enumerate(ROLOS)}
            res = [ys[j] - Xs[j] @ self.coef[ROLOS[j]] - (OH * nk[:, [j]]) @ b[3 * p:3 * p + len(outras)]
                   - ((OH_b @ b[3 * p + len(outras):]) if (nb_ == 2 and j == 2) else 0.0) for j in range(3)]
            w = 1.0 / np.array([r.var() for r in res])
        self.theta_rec = {v: 1.0 for v in vals}
        self.beta_rec = {v: 0.0 for v in vals}
        for i_r, v in enumerate(outras):
            self.theta_rec[v] = 1.0 + float(b[3 * p + i_r])
            if nb_ == 2:
                self.beta_rec[v] = float(b[3 * p + len(outras) + i_r])
        self.theta0 = 1.0
        if self.theta0_identificavel:
            # escala absorvida em c (N passa a ser r*EA_efetivo*(...)); theta das receitas fica relativo a
            # referencia (= 1) e os limites de theta valem em fracao do EA REAL do treino
            th0 = float(np.clip(self.theta_rec[self.receita_ref], 0.5, 2.0))
            self.theta0 = th0
            self.c = np.full(3, th0)
            self.theta_rec = {v: t_ / th0 for v, t_ in self.theta_rec.items()}
            if abs(th0 - 1) > 0.2:
                warnings.warn(f"theta0 = {th0:.3f}: EA_nom x raio da configuracao difere {100 * (th0 - 1):+.0f} % do "
                              "estimado no treino (absorvido; conferir ea_nominal/raio)", stacklevel=2)
        del Xs, ys, nk, res
        # estatistica dos residuos de 1 min (theta/beta da receita aplicados)
        self.mu, self.sigma, self.S = np.zeros(3), np.ones(3), np.eye(3)
        partes = [self._blocos(d) for d in dfs]
        R0 = np.vstack([q[0] for q in partes])
        N = np.vstack([q[1] for q in partes])
        rec = np.concatenate([q[3] for q in partes])
        th_b = pd.Series(rec).map(self.theta_rec).fillna(1.0).to_numpy(float)
        be_b = pd.Series(rec).map(self.beta_rec).fillna(0.0).to_numpy(float)
        R = R0 - th_b[:, None] * N
        R[:, 2] -= be_b
        self.mu = R.mean(axis=0)
        self.sigma = R.std(axis=0)
        self.S = np.corrcoef(((R - self.mu) / self.sigma).T)
        self._montar()
        # escalas do detector de degrau e da porta do rastreamento, medidas no proprio treino
        self.sigma_d, det = None, self.detectar_degrau
        self.detectar_degrau = False
        ds, qs = [], []
        for q_ in partes:
            lg = self._serie(None, _blocos=q_)[3]
            ds.append(lg["d"])
            qs.append(lg["q"])
        self.detectar_degrau = det
        self.sigma_d = float(np.nanstd(np.concatenate(ds)))
        self.porta_perp_treino = float(np.nanpercentile(np.concatenate(qs), 95))
        return self

    def _montar(self):
        Si = np.linalg.inv(self.S)
        self.Ct = np.linalg.cholesky(Si).T                     # u = Ct e  ->  u'u = e' S^-1 e
        H = matriz_assinaturas() / self.sigma[:, None]
        St = self.Ct @ H                                        # assinaturas no espaco branqueado
        self.Wu = (St / np.linalg.norm(St, axis=0)).T           # 6 x 3: z = Wu u

    # ================================================================ serie de z (sequencial, sem h)
    def _serie(self, df, diag=False, _blocos=None):
        """z por bloco (sequencial; nao depende de h). Devolve (Z emitidos, indice 10 Hz, residuo, log)."""
        R0, N, idx, rec = self._blocos(df) if _blocos is None else _blocos
        nb = len(R0)
        Z = np.zeros((nb, len(HIPOTESES)))
        emit = np.zeros(nb, bool)
        dev = np.zeros((nb, 3))
        log = {"theta": np.full(nb, np.nan), "beta": np.full(nb, np.nan), "trocas": [], "degraus": [],
               "q": np.full(nb, np.nan), "d": np.full(nb, np.nan), "uperp": np.full((nb, 3), np.nan),
               "a": np.full(nb, np.nan), "quar": np.zeros(nb, bool)}
        H = max(1, int(round(self.holdout_min * 60 / self.media_s)))
        lam_g = self.media_s / (self.tau_porta_min * 60)
        lam_t = self.media_s / (self.tau_ea_h * 3600) if self.tau_ea_h else 0.0
        porta = self.porta_perp if self.porta_perp is not None else getattr(self, "porta_perp_treino", np.inf)
        sd = getattr(self, "sigma_d", None)
        detectar = bool(getattr(self, "detectar_degrau", False) and sd)
        theta_r, beta_r = dict(self.theta_rec), dict(self.beta_rec)
        e3 = np.array([0.0, 0.0, 1.0])
        rec_ant, quar, buf, modo = None, 0, [], None
        theta, beta, ancora, q, th_ant = 1.0, 0.0, 1.0, np.zeros(3), 1.0
        gp = gm = 0.0
        ini_p = ini_m = 0
        pend = []                                               # blocos suspensos por um degrau suspeito
        causal = getattr(self, "causal", False)
        dec = np.arange(nb)                     # bloco em que a emissao do bloco b foi decidida (causal)
        for b in range(nb):
            rb = rec[b].item() if hasattr(rec[b], "item") else rec[b]
            if rb != rec_ant:
                conhecida = rb in theta_r
                theta = theta_r.get(rb, theta if rec_ant is not None else 1.0)
                beta = beta_r.get(rb, beta if rec_ant is not None else 0.0)
                ancora = theta
                if pend:                                        # degrau interrompido pela troca: reemite
                    emit[pend] = True
                    if causal:
                        dec[pend] = b
                    pend = []
                modo, quar = None, 0
                if self.reestimar and (rec_ant is not None or not conhecida):
                    quar, buf, modo = H, [], "receita"
                    log["trocas"].append((int(idx[b]), rb, conhecida))
                gp = gm = 0.0
                ini_p = ini_m = b
                rec_ant = rb
            if not (np.isfinite(R0[b]).all() and np.isfinite(N[b]).all()):
                log["theta"][b], log["beta"][b] = theta, beta  # bloco invalido: fica fora, estado intacto
                continue
            r = R0[b] - theta * N[b] - self.mu
            r[2] -= beta
            if modo == "receita":                               # quarentena: estatistica congelada
                log["quar"][b] = True
                buf.append((R0[b], N[b]))
                quar -= 1
                if quar == 0:
                    theta, beta = self._reestimar(buf, theta, beta, rb in self.theta_rec, e3)
                    theta_r[rb], beta_r[rb] = theta, beta
                    ancora, q, modo = theta, np.zeros(3), None
                    gp = gm = 0.0
                    ini_p = ini_m = b + 1
                log["theta"][b], log["beta"][b] = theta, beta
                continue
            e = r / self.sigma
            u = self.Ct @ e
            mv = self.Ct @ (N[b] / self.sigma)
            nm = np.linalg.norm(mv)
            if nm > 1e-9:
                mh = mv / nm
                a = float(u @ mh)
                uperp = u - a * mh
                d = a / nm                                      # desvio ao longo de N, em unidades de theta
                q = (1 - lam_g) * q + lam_g * uperp
                log["q"][b], log["d"][b], log["uperp"][b], log["a"][b] = np.linalg.norm(q), d, uperp, a
                if lam_t and modo is None and np.isfinite(d) and np.linalg.norm(q) < porta:
                    theta = float(np.clip(theta + lam_t * d, ancora - self.delta_ea_rastreio,
                                          ancora + self.delta_ea_rastreio))
                band = self.banda_ea * nm
                a2 = np.sign(a) * max(abs(a) - band, 0.0)
                u = uperp + a2 * mh
                # ---- detector de DEGRAU de material (lote novo nao anunciado): opcao 5
                if detectar and modo is None:
                    gp, gm = max(0.0, gp + d / sd - self.k_degrau), max(0.0, gm - d / sd - self.k_degrau)
                    if gp == 0.0:
                        ini_p = b + 1
                    if gm == 0.0:
                        ini_m = b + 1
                    if max(gp, gm) > self.h_degrau:
                        b0 = ini_p if gp > gm else ini_m
                        jan = [x for x in range(b0, b + 1) if not np.isnan(log["a"][x])]
                        am = float(np.mean(log["a"][jan]))
                        pm = float(np.linalg.norm(np.mean(log["uperp"][jan], axis=0)))
                        if pm < self.razao_degrau * abs(am):    # so ao longo de N: consistente com material
                            # causal: blocos ja emitidos nao sao suspensos (sem retratacao); so a
                            # quarentena a partir daqui fica retida ate a decisao
                            pend = [] if causal else [x for x in jan if x < b and emit[x]]
                            emit[pend] = False
                            quar, modo, th_ant = H, "degrau", theta
                            buf = [(R0[x], N[x]) for x in jan if x < b]
                        gp = gm = 0.0
                        ini_p = ini_m = b + 1
            Z[b] = self.Wu @ u
            dev[b] = r
            log["theta"][b], log["beta"][b] = theta, beta
            if modo == "degrau":                                # suspenso ate decidir degrau x rampa
                pend.append(b)
                buf.append((R0[b], N[b]))
                quar -= 1
                if quar == 0:
                    th_novo, _ = self._reestimar(buf, th_ant, beta, True, e3)
                    th_novo = float(np.clip(th_novo, ancora - self.delta_ea_rastreio,
                                            ancora + self.delta_ea_rastreio))
                    # rampa x degrau: variacao do desvio ao longo da janela, ja com o theta novo
                    dd = np.array([self._d_bloco(x[0], x[1], th_novo, beta) for x in buf])
                    tt = np.arange(len(dd)) - (len(dd) - 1) / 2
                    incl = float(tt @ dd / (tt @ tt)) * len(dd)
                    rampa = abs(incl) > self.rampa_frac * abs(th_novo - th_ant)
                    log["degraus"].append((int(idx[b]), th_ant, th_novo, incl, bool(rampa)))
                    if rampa:                                   # parece desgaste/deriva: nao absorve
                        emit[pend] = True
                        if causal:
                            dec[pend] = b                       # emitidos no instante da decisao
                    else:
                        theta = th_novo
                        theta_r[rb] = theta
                    pend, modo, q = [], None, np.zeros(3)
                continue
            emit[b] = True
        log["pendentes_fim"] = len(pend)
        if pend and not causal:                                 # lote: fim do arquivo com degrau pendente
            emit[pend] = True                                   # causal: decisao desconhecida -> nao emite
        log["emit"], log["dec"] = emit, dec
        log["idx_dec"] = idx[dec] if nb else np.zeros(0, int)
        return Z[emit], idx[emit], dev[emit], log

    def _d_bloco(self, R0b, Nb, theta, beta):
        r = R0b - theta * Nb - self.mu
        r[2] -= beta
        mv = self.Ct @ (Nb / self.sigma)
        nm = np.linalg.norm(mv)
        return float((self.Ct @ (r / self.sigma)) @ mv / nm ** 2) if nm > 1e-9 else 0.0

    def _reestimar(self, buf, theta, beta, conhecida, e3):
        R0 = np.array([x[0] for x in buf]) - self.mu
        N = np.array([x[1] for x in buf])
        Wm = np.diag(1 / self.sigma) @ np.linalg.inv(self.S) @ np.diag(1 / self.sigma)
        lim = self.delta_ea_max if conhecida else self.delta_ea_max_nova
        base = theta if conhecida else 1.0
        if self.usar_tensao:
            num = np.einsum("bi,ij,bj->", N, Wm, R0)
            den = np.einsum("bi,ij,bj->", N, Wm, N)
            th = num / den if den > 1e-12 else theta            # N ~ 0 na janela: theta nao muda
            be = beta
        else:                                                   # [theta, beta] por MQ ponderado
            A = np.zeros((2, 2))
            y = np.zeros(2)
            for n_b, r_b in zip(N, R0):
                G = np.column_stack([n_b, e3])
                A += G.T @ Wm @ G
                y += G.T @ Wm @ r_b
            th, be = np.linalg.solve(A + 1e-9 * np.eye(2), y)
            if not (np.isfinite(th) and np.isfinite(be)) or A[0, 0] <= 1e-12:
                th, be = theta, beta
        return float(np.clip(th, base - lim, base + lim)), float(be)

    # ================================================================ calibracao
    def calibrar(self, normais, meta_fa_semana=1.0, h_max=1e4):
        dfs = [normais] if isinstance(normais, pd.DataFrame) else list(normais)
        dfs = [self._preparar(d) for d in dfs]
        Zs, n_tot = [], 0
        for d in dfs:
            Zs.append(self._serie(d)[0])
            n_tot += len(d)
        Z = np.vstack(Zs)
        m = self.m
        if len(Z) <= m.bloco:
            raise ValueError("calibracao curta demais: junte mais trechos normais (minimo > bloco_h em regime)")
        horas_cal = n_tot / m.fs / 3600
        self.horas_regime_semana = 168 * len(Z) * self.media_s / 3600 / horas_cal
        n_sem = int(self.horas_regime_semana * 3600 / self.media_s)
        rng = np.random.default_rng(self.seed_boot)
        nbk = int(np.ceil(n_sem / m.bloco))
        semanas = [np.vstack([Z[i:i + m.bloco] for i in rng.integers(0, len(Z) - m.bloco, nbk)])[:n_sem]
                   for _ in range(self.semanas_boot)]

        def taxa(h):
            return sum(len(m._alarmes(S, h)) for S in semanas) / len(semanas)

        lo, hi = 0.1, h_max
        while hi / lo > 1.02:
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if taxa(mid) > meta_fa_semana else (lo, mid)
        self.h = m.h = float(hi)
        self.meta_fa_semana = meta_fa_semana
        self.fa_boot_semana = taxa(self.h)
        return self

    # ================================================================ analise
    def analisar(self, df, eventos=None):
        if self.h is None:
            raise RuntimeError("chame treinar() e calibrar() antes de analisar()")
        df = self._preparar(df)                                 # colunas, fs, lacunas, NaN (ValueError claro)
        if len(df) < self.media_s * self.m.fs:                  # menos de um bloco
            return []
        Z, idx, dev, log = self._serie(df, diag=True)
        if not len(Z):
            return []
        idx_dec = log["idx_dec"][log["emit"]]
        desvio = _ewma(dev, 0.3)
        t = df["t_s"].to_numpy(float)
        H = matriz_assinaturas()
        out = []
        for i, j, sinal in self.m._alarmes(Z, self.h):
            tipo, rolo = HIPOTESES[j]
            s = H[:, j] * sinal
            amp = float(desvio[i] @ s / (s @ s))                # amplitude na direcao da hipotese (N.m)
            sev = "baixa" if amp < 2.0 else ("media" if amp < 6.0 else "alta")
            out.append({"t_s": float(t[idx_dec[i]]), "t_bloco_s": float(t[idx[i]]), "rolo": rolo, "causa": tipo,
                        "severidade": sev,
                        "modelo": self.nome, "sinal": sinal, "indice": int(idx[i]),
                        "desvio_nm": dict(zip(ROLOS, np.round(desvio[i], 2).tolist())),
                        "mensagem": self.m._mensagem(tipo, rolo, sinal, desvio[i])})
        for n_e, e in enumerate(out):                           # 2 de 3 alarmes em 90 min
            jan = [x for x in out[max(0, n_e - 2): n_e + 1] if e["t_s"] - x["t_s"] <= 5400]
            votos = [(x["causa"], x["rolo"], x["sinal"]) for x in jan]
            melhor = max(set(votos), key=votos.count)
            e["confirmada"] = melhor if votos.count(melhor) >= 2 else None
            if e["confirmada"] and e["severidade"] == "baixa":
                e["severidade"] = "media"
        return out
