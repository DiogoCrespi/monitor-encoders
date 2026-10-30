"""
Arquitetura candidata "CVA + Modelo 2 (+ Modelo 1 opcional)".

Ideia: a CVA (analise de variaveis canonicas) e o melhor DETECTOR sem avisos no simulador v2 (pega
deriva, atrito, perda de pulsos e parte do ruido), mas nao diz o rolo; o Modelo 2 (residuo de torque
+ CUSUM, monitor/torque_v2.py) detecta um pouco menos e ISOLA o rolo pelo padrao de torque nos tres
rolos. Aqui os dois rodam sobre os MESMOS blocos de 60 s (mesmo portao de regime do Modelo 2), e o
Modelo 1 (avisos do drive, monitor/avisos.py) entra se houver log de avisos.

Componentes
    CVA   atributos por bloco de 60 s em regime: media do setpoint, das velocidades, dos torques e das
          tensoes das pontas; log da aspereza (RMS da diferenca a 10 Hz) da velocidade e do torque de
          cada rolo. Passado de p blocos (incluindo o atual), estados canonicos com o futuro de f blocos.
          Estatisticas T2 dos estados retidos e T2 dos estados descartados; p-valor contra a calibracao
          (empirico + cauda exponencial), combinado pelo minimo.
            p      ordem do VAR com menor AIC nos atributos do treino (f = p);
            a      estados retidos = correlacoes canonicas significativas (teste de Bartlett, 1 %).
          Paradas: o passado nao atravessa trechos fora do regime maiores que `ponte_blocos`; depois de
          cada parada (e no inicio do arquivo) a CVA fica cega por p - 1 blocos (aquecimento da janela).
    M2    MonitorTorqueV2 (parametros do monitor conjunto, com tensao das pontas), com uma correcao para
          arquivo que comeca com a linha rodando ("partida a quente"): o estado do aquecimento do
          redutor (F15/F45) comeca em 1 em vez de 0 (`estado_inicial_torque='auto'`).
    M1    MonitorAvisos (razao 8, fator de seguranca 3), opcional.

Fusao (`fusao`)
    'ou'        (padrao, frac_cva = 0,7; frac_avisos = 0,2 com log de avisos) alarme quando QUALQUER componente alarma; a meta de alarme falso (1/semana) e dividida
                entre os componentes (`frac_cva`, `frac_avisos`) e conferida na reamostragem conjunta
                da calibracao (mesmas semanas sorteadas para CVA e Modelo 2).
    'cva_isola' so a CVA dispara (meta inteira para a CVA); o Modelo 2 so isola.
    'm2'        so o Modelo 2 (referencia).
Atribuicao do rolo quando a CVA dispara (`atribuicao`)
    'familia'   (padrao) familia pela energia do desvio padronizado (aspereza do torque x media do torque);
                aspereza -> rolo cujo par de vaos concentra a aspereza; media -> maior RBC entre as 6
                hipoteses do Modelo 2. Dev: 94/100 certas no 1o alarme da CVA (rbc 79, hibrida 67).
    'rbc'       contribuicao por reconstrucao (Alcala & Qin 2009) da CVA ao longo de direcoes fisicas:
                assinatura de encoder nos torques, torque de um rolo so (mecanica), aspereza do torque
                nos rolos vizinhos do encoder (ruido / perda de pulsos);
    'm2'        projecoes do Modelo 2 (media de z dos ultimos `janela_proj` blocos);
    'hibrida'   RBC se a maior direcao for de aspereza (o Modelo 2 nao ve ruido), senao projecoes do M2.
Um alarme do Modelo 2 ate `janela_fusao_min` depois de um alarme da CVA vira 'cva+torque' (severidade
alta se concordarem no rolo).

Interface (arquiteturas/base.py): treinar(normais, eventos_normais, duracao_eventos_s) ->
calibrar(normais, meta_fa_semana) -> analisar(df, eventos) -> lista de eventos com
t_s, rolo, causa ('encoder' | 'mecanica' | None), severidade, modelo (+ mensagem e detalhes).
"""
import warnings

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import chi2

from monitor import SP_AQUECIMENTO
from monitor.avisos import MonitorAvisos
from monitor.conjunto import PARAM_AVISOS, PARAM_TORQUE
from monitor.torque_v2 import HIPOTESES, ROLOS, MonitorTorqueV2, _ENC, _atraso, _ewma

try:                                                    # interface comum (escrita em paralelo)
    from .base import Arquitetura as _Base
except Exception:                                       # pragma: no cover - base ainda ausente
    class _Base:
        """Base minima: treinar -> calibrar -> analisar."""

SEMANA_S = 168 * 3600.0
CAUDA = 50
ENCODERS_PADRAO = {"festoon": "tipo_a", "pb1": "tipo_a", "pb2": "tipo_b"}


# ======================================================================== Modelo 2 com partida a quente
class MonitorTorqueAquecido(MonitorTorqueV2):
    """MonitorTorqueV2 com o estado inicial do aquecimento do redutor escolhido por arquivo.

    No original as entradas F15/F45 (indicador 'linha rodando' filtrado) comecam em 0 = redutor frio.
    Num arquivo que comeca com a linha em producao (coleta iniciada no meio do turno) o redutor ja esta
    quente e o modelo espera ~1,5 N.m de atrito a mais nos tres rolos durante ~45 min.
    estado_inicial: 'frio' (original), 'quente', ou 'auto' (quente se o 1o setpoint > SP_AQUECIMENTO)."""

    def __init__(self, *a, estado_inicial="auto", **kw):
        super().__init__(*a, **kw)
        self.estado_inicial = estado_inicial

    def _x(self, df, rolo):
        sp = df["v_setpoint"].to_numpy(float)
        v = df[f"v_{rolo}"].to_numpy(float) / 60.0
        a = np.gradient(v) * self.fs
        rod = (sp > SP_AQUECIMENTO).astype(float)
        quente = self.estado_inicial == "quente" or (self.estado_inicial == "auto" and len(sp) and
                                                     sp[0] > SP_AQUECIMENTO)
        cols = [sp / 600, v, v ** 2, a] + [_atraso(a, i) for i in range(1, self.lags + 1)]
        for tau_min in (15, 45):
            lam = 1 - np.exp(-1 / (self.fs * 60 * tau_min))
            if quente:
                cols.append(lfilter([lam], [1, -(1 - lam)], rod, zi=[1 - lam])[0])
            else:
                cols.append(_ewma(rod, lam))
        if self.usar_tensao:
            for c in ("tensao_entrada", "tensao_saida"):
                x = df[c].to_numpy(float) / 1000
                cols += [x, _ewma(x, 1 - np.exp(-1 / (self.fs * 2.0)))]
        return np.column_stack(cols + [np.ones(len(df))])


# ======================================================================== p-valores e alarmes por bloco
def p_cauda(r, x, dentro=False, m=CAUDA):
    """p-valor da cauda superior contra a referencia ORDENADA r; acima do m-esimo maior, cauda
    exponencial (POT). dentro=True: x pertence a r (deixa-um-de-fora). Igual a comparacao_metodos.py."""
    k = len(r) - np.searchsorted(r, x, side="left")
    p = np.maximum(k, 1) / len(r) if dentro else (k + 1) / (len(r) + 1)
    n = len(r)
    if n <= m:
        return p
    u = r[n - m - 1]
    beta = max(float(np.mean(r[n - m:] - u)), 1e-12)
    return np.where(x > u, np.maximum((m / n) * np.exp(-(x - u) / beta), 1e-300), p)


def alarmes_limiar(s, L, holdoff):
    """indices onde s > L com intervalo minimo de `holdoff` blocos."""
    out, prox = [], -1
    for i in np.flatnonzero(s > L):
        if i >= prox:
            out.append(int(i))
            prox = i + holdoff
    return out


# ======================================================================== CVA
def _defasar(F, inicio, L):
    """[F_t, F_{t-1}, ..., F_{t-L}] so dentro do mesmo trecho; devolve (X, valido)."""
    seg = np.cumsum(inicio)
    X, ok = [F], np.ones(len(F), bool)
    for i in range(1, L + 1):
        X.append(np.vstack([np.zeros((i, F.shape[1])), F[:-i]]))
        ok[i:] &= seg[i:] == seg[:-i]
    ok[:L] = False
    return np.hstack(X), ok


def _isqrt(M):
    w, V = np.linalg.eigh(M)
    return V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-12))) @ V.T


class CVA:
    """CVA para monitoramento (Russell, Chiang & Braatz 2000; Odiowei & Cao 2010).
    p, f, a = None -> escolhidos nos dados de treino (AIC do VAR; Bartlett a `alfa_estados`)."""

    def __init__(self, p=None, f=None, a=None, p_max=6, criterio="aic", alfa_estados=0.01, ridge=1e-6):
        self.p, self.f, self.a = p, f, a
        self.p_max, self.criterio, self.alfa, self.ridge = p_max, criterio, alfa_estados, ridge

    # ---------------------------------------------------- escolha de hiperparametros (so treino)
    @staticmethod
    def ordem_var(Fs, p_max, criterio="aic"):
        """Ordem do VAR (com intercepto) de menor AIC (Larimore: criterio usual para o passado da CVA) ou
        BIC, ajustado nos mesmos pares validos para todas as ordens (as p_max primeiras amostras de cada
        trecho ficam fora para a comparacao ser justa)."""
        m = Fs[0][0].shape[1]
        Xs, Ys = [], []
        for F, ini in Fs:
            D, ok = _defasar(F, ini, p_max)
            Xs.append(D[ok, m:]); Ys.append(D[ok, :m])
        X, Y = np.vstack(Xs), np.vstack(Ys)
        mu, sd = Y.mean(0), Y.std(0) + 1e-12
        Y = (Y - mu) / sd
        X = (X - np.tile(mu, p_max)) / np.tile(sd, p_max)
        n = len(Y)
        bic = {}
        for k in range(1, p_max + 1):
            Xk = np.hstack([X[:, : k * m], np.ones((n, 1))])
            B = np.linalg.lstsq(Xk, Y, rcond=None)[0]
            E = Y - Xk @ B
            S = E.T @ E / n + 1e-12 * np.eye(m)
            pen = (np.log(n) if criterio == "bic" else 2.0) * k * m * m / n
            bic[k] = float(np.linalg.slogdet(S)[1] + pen)
        return min(bic, key=bic.get), bic

    @staticmethod
    def estados_bartlett(rho, n, dp, df_, alfa):
        """Menor a tal que as correlacoes canonicas a+1.. nao sao significativas (Bartlett)."""
        rho = np.clip(rho, 0, 1 - 1e-12)
        for a in range(len(rho)):
            est = -(n - 1 - (dp + df_ + 1) / 2) * np.sum(np.log(1 - rho[a:] ** 2))
            gl = (dp - a) * (df_ - a)
            if chi2.sf(est, gl) > alfa:
                return max(a, 1)
        return len(rho)

    # ---------------------------------------------------- treino
    def treinar(self, Fs):
        """Fs: lista de (F, inicio) - atributos por bloco e marca de inicio de trecho."""
        if self.p is None:
            self.p, self.criterio_valores = self.ordem_var(Fs, self.p_max, self.criterio)
        self.f = self.p if self.f is None else self.f
        m = Fs[0][0].shape[1]
        P, Fu = [], []
        for F, ini in Fs:
            D, ok = _defasar(F, ini, self.p + self.f - 1)
            Fu.append(D[ok, : self.f * m]); P.append(D[ok, self.f * m:])
        P, Fu = np.vstack(P), np.vstack(Fu)
        self.m_at = m
        self.mp, self.sp = P.mean(0), P.std(0) + 1e-9
        self.mf, self.sf = Fu.mean(0), Fu.std(0) + 1e-9
        P, Fu = (P - self.mp) / self.sp, (Fu - self.mf) / self.sf
        n = len(P)
        Spp = P.T @ P / n + self.ridge * np.eye(P.shape[1])
        Sff = Fu.T @ Fu / n + self.ridge * np.eye(Fu.shape[1])
        Sfp = Fu.T @ P / n
        U, S, Vt = np.linalg.svd(_isqrt(Sff) @ Sfp @ _isqrt(Spp))
        self.rho = S
        if self.a is None:
            self.a = self.estados_bartlett(S, n, P.shape[1], Fu.shape[1], self.alfa)
        Jfull = Vt @ _isqrt(Spp)
        self.J, self.Lr = Jfull[: self.a], Jfull[self.a:]
        self.n_treino = n
        # matriz do indice combinado (cada T2 dividido pelos graus de liberdade) para a RBC
        self.M = self.J.T @ self.J / self.a + self.Lr.T @ self.Lr / max(len(self.Lr), 1)
        return self

    def passado(self, F, inicio):
        """Vetor do passado PADRONIZADO terminando no bloco atual: [F_t .. F_{t-p+1}]."""
        D, ok = _defasar(F, inicio, self.p - 1)
        return (D - self.mp) / self.sp, ok

    def stat(self, F, inicio):
        P, ok = self.passado(F, inicio)
        z, r = P @ self.J.T, P @ self.Lr.T
        return np.column_stack([(z ** 2).sum(1), (r ** 2).sum(1)]), ok, P

    def rbc(self, Pi, direcoes):
        """Contribuicao por reconstrucao de cada direcao (no espaco dos atributos ORIGINAIS, repetida
        nas p defasagens e padronizada) no indice combinado. Devolve vetor (n_direcoes,)."""
        out = []
        for xi in direcoes:
            X = np.tile(xi, self.p) / self.sp
            X = X / (np.linalg.norm(X) + 1e-300)
            MX = self.M @ X
            out.append(float((MX @ Pi) ** 2 / (X @ MX)))
        return np.array(out)


# ======================================================================== arquitetura
class ArquiteturaCVAM2(_Base):
    nome = "cva_m2"

    def __init__(self, fusao="ou", frac_cva=0.7, frac_avisos=0.2, atribuicao="familia",
                 p=None, f=None, a=None, ponte_blocos=0, aspereza_torque=True,
                 estado_inicial_torque="auto", readaptar_receita=True, readapt_blocos=60,
                 janela_fusao_min=120.0, janela_proj=15, limiar_proj=2.0, fs=10, media_s=60,
                 holdoff_min=10.0, semanas_boot=40, bloco_boot_h=2.0, seed_boot=0, encoders=None,
                 mapa_encoders=None, param_torque=None, param_avisos=None):
        if fusao not in ("ou", "cva_isola", "m2", "cva"):
            raise ValueError(f"fusao desconhecida: {fusao}")
        if atribuicao not in ("familia", "rbc", "m2", "hibrida"):
            raise ValueError(f"atribuicao desconhecida: {atribuicao}")
        self.fusao, self.frac_cva, self.frac_avisos, self.atribuicao = fusao, frac_cva, frac_avisos, atribuicao
        self.ponte, self.aspereza_torque = int(ponte_blocos), aspereza_torque
        self.estado_inicial_torque = estado_inicial_torque
        self.readaptar, self.readapt_blocos = readaptar_receita, int(readapt_blocos)
        self.janela_fusao_s, self.janela_proj, self.limiar_proj = janela_fusao_min * 60.0, janela_proj, limiar_proj
        self.fs, self.media_s = fs, int(media_s)
        self.holdoff = max(1, int(round(holdoff_min * 60 / media_s)))
        self.semanas_boot, self.bloco_boot = semanas_boot, max(1, int(bloco_boot_h * 3600 / media_s))
        self.seed_boot = seed_boot
        self.encoders = dict(encoders or ENCODERS_PADRAO)
        self.mapa_encoders = dict(mapa_encoders or {})
        self.param_torque = {**PARAM_TORQUE, **(param_torque or {})}
        self.param_avisos = {**PARAM_AVISOS, **(param_avisos or {})}
        self.cva_kw = dict(p=p, f=f, a=a)
        self.m2 = self.cva = self.m1 = None
        self.L_cva, self.calibrado = np.inf, False
        self._ev_treino, self.draw_ref = None, None

    # ------------------------------------------------------------------ atributos
    def _cols(self, df):
        tens = ["tensao_entrada", "tensao_saida"] if self.m2.usar_tensao else []
        return ["v_setpoint"] + [f"v_{r}" for r in ROLOS] + [f"tq_{r}" for r in ROLOS] + tens

    def nomes_atributos(self):
        n = ["sp"] + [f"v_{r}" for r in ROLOS] + [f"tq_{r}" for r in ROLOS]
        n += ["tensao_entrada", "tensao_saida"] if self.m2.usar_tensao else []
        n += [f"asp_v_{r}" for r in ROLOS]
        n += [f"asp_tq_{r}" for r in ROLOS] if self.aspereza_torque else []
        return n

    def _atributos(self, df):
        """(F, inicio, idx_fim_bloco) dos blocos de 60 s com >= 80 % das amostras em regime (portao do
        Modelo 2: os MESMOS blocos que o Modelo 2 usa). Medias e log da aspereza (RMS da diferenca)."""
        nb = self.media_s * self.fs
        g = self.m2.portao(df)
        n = len(df) // nb
        gg = g[: n * nb].reshape(n, nb)
        cont = gg.sum(axis=1)
        sel = np.flatnonzero(cont >= 0.8 * nb)
        cols = self._cols(df)
        X = df[cols].to_numpy(float)[: n * nb].reshape(n, nb, len(cols))
        w = gg[..., None]
        med = (X * w).sum(axis=1) / np.maximum(cont, 1)[:, None]
        asp_cols = [1, 2, 3] + ([4, 5, 6] if self.aspereza_torque else [])
        dv = np.diff(X[..., asp_cols], axis=1)
        asp = np.sqrt((dv ** 2 * w[:, 1:]).sum(axis=1) / np.maximum(cont, 2)[:, None])
        F = np.hstack([med, np.log(asp + 1e-9)])[sel]
        inicio = np.ones(len(sel), bool)
        inicio[1:] = np.diff(sel) > 1 + self.ponte
        return F, inicio, (sel + 1) * nb - 1

    def _direcoes(self):
        """Direcoes fisicas no espaco dos atributos: [(familia, causa, rolo), vetor].
        media/encoder: assinatura de torque do Modelo 2; media/mecanica: torque de um rolo so;
        aspereza/encoder r: aspereza da velocidade do rolo r e do torque dos rolos dos vaos dele
        (o ruido de medicao, dentro da banda da malha, vira ruido na velocidade REAL e na tensao)."""
        nomes = self.nomes_atributos()
        pos = {n: i for i, n in enumerate(nomes)}
        out = []
        for tipo, r in HIPOTESES:
            xi = np.zeros(len(nomes))
            if tipo == "encoder":
                for rr, s in zip(ROLOS, _ENC[r]):
                    xi[pos[f"tq_{rr}"]] = s
            else:
                xi[pos[f"tq_{r}"]] = 1.0
            out.append((("media", tipo, r), xi))
        for r in ROLOS:
            xi = np.zeros(len(nomes))
            xi[pos[f"asp_v_{r}"]] = 1.0
            if self.aspereza_torque:
                for rr, s in zip(ROLOS, _ENC[r]):
                    if s:
                        xi[pos[f"asp_tq_{rr}"]] = 1.0
            out.append((("aspereza", "encoder", r), xi))
        return out

    # ------------------------------------------------------------------ series por bloco
    _COLS_RECEITA = ("receita", "draw_pb1", "draw_pb2")

    def _na_receita_do_treino(self, df, k):
        """A amostra k esta na receita do treino (mesmo id de receita e mesmo draw)?"""
        for c, ref in (self.draw_ref or {}).items():
            if c in df.columns:
                x = float(df[c].iloc[k])
                if (abs(x - ref) > 1e-3 * abs(ref)) if c != "receita" else (x != ref):
                    return False
        return True

    def _mudancas(self, df, idx):
        """[(bloco, receita_do_treino)] onde comeca um trecho de receita/draw: cada troca das colunas
        'receita', 'draw_pb1', 'draw_pb2' (o simulador do benchmark sempre as grava; receita 0 = a do
        treino) e o 1o bloco se o arquivo ja comeca fora da receita do treino. Voltar a receita do treino
        nao precisa readaptar (offset zero, sem cegueira)."""
        if not self.readaptar or not len(idx) or not any(c in df.columns for c in self._COLS_RECEITA):
            return []
        marcas = np.zeros(len(df), bool)
        for c in self._COLS_RECEITA:
            if c in df.columns:
                x = df[c].to_numpy(float)
                marcas[1:] |= np.abs(np.diff(x)) > 1e-12 * np.maximum(np.abs(x[1:]), 1)
        if not self._na_receita_do_treino(df, 0):
            marcas[0] = True
        out = {}
        for k in np.flatnonzero(marcas):
            b = int(np.searchsorted(idx, k))
            if b < len(idx):
                out[b] = self._na_receita_do_treino(df, int(k))      # vale o ultimo estado no bloco
        return sorted(out.items())

    def _series(self, df):
        """Tudo o que a fusao precisa, por bloco de regime (um so passe pelos sinais)."""
        F, ini, idx = self._atributos(df)
        Z, idx2, R = self.m2._z(df)
        if not np.array_equal(idx2, idx):                # mesmos blocos (mesmo portao e regra de 80 %)
            pos = np.searchsorted(idx, idx2)
            Zf, Rf = np.zeros((len(idx), Z.shape[1])), np.tile(self.m2.mu, (len(idx), 1))
            Zf[pos], Rf[pos] = Z, R
            Z, R = Zf, Rf
        cego = np.zeros(len(idx), bool)
        muds = self._mudancas(df, idx)
        if muds:                                         # readaptacao automatica a receita nova
            F0, R0 = F, R
            F, R = F.copy(), R.copy()
            m0 = self.cva.mp[: self.cva.m_at]
            for k, (b, treinada) in enumerate(muds):
                fim = muds[k + 1][0] if k + 1 < len(muds) else len(idx)
                if treinada:                             # de volta a receita do treino: medias originais
                    F[b:fim], R[b:fim] = F0[b:fim], R0[b:fim]
                    continue
                w = np.arange(b, min(fim, b + self.readapt_blocos))
                cego[w] = True
                if len(w) < self.readapt_blocos:         # sem dado para readaptar: cego ate a proxima
                    cego[b:fim] = True
                    continue
                F[b:fim] -= F[w].mean(0) - m0
                R[b:fim] -= R[w].mean(0) - self.m2.mu
            Z = ((R - self.m2.mu) / self.m2.sigma_e) @ self.m2.W
            Z[cego] = 0.0
        out = {"t": df["t_s"].to_numpy(float)[idx] if len(idx) else np.zeros(0), "idx": idx, "F": F,
               "inicio": ini, "Z": Z, "R": R, "cego": cego, "mudancas": muds}
        if len(F):
            S, ok, P = self.cva.stat(F, ini)
            ok = ok & ~cego
        else:
            S, ok, P = np.zeros((0, 2)), np.zeros(0, bool), np.zeros((0, len(self.cva.mp)))
        out.update(S=S, ok=ok, P=P)
        return out

    def _pval_cva(self, S, ok, dentro=False):
        out = np.zeros(len(S))
        for c in range(S.shape[1]):
            out = np.maximum(out, -np.log10(p_cauda(self.ref_cva[c], S[:, c], dentro)))
        return np.where(ok & np.isfinite(out), out, 0.0)     # NaN nunca vira alarme nem cala a serie

    # ------------------------------------------------------------------ treino
    def treinar(self, normais, eventos_normais=None, duracao_eventos_s=None, contadores_normais=None):
        normais = [normais] if isinstance(normais, pd.DataFrame) else list(normais)
        tensao = all({"tensao_entrada", "tensao_saida"} <= set(d.columns) for d in normais)
        self.m2 = MonitorTorqueAquecido(fs=self.fs, usar_tensao_pontas=tensao,
                                        estado_inicial=self.estado_inicial_torque,
                                        **self.param_torque).treinar(normais)
        self.cva = CVA(**self.cva_kw).treinar([self._atributos(d)[:2] for d in normais])
        self.draw_ref = {c: float(np.median(np.concatenate([d[c].to_numpy(float) for d in normais])))
                         for c in self._COLS_RECEITA if all(c in d.columns for d in normais)}
        self.m1, self._ev_treino = None, None
        if eventos_normais is not None:
            if duracao_eventos_s is None:
                raise ValueError("informe duracao_eventos_s (duracao do historico normal de avisos)")
            if contadores_normais is None:
                contadores_normais = eventos_normais.attrs.get("contadores")
            enc = eventos_normais.attrs.get("encoders") or self.encoders
            self._ev_treino = (eventos_normais, float(duracao_eventos_s), contadores_normais, enc)
            self.m1 = self._montar_m1(*self._ev_treino, meta=self._metas(1.0)["avisos"])
        return self

    def _montar_m1(self, ev, dur, cont, enc, meta):
        return MonitorAvisos(meta_fa_semana=meta, **self.param_avisos).treinar(ev, dur, enc, cont)

    def _metas(self, meta):
        """Divisao da meta de alarme falso entre os componentes que DISPARAM evento."""
        tem_m1 = self._ev_treino is not None
        m1 = meta * self.frac_avisos if tem_m1 else 0.0
        resto = meta - m1
        if self.fusao == "m2":
            return {"cva": 0.0, "torque": resto, "avisos": m1}
        if self.fusao == "cva":
            return {"cva": resto, "torque": 0.0, "avisos": m1}
        if self.fusao == "cva_isola":                    # M2 calibrado so para isolar (nao dispara sozinho)
            return {"cva": resto, "torque": 0.0, "torque_isola": meta, "avisos": m1}
        return {"cva": resto * self.frac_cva, "torque": resto * (1 - self.frac_cva), "avisos": m1}

    # ------------------------------------------------------------------ calibracao
    def calibrar(self, normais, meta_fa_semana=1.0):
        normais = [normais] if isinstance(normais, pd.DataFrame) else list(normais)
        horas = sum(len(d) for d in normais) / self.fs / 3600
        return self.calibrar_series([self._series(d) for d in normais], horas, meta_fa_semana)

    def calibrar_series(self, ser, horas, meta_fa_semana=1.0):
        """Limites pela meta de alarme falso (por semana de calendario), com reamostragem em blocos de
        bloco_boot_h horas de regime; as MESMAS semanas sorteadas para a CVA e o Modelo 2."""
        S = np.vstack([s["S"][s["ok"]] for s in ser])
        self.ref_cva = [np.sort(S[:, c]) for c in range(S.shape[1])]
        pv = np.concatenate([self._pval_cva(s["S"], s["ok"], dentro=True) for s in ser])
        Z = np.vstack([s["Z"] for s in ser])
        E = np.concatenate([s.get("emit", np.ones(len(s["Z"]), bool)) for s in ser])   # blocos que o M2 usa
        n_reg = len(pv)
        if n_reg <= self.bloco_boot:
            raise ValueError("calibracao curta demais para a reamostragem em blocos")
        n_sem = int(n_reg * 168 / horas)
        rng = np.random.default_rng(self.seed_boot)
        nb = int(np.ceil(n_sem / self.bloco_boot))
        sem = [np.concatenate([np.arange(i, i + self.bloco_boot)
                               for i in rng.integers(0, n_reg - self.bloco_boot, nb)])[:n_sem]
               for _ in range(self.semanas_boot)]
        self.metas = self._metas(meta_fa_semana)
        self.L_cva = (self._bissecao(lambda L: sum(len(alarmes_limiar(pv[w], L, self.holdoff)) for w in sem),
                                     self.metas["cva"], len(sem), 1e-2, 1e3)
                      if self.metas["cva"] > 0 else np.inf)
        meta_m2 = self.metas["torque"] or self.metas.get("torque_isola", 0.0)
        self.m2.h = (self._bissecao(lambda h: sum(len(self.m2._alarmes(Z[w][E[w]], h)) for w in sem),
                                    meta_m2, len(sem), 0.1, 1e4) if meta_m2 > 0 else np.inf)
        tot_c = tot_t = 0                                # taxa conjunta (alarmes que viram evento)
        for w in sem:
            tot_c += len(alarmes_limiar(pv[w], self.L_cva, self.holdoff)) if np.isfinite(self.L_cva) else 0
            tot_t += len(self.m2._alarmes(Z[w][E[w]], self.m2.h)) if self.metas["torque"] > 0 else 0
        self.fa_boot_semana = (tot_c + tot_t) / len(sem)  # CVA + M2 (mesmas semanas; sem o M1)
        comp = {"cva": tot_c / len(sem), "torque": tot_t / len(sem), "avisos": None}
        if self._ev_treino is not None:
            self.m1 = self._montar_m1(*self._ev_treino, meta=self.metas["avisos"])
            comp["avisos"] = self._fa_m1_treino()
        self.fa_boot_componentes = comp
        # total com o M1 (taxa do M1 no proprio historico normal de avisos do treino; outra amostra)
        self.fa_boot_total = self.fa_boot_semana + (comp["avisos"] or 0.0)
        meta_tot = sum(v for k, v in self.metas.items() if k in ("cva", "torque", "avisos"))
        baixos = [f"{k} {comp[k]:.2f} (meta {self.metas[k]:.2f})" for k in ("cva", "torque", "avisos")
                  if comp[k] is not None and self.metas.get(k, 0) > 0 and comp[k] < 0.3 * self.metas[k]]
        if meta_tot > 0 and (self.fa_boot_total < 0.3 * meta_tot or baixos):
            # NAO muda os limites (calibracao conservadora documentada): so avisa
            warnings.warn(f"calibracao conservadora: FA na reamostragem {self.fa_boot_total:.2f}/semana contra meta "
                          f"{meta_tot:.2f} (< 0,3 x meta em: {', '.join(baixos) or 'total'}); a bissecao parou num "
                          "degrau definido por poucos blocos atipicos da calibracao", stacklevel=2)
        self.calibrado = True
        return self

    def _fa_m1_treino(self):
        """FA/semana do Modelo 1 no proprio historico normal de avisos do treino (in-sample)."""
        ev, dur, cont, _ = self._ev_treino
        try:
            t0 = float(ev.attrs.get("t_ini_s", 0.0))
            al = self.m1.analisar(ev, t0, t0 + dur, cont)[0]
            return len(al) / (dur / SEMANA_S) if dur > 0 else None
        except Exception:                                # pragma: no cover - so relatorio
            return None

    @staticmethod
    def _bissecao(contar, meta, n, lo, hi):
        while hi / lo > 1.02:
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if contar(mid) / n > meta else (lo, mid)
        return float(hi)

    def recentrar(self, normais):
        """Mitigacao MANUAL para mudanca invisivel nos dados (ex.: papel novo, EA diferente): re-estima so
        as MEDIAS (residuo do Modelo 2 e atributos da CVA) num trecho normal curto da planta nova, sem
        mexer nos limites. Uma falha em curso nesse trecho e absorvida (ver cva_m2.md)."""
        normais = [normais] if isinstance(normais, pd.DataFrame) else list(normais)
        self.m2.mu = np.vstack([self.m2._blocos(d)[0] for d in normais]).mean(axis=0)
        m = self.cva.m_at
        F = np.vstack([self._atributos(d)[0] for d in normais])
        self.cva.mp = self.cva.mp + np.tile(F.mean(0) - self.cva.mp[:m], self.cva.p)
        return self

    # ------------------------------------------------------------------ uso
    def analisar(self, df, eventos=None, contadores=None):
        if not self.calibrado:
            raise RuntimeError("calibrar() antes de analisar()")
        out = self.fundir(self._series(df))
        if self._ev_treino is not None and eventos is not None:
            out += self._avisos(df, eventos, contadores)
        out.sort(key=lambda e: e["t_s"])
        self._corroborar(out)
        return out

    def _avisos(self, df, eventos, contadores):
        """Modelo 1; com historico normal no proprio log (t_s < inicio de df), retreina as taxas com ele
        (como validacao_conjunta.py), com a mesma fatia da meta."""
        t = df["t_s"].to_numpy(float)
        t0 = float(t[0])
        t_fim = float(t[-1] + (t[1] - t[0] if len(t) > 1 else 1.0 / self.fs))
        cont = contadores if contadores is not None else eventos.attrs.get("contadores")
        m1 = self.m1
        hist = eventos[eventos["t_s"] < t0]
        if t0 > 0 and len(hist):
            hc = None if cont is None else cont[cont["t_s"] < t0]
            enc = eventos.attrs.get("encoders") or self._ev_treino[3]
            m1 = self._montar_m1(hist, t0, hc, enc, self.metas["avisos"])
        out = []
        for a in m1.analisar(eventos, t0, t_fim, cont)[0]:
            rolo = self.mapa_encoders.get(a["encoder"], a["encoder"])
            out.append({"t_s": float(a["t_s"]), "rolo": rolo if rolo in ROLOS else None, "causa": "encoder",
                        "incidente": f"m1_{a['encoder']}_{a['familia']}_{a['t_s']:.0f}",
                        "severidade": "baixa" if a["causa_comum"] else "media", "modelo": "avisos",
                        "mensagem": "[Avisos] " + a["mensagem"], "hipotese": ("encoder", rolo)})
        return out

    # ------------------------------------------------------------------ fusao
    def _hip_m2(self, Z, i):
        """Hipotese das projecoes do Modelo 2: soma normalizada de z nos ultimos janela_proj blocos."""
        j0 = max(0, i - self.janela_proj + 1)
        mz = Z[j0:i + 1].sum(0) / np.sqrt(i + 1 - j0)
        j = int(np.argmax(np.abs(mz)))
        return HIPOTESES[j], float(abs(mz[j]))

    def _hip_rbc(self, P, i):
        dirs = self._direcoes()
        c = self.cva.rbc(P[i], [x for _, x in dirs])
        j = int(np.argmax(c))
        return dirs[j][0], float(c[j] / (c.sum() + 1e-300))

    def _hip_familia(self, P, i, n_blocos=3):
        """Atribuicao em dois passos (a escolhida): (1) familia pela energia do desvio padronizado medio
        dos ultimos n_blocos - aspereza do torque (so a parte positiva) contra media do torque; (2) na
        aspereza, o rolo cujo par de vaos (suporte da assinatura |s_r| do Modelo 2) concentra o desvio;
        na media, a maior RBC da CVA entre as 6 hipoteses do Modelo 2 (encoder / mecanica por rolo)."""
        nomes = self.nomes_atributos()
        pos = {n: k for k, n in enumerate(nomes)}
        m = self.cva.m_at
        x = P[max(0, i - n_blocos + 1): i + 1, :m].mean(0)          # bloco atual padronizado (defasagem 0)
        em = x[[pos[f"tq_{r}"] for r in ROLOS]]
        if self.aspereza_torque:
            ea = np.clip(x[[pos[f"asp_tq_{r}"] for r in ROLOS]], 0, None)
            if (ea ** 2).sum() > (em ** 2).sum():
                sc = [ea @ np.abs(np.array(_ENC[r])) for r in ROLOS]
                j = int(np.argmax(sc))
                return "encoder", ROLOS[j], {"metodo": "familia", "direcao": "aspereza",
                                             "fracao": round(float(sc[j] / (sum(sc) + 1e-12)), 2)}
        dirs = [d for d in self._direcoes() if d[0][0] == "media"]
        c = self.cva.rbc(P[i], [d for _, d in dirs])
        j = int(np.argmax(c))
        _, tipo, rolo = dirs[j][0]
        return tipo, rolo, {"metodo": "familia", "direcao": "media",
                            "fracao": round(float(c[j] / (c.sum() + 1e-300)), 2)}

    def atribuir(self, ser, i, metodo=None):
        """(causa, rolo, detalhe) para um alarme da CVA no bloco i."""
        metodo = metodo or self.atribuicao
        if metodo == "familia":
            return self._hip_familia(ser["P"], i)
        (fam, tipo_r, rolo_r), fr = self._hip_rbc(ser["P"], i)
        (tipo_m, rolo_m), forca = self._hip_m2(ser["Z"], i)
        rbc = (tipo_r, rolo_r, {"metodo": "rbc", "direcao": fam, "fracao": round(fr, 2)})
        m2 = (tipo_m, rolo_m, {"metodo": "m2", "forca": round(forca, 2)})
        if metodo == "rbc":
            return rbc
        if metodo == "m2":
            return m2 if forca >= self.limiar_proj else (None, None, m2[2])
        if fam == "aspereza" or forca < self.limiar_proj:   # hibrida
            return rbc
        return m2

    def fundir(self, ser):
        """Alarmes da CVA e do M2. Com ser['dec'] (modo causal do M2 adaptativo), o alarme de um bloco
        retido sai no instante da DECISAO (t_s) e o instante do bloco vai em 't_bloco_s'."""
        t, Z, R = ser["t"], ser["Z"], ser["R"]
        out = []
        if not len(t):
            return out
        td = t[ser["dec"]] if "dec" in ser else t           # instante em que o alarme pode sair
        pv = self._pval_cva(ser["S"], ser["ok"])
        cva_al = alarmes_limiar(pv, self.L_cva, self.holdoff) if np.isfinite(self.L_cva) else []
        for i in cva_al:
            causa, rolo, det = self.atribuir(ser, i)
            out.append({"t_s": float(td[i]), "t_bloco_s": float(t[i]), "rolo": rolo, "causa": causa,
                        "incidente": f"cva_{i}",
                        "severidade": "media" if rolo else "baixa", "modelo": "cva",
                        "mensagem": self._msg_cva(causa, rolo, det, pv[i]), "bloco": int(i),
                        "hipotese": (causa, rolo), "detalhe": det})
        emit = ser.get("emit")
        if emit is None:
            m2_al = self.m2._alarmes(Z, self.m2.h) if np.isfinite(self.m2.h) else []
        else:                                            # M2 so ve os blocos que emitiu (quarentenas fora)
            pos = np.flatnonzero(emit)
            m2_al = [(int(pos[i]), j, sg) for i, j, sg in self.m2._alarmes(Z[emit], self.m2.h)]                 if np.isfinite(self.m2.h) and len(pos) else []
        desvio = _ewma(R - (ser["mu_R"] if "mu_R" in ser else self.m2.mu), 0.3)
        hist = []
        for i, j, sinal in m2_al:
            tipo, rolo = HIPOTESES[j]
            hist.append((td[i], tipo, rolo, sinal))
            votos = [h[1:] for h in hist[-3:] if td[i] - h[0] <= 5400]
            melhor = max(set(votos), key=votos.count)
            conf = melhor if votos.count(melhor) >= 2 else None
            ant = [e for e in out if e["modelo"] == "cva" and 0 <= td[i] - e["t_s"] <= self.janela_fusao_s]
            if self.fusao == "cva_isola" and not ant:    # M2 so isola: sem CVA antes, nao vira evento
                continue
            concorda = any(e["hipotese"] == (tipo, rolo) for e in ant)
            # confirmacao do M2 ate janela_fusao depois de um alarme da CVA = mesmo incidente (o mais recente)
            inc = ant[-1]["incidente"] if ant else f"m2_{i}"
            out.append({"t_s": float(td[i]), "t_bloco_s": float(t[i]), "rolo": rolo, "causa": tipo, "incidente": inc,
                        "severidade": "alta" if (conf and conf[:2] == (tipo, rolo)) or concorda else "media",
                        "modelo": "cva+torque" if ant else "torque",
                        "mensagem": "[Torque] " + self.m2._mensagem(tipo, rolo, sinal, desvio[i]),
                        "bloco": int(i), "hipotese": (tipo, rolo), "confirmada": conf})
        out.sort(key=lambda e: e["t_s"])
        return out

    @staticmethod
    def _msg_cva(causa, rolo, det, lp):
        base = f"[CVA] Padrao conjunto dos sinais fora do normal (-log10 p = {lp:.1f})."
        if rolo is None:
            return base + " Rolo nao identificado; aguardar o Modelo 2 ou inspecionar os tres."
        if det.get("direcao") == "aspereza":
            return (base + f" Aspereza de velocidade/torque ligada ao encoder do {rolo.upper()}: ruido ou "
                    "perda de pulsos. Acao: conector, cabo, blindagem e disco do encoder.")
        if causa == "mecanica":
            return base + f" Torque do {rolo.upper()} sozinho: suspeita MECANICA (rolamento, redutor)."
        return base + f" Padrao de torque de erro de escala no encoder do {rolo.upper()}."

    @staticmethod
    def _corroborar(ev, janela_s=7200.0):
        """Sobe para 'alta' quando outro modelo apontou o mesmo rolo e causa nas ultimas 2 h."""
        for k, e in enumerate(ev):
            if e["rolo"] is None:
                continue
            for a in ev[:k]:
                if (a["modelo"] != e["modelo"] and a["rolo"] == e["rolo"] and a["causa"] == e["causa"]
                        and e["t_s"] - a["t_s"] <= janela_s and a["severidade"] != "baixa"):
                    e["severidade"] = "alta"
                    break

    # ------------------------------------------------------------------ resumo
    def resumo(self):
        return {"p": self.cva.p, "f": self.cva.f, "a": self.cva.a, "n_atributos": self.cva.m_at,
                "metas": getattr(self, "metas", None), "L_cva": self.L_cva, "h_m2": self.m2.h,
                "fa_boot_semana": getattr(self, "fa_boot_semana", None),
                "fa_boot_componentes": getattr(self, "fa_boot_componentes", None),
                "fa_boot_total": getattr(self, "fa_boot_total", None)}


__all__ = ["ArquiteturaCVAM2", "CVA", "MonitorTorqueAquecido"]
