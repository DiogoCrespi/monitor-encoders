"""
Arquitetura candidata "CVA + Modelo 2 adaptativo (+ Modelo 1 opcional)" (cva_m2a).

Igual a arquiteturas/cva_m2.py (ArquiteturaCVAM2: fusao OU, meta CVA 0,7 / M2 0,3, M1 0,2 com log de
avisos, atribuicao 'familia', mesmos atributos e hiperparametros da CVA), com o componente de torque
trocado por arquiteturas/m2_adaptativo.py (ArquiteturaM2Adaptativo, parametros padrao) e a CVA seguindo a
MESMA adaptacao de planta que ele:

  1. Torque fisico na CVA. Os atributos de media de torque da CVA recebem a mesma correcao fisica do M2
     adaptativo: tq_k - (theta_b * n_k,b - n_k,treino), com n = r*EA_nom*(-D1, D1-D2, D2) por bloco e
     theta_b o multiplicador de EA que o M2 adaptativo esta usando naquele bloco (por receita, rastreado
     devagar, ou trocado num degrau de material aceito). Uma troca de draw ou um EA novo absorvido pelo M2
     deixa de deslocar a media de torque que a CVA ve - sem cegueira.
  2. Blocos que o M2 adaptativo nao emite (quarentena de troca de receita, degrau de material suspenso ou
     descartado) ficam fora da CVA tambem (ok = False). Um degrau de EA sem aviso, aceito pelo M2, nao
     gera alarme da CVA nos blocos da quarentena do degrau.
  3. Troca de receita aceita pelo M2 (log 'trocas'): a CVA readapta as OUTRAS medias (tensao das pontas,
     velocidade) como em ArquiteturaCVAM2 - `readapt_blocos` blocos de regime cegos e medias re-estimadas;
     volta para a receita do treino = medias originais, sem cegueira.
  4. Draw: o M2 adaptativo usa a coluna do CLP; aqui `fonte_draw='cruzado'` usa o draw lido pelos
     encoders (media movel de 60 s de (v_pb - v_fest)/v_fest, que a malha faz seguir o comando) onde ele
     diverge do CLP por mais de `tol_draw` durante 10 min seguidos (coluna parada/atrasada), e o CLP no
     resto. Sem as colunas do CLP: so o draw dos encoders.

MODO CAUSAL (padrao, `causal=True`; ver arquiteturas/m2_adaptativo.py): nenhum alarme e retratado. Os
blocos da quarentena de um degrau de material suspeito ficam RETIDOS ate a decisao do M2 adaptativo; se a
decisao e "rampa", os alarmes da CVA e do M2 desses blocos saem com t_s = instante da DECISAO (o instante
do bloco vai em 't_bloco_s'); se e "degrau", os blocos sao descartados. Fim de arquivo com decisao
pendente: os blocos retidos nao alarmam. `causal=False` (ArquiteturaCVAM2ALote) reproduz a decisao em lote
antiga - so para comparacao.

ENTRADA (arquiteturas/m2_adaptativo.preparar_entrada; unidades no docstring de m2_adaptativo.py):
validada em treinar, calibrar e analisar. Tensao das pontas ausente no treino -> modo degradado
(usar_tensao_pontas=False, com aviso), decidido no treino e mantido na analise (arquivo sem tensao para um
modelo treinado com tensao -> ValueError). Sem draw_pb1/draw_pb2 -> draw dos encoders; sem 'receita' ->
receita unica; sem log de avisos -> Modelo 1 desligado. NaN -> blocos fora do regime; lacuna > 2 s em t_s
-> parada; amostragem diferente de 10 Hz -> ValueError (ou `reamostrar=True`).

CONFIGURACAO DA PLANTA: `raio` ({rolo: raio/relacao} em m) e `ea_nominal` (N) vao para o M2 adaptativo;
os padroes sao as constantes do simulador (poc/malha_fechada_v2.R = 0,15 m, .EA = 2e6 N).

Os limites sao calibrados como na ArquiteturaCVAM2 (mesmas semanas reamostradas; o M2 so conta os blocos
que emite). Nenhum hiperparametro dos dois componentes foi mudado; o h do M2 sai da fatia 0,3 da meta.
"""
import warnings

import numpy as np
import pandas as pd

from .cva_m2 import CVA, ArquiteturaCVAM2
from .m2_adaptativo import COLS_DRAW, COLS_TENSAO, LACUNA_S, ArquiteturaM2Adaptativo, preparar_entrada

ROLOS = ("festoon", "pb1", "pb2")


class M2AdaptativoCruzado(ArquiteturaM2Adaptativo):
    """ArquiteturaM2Adaptativo com fonte de draw 'cruzado': CLP, trocado pelo draw lido pelos encoders onde
    os dois divergem (coluna do CLP parada ou atrasada)."""

    def __init__(self, *a, tol_draw=0.02, **kw):
        super().__init__(*a, **kw)
        self.tol_draw = tol_draw

    def _draw(self, df, fonte=None):
        fonte = fonte or self.fonte_draw
        if fonte != "cruzado":
            return super()._draw(df, fonte)
        if not set(COLS_DRAW) <= set(df.columns):            # sem CLP: so os encoders
            return super()._draw(df, "encoders")
        e1, e2, _ = super()._draw(df, "encoders")            # sem mudar self.fonte_draw (F6)
        out = []
        for c, e in (("draw_pb1", e1), ("draw_pb2", e2)):
            clp = df[c].to_numpy(float)
            diverge = np.abs(e - clp) > self.tol_draw * np.maximum(np.abs(clp), 1e-9)
            # so troca com divergencia SUSTENTADA por 10 min (coluna parada), nao em rampas/paradas
            jan = 600 * self.m.fs
            diverge = pd.Series(diverge.astype(float)).rolling(jan, min_periods=jan).min().fillna(0.0) \
                .to_numpy() > 0
            out.append(np.where(diverge, e, clp))
        return out[0], out[1], "cruzado"


class ArquiteturaCVAM2A(ArquiteturaCVAM2):
    nome = "cva_m2a"

    def __init__(self, param_m2a=None, tol_draw=0.02, causal=True, raio=None, ea_nominal=None,
                 reamostrar=False, lacuna_s=LACUNA_S, **kw):
        kw.setdefault("readaptar_receita", True)
        super().__init__(**kw)
        self.causal = bool(causal)
        self.reamostrar, self.lacuna_s = reamostrar, lacuna_s
        extra = {"causal": self.causal, "reamostrar": reamostrar, "lacuna_s": lacuna_s}
        if raio is not None:
            extra["raio"] = raio
        if ea_nominal is not None:
            extra["ea_nominal"] = ea_nominal
        self.param_m2a = {"fonte_draw": "cruzado", **extra, **(param_m2a or {})}
        self.tol_draw = tol_draw
        self.ma = None

    # ------------------------------------------------------------------ entrada
    def _preparar(self, df, usar_tensao=None):
        u = self.ma.usar_tensao if usar_tensao is None else usar_tensao
        return preparar_entrada(df, self.fs, u, getattr(self, "reamostrar", False),
                                getattr(self, "lacuna_s", LACUNA_S))[0]

    # ------------------------------------------------------------------ componentes
    def _tq_cols(self):
        pos = {n: k for k, n in enumerate(self.nomes_atributos())}
        return [pos[f"tq_{r}"] for r in ROLOS]

    def _corrigir_torque(self, F, N, theta):
        """Media de torque da CVA com a correcao fisica do M2 adaptativo."""
        F = F.copy()
        F[:, self._tq_cols()] -= theta[:, None] * N - self.n_ref[None, :]
        return F

    def _blocos_m2a(self, df, idx):
        """Saida do M2 adaptativo alinhada aos blocos da CVA (mesmo portao e regra de 80 %). Tudo local
        (nada e gravado no objeto). dec[i] = bloco da CVA em que a emissao do bloco i foi decidida."""
        bl = self.ma._blocos(df)
        Ze, idx_e, dev_e, log = self.ma._serie(None, diag=True, _blocos=bl)
        R0, N, idx_m, rec = bl
        n = len(idx)
        pos_m = np.searchsorted(idx, idx_m)
        ok_m = (pos_m < n) & (idx[np.minimum(pos_m, n - 1)] == idx_m) if n else np.zeros(0, bool)
        Nf, theta = np.zeros((n, 3)), np.ones(n)
        Nf[pos_m[ok_m]] = N[ok_m]
        # blocos da quarentena de receita: theta REESTIMADO no fim dela (o do log e o antigo). Eles sao cegos
        # e so entram na media de readaptacao da CVA, usada depois que a janela (>= quarentena) termina
        th = pd.Series(np.where(log["quar"], np.nan, log["theta"])).bfill().ffill()
        th = th.fillna(pd.Series(log["theta"]).ffill().bfill()).fillna(1.0).to_numpy()
        theta[pos_m[ok_m]] = th[ok_m]
        recf = np.full(n, self.ma.receita_ref, dtype=object)
        recf[pos_m[ok_m]] = rec[ok_m]
        emit = np.isin(idx, idx_e)
        Z, R = np.zeros((n, Ze.shape[1] if len(Ze) else 6)), np.zeros((n, 3))
        pe = np.searchsorted(idx, idx_e)
        Z[pe], R[pe] = Ze, dev_e
        dec = np.arange(n)
        if n and len(idx_m):
            dm = np.minimum(np.searchsorted(idx, log["idx_dec"]), n - 1)
            dec[pos_m[ok_m]] = np.maximum(dm[ok_m], pos_m[ok_m])
        return Z, R, emit, Nf, theta, recf, log, dec

    def treinar(self, normais, eventos_normais=None, duracao_eventos_s=None, contadores_normais=None):
        normais = [normais] if isinstance(normais, pd.DataFrame) else list(normais)
        # F8: tensao das pontas decidida no treino pelas colunas disponiveis (modo degradado sem elas)
        tensao = all(set(COLS_TENSAO) <= set(d.columns) and all(np.isfinite(d[c].to_numpy(float)).any()
                                                                 for c in COLS_TENSAO) for d in normais)
        kw = dict(self.param_m2a)
        if kw.get("usar_tensao_pontas", "auto") == "auto":
            kw["usar_tensao_pontas"] = tensao
            if not tensao:
                warnings.warn("treino sem tensao_entrada/tensao_saida: MODO DEGRADADO (usar_tensao_pontas=False "
                              "no M2 adaptativo e na CVA)", stacklevel=2)
        if not all(set(COLS_DRAW) <= set(d.columns) for d in normais):
            warnings.warn("treino sem draw_pb1/draw_pb2: draw lido pelos encoders", stacklevel=2)
        if eventos_normais is None:
            warnings.warn("treino sem log de avisos: Modelo 1 desligado", stacklevel=2)
        normais = [preparar_entrada(d, self.fs, kw["usar_tensao_pontas"], self.reamostrar, self.lacuna_s)[0]
                   for d in normais]
        self.ma = M2AdaptativoCruzado(tol_draw=self.tol_draw, **kw).treinar(normais)
        self.m2 = self.ma.m                                  # portao, _alarmes, _mensagem, usar_tensao, h
        Fs, Ns = [], []
        for d in normais:
            F, ini, idx = self._atributos(d)
            _, _, _, N, theta, _, _, _ = self._blocos_m2a(d, idx)
            Fs.append((F, ini, N, theta))
            Ns.append(N)
        self.n_ref = np.vstack(Ns).mean(axis=0)
        self.cva = CVA(**self.cva_kw).treinar([(self._corrigir_torque(F, N, th), ini) for F, ini, N, th in Fs])
        self.draw_ref = None                                 # trocas vem do M2 adaptativo
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

    # ------------------------------------------------------------------ series
    def _series(self, df):
        """Por bloco de regime (df ja preparado). Todo o estado e local a este arquivo."""
        F, ini, idx = self._atributos(df)
        Z, R, emit, N, theta, rec, log, dec = self._blocos_m2a(df, idx)
        F = self._corrigir_torque(F, N, theta)
        cego = ~emit
        muds = []
        if self.readaptar and len(idx):
            for i10, rb, _ in log["trocas"]:
                b = int(np.searchsorted(idx, i10))
                if b < len(idx):
                    muds.append((b, rb == self.ma.receita_ref))
            if len(rec) and rec[0] != self.ma.receita_ref and (not muds or muds[0][0] != 0):
                muds.insert(0, (0, False))
        if muds:
            F0 = F
            F = F.copy()
            m0 = self.cva.mp[: self.cva.m_at]
            for k, (b, treinada) in enumerate(muds):
                fim = muds[k + 1][0] if k + 1 < len(muds) else len(idx)
                if treinada:
                    F[b:fim] = F0[b:fim]
                    continue
                w = np.arange(b, min(fim, b + self.readapt_blocos))
                cego[w] = True
                if len(w) < self.readapt_blocos:
                    cego[b:fim] = True
                    continue
                F[b:fim] -= F[w].mean(0) - m0
        out = {"t": df["t_s"].to_numpy(float)[idx] if len(idx) else np.zeros(0), "idx": idx, "F": F,
               "inicio": ini, "Z": Z, "R": R, "mu_R": np.zeros(3), "emit": emit, "cego": cego,
               "mudancas": muds, "theta": theta, "degraus": log["degraus"],
               "dec": dec if getattr(self, "causal", False) else np.arange(len(idx)),
               "pendentes_fim": log.get("pendentes_fim", 0)}
        if len(F):
            S, ok, P = self.cva.stat(F, ini)
            ok = ok & ~cego & np.isfinite(S).all(axis=1)
        else:
            S, ok, P = np.zeros((0, 2)), np.zeros(0, bool), np.zeros((0, len(self.cva.mp)))
        out.update(S=S, ok=ok, P=P)
        return out

    def calibrar(self, normais, meta_fa_semana=1.0):
        normais = [normais] if isinstance(normais, pd.DataFrame) else list(normais)
        return super().calibrar([self._preparar(d) for d in normais], meta_fa_semana)

    def calibrar_series(self, ser, horas, meta_fa_semana=1.0):
        super().calibrar_series(ser, horas, meta_fa_semana)
        self.ma.h = self.m2.h
        return self

    def analisar(self, df, eventos=None, contadores=None):
        """Nao muda o objeto treinado (todo o estado do arquivo e local)."""
        if not self.calibrado:
            raise RuntimeError("calibrar() antes de analisar()")
        df = self._preparar(df)
        out = []
        if len(df) >= self.media_s * self.fs:                # CVA/M2 precisam de pelo menos um bloco
            out = self.fundir(self._series(df))
        if self._ev_treino is not None and eventos is not None and len(df):
            out += self._avisos(df, eventos, contadores)
        out.sort(key=lambda e: e["t_s"])
        self._corroborar(out)
        return out

    def recentrar(self, normais):
        """Nao usado: a adaptacao de planta vem do M2 adaptativo."""
        raise NotImplementedError("ArquiteturaCVAM2A segue a adaptacao do M2 adaptativo")


class ArquiteturaCVAM2ALote(ArquiteturaCVAM2A):
    """Decisao de degrau em LOTE (comportamento antigo, nao causal): so para comparacao."""
    nome = "cva_m2a_lote"

    def __init__(self, **kw):
        kw["causal"] = False
        super().__init__(**kw)


__all__ = ["ArquiteturaCVAM2A", "ArquiteturaCVAM2ALote", "M2AdaptativoCruzado"]
