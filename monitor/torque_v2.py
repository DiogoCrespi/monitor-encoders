"""
Modelo 2 - torque esperado x torque real, com isolacao do encoder (versao 2).

Entrada: o mesmo CSV do MonitorEncoder (t_s, v_setpoint, v_<rolo>, tq_<rolo>) com os TRES
rolos (festoon, pb1, pb2). Opcional: tensao_entrada / tensao_saida (celula de carga ou
bailarino das pontas), usadas como entradas se `usar_tensao_pontas=True`.

1. Gerador de residuo por rolo (regressao com dinamica, so dado normal, so em regime):
       r_k(t) = tq_k(t) - [ g0 + g1 sp + g2 v_k + g3 v_k^2 + sum_{i=0..L} gA_i a_k(t-i)
                            + gT1 F15(t) + gT2 F45(t) (+ gE T_entrada + gS T_saida) ]
   F15/F45 = indicador "linha rodando" filtrado com tau de 15 e 45 min (aquecimento do redutor).
   A saida passada tq_k(t-1) NAO entra: o preditor seguiria a deriva.
2. Portao de regime: so conta com setpoint >= v_min e parado ha >= janela_s (fora disso -
   parada, rampa, emenda - a estatistica fica congelada).
3. Medias em blocos de `media_s` segundos do residuo e branqueamento AR(p) opcional por rolo:
   e_k = A_k(q) r_k. Passo 1 + passo 3 = modelo ARX  A(q) tq = A(q) B(q) x + e.
4. Residuos estruturados (filtro casado): cada hipotese j tem uma assinatura s_j nos tres torques
       encoder festoon (+1, 0, -1)   encoder pb1 (-1, +1, 0)   encoder pb2 (0, -1, +1)
       mecanica k      (vetor unitario do rolo k)  - atrito/rolamento, nao e encoder
   z_j = s_j' S^-1 e / sqrt(s_j' S^-1 s_j), em unidades de sigma (S = covariancia de e).
5. CUSUM bilateral em cada z_j (12 estatisticas); alarme quando a maior passa h; a hipotese
   da maior e a isolacao. Reinicia depois do alarme; intervalo minimo `holdoff_min`.
6. h calibrado por META de falso alarme em tempo (ex.: 1 por semana) com reamostragem em
   blocos (`bloco_h`; v2.1 usa 2 h) do residuo normal, reproduzindo portao, persistencia,
   reinicio e intervalo.
"""
import json

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import SP_AQUECIMENTO, V_MIN_REGIME

ROLOS = ("festoon", "pb1", "pb2")
_ENC = {"festoon": (1.0, 0.0, -1.0), "pb1": (-1.0, 1.0, 0.0), "pb2": (0.0, -1.0, 1.0)}
HIPOTESES = [("encoder", r) for r in ROLOS] + [("mecanica", r) for r in ROLOS]
# coluna booleana opcional: amostra fora do regime por dado invalido (NaN, lacuna), marcada pelo preparo da
# entrada (arquiteturas/m2_adaptativo.preparar_entrada); o portao a respeita
COL_FORA = "_fora_regime"


def _ewma(x, lam):
    return lfilter([lam], [1, -(1 - lam)], x)


def _atraso(x, k):
    return np.concatenate([np.full(k, x[0]), x[:-k]])


def matriz_assinaturas():
    """3 x 6: colunas = efeito nos torques (festoon, pb1, pb2) de cada hipotese (sinal +).
    Encoder: lendo A MENOS (rolo girando rapido). Mecanica: torque do proprio rolo sobe."""
    H = np.zeros((3, len(HIPOTESES)))
    for j, (tipo, rolo) in enumerate(HIPOTESES):
        H[:, j] = _ENC[rolo] if tipo == "encoder" else np.eye(3)[ROLOS.index(rolo)]
    return H


class MonitorTorqueV2:
    def __init__(self, fs=10, k=0.5, h=None, media_s=60, persist_s=60, holdoff_min=10.0,
                 ordem_ar=0, lags_acel=5, janela_s=30.0, v_min=V_MIN_REGIME, estatistica="cusum",
                 lam=0.1, usar_tensao_pontas=False, hipoteses_mecanicas=True,
                 meta_fa_semana=1.0, bloco_h=1.0, semanas_boot=40, seed_boot=0):
        self.fs, self.k, self.h, self.media_s = fs, k, h, int(media_s)
        self.persist = max(1, int(np.ceil(persist_s / media_s)))
        self.holdoff = max(1, int(round(holdoff_min * 60 / media_s)))
        self.ordem_ar, self.lags = ordem_ar, lags_acel
        self.janela_s, self.janela, self.v_min = janela_s, int(janela_s * fs), v_min
        self.estatistica, self.lam = estatistica, lam
        self.usar_tensao, self.mecanicas = usar_tensao_pontas, hipoteses_mecanicas
        self.meta, self.semanas, self.seed_boot = meta_fa_semana, semanas_boot, seed_boot
        self.bloco = max(1, int(bloco_h * 3600 / media_s))

    # ------------------------------------------------------------ sinais
    def portao(self, df):
        """True quando a linha esta em regime: setpoint alto e sem mudanca na janela."""
        s = df["v_setpoint"].astype(float)
        faixa = (s.rolling(self.janela, min_periods=1).max()
                 - s.rolling(self.janela, min_periods=1).min()).to_numpy()
        ok = (s.to_numpy() >= self.v_min) & (faixa < 1.0)
        ok[: self.janela] = False
        if COL_FORA in df.columns:
            ok &= ~df[COL_FORA].to_numpy(bool)
        return ok

    def _x(self, df, rolo):
        sp = df["v_setpoint"].to_numpy(float)
        v = df[f"v_{rolo}"].to_numpy(float) / 60.0
        a = np.gradient(v) * self.fs
        rod = (sp > SP_AQUECIMENTO).astype(float)          # motor girando (aquece o redutor)
        cols = [sp / 600, v, v ** 2, a] + [_atraso(a, i) for i in range(1, self.lags + 1)]
        for tau_min in (15, 45):
            cols.append(_ewma(rod, 1 - np.exp(-1 / (self.fs * 60 * tau_min))))
        if self.usar_tensao:
            for c in ("tensao_entrada", "tensao_saida"):
                x = df[c].to_numpy(float) / 1000
                cols += [x, _ewma(x, 1 - np.exp(-1 / (self.fs * 2.0)))]
        return np.column_stack(cols + [np.ones(len(df))])

    def _residuo10(self, df):
        return np.column_stack([df[f"tq_{r}"].to_numpy(float) - self._x(df, r) @ self.coef[r]
                                for r in ROLOS])

    def _blocos(self, df):
        """Media do residuo em blocos de media_s com >= 80 % das amostras em regime.
        Devolve (R, indice 10 Hz do fim do bloco, inicio_de_trecho)."""
        nb = self.media_s * self.fs
        g = self.portao(df)
        r = self._residuo10(df)
        n = len(df) // nb
        gg = g[: n * nb].reshape(n, nb)
        cont = gg.sum(axis=1)
        R = (r[: n * nb] * g[: n * nb, None]).reshape(n, nb, 3).sum(axis=1) / np.maximum(cont, 1)[:, None]
        sel = np.flatnonzero(cont >= 0.8 * nb)
        inicio = np.ones(len(sel), bool)
        inicio[1:] = np.diff(sel) > 1
        return R[sel], (sel + 1) * nb - 1, inicio

    def _branquear(self, R, inicio):
        """e = A(q)(r - mu) dentro de cada trecho continuo; as p primeiras amostras do trecho
        nao sao usadas (valido=False)."""
        p = self.ordem_ar
        X = R - self.mu
        E = X.copy()
        valido = np.ones(len(R), bool)
        if p:
            seg = np.cumsum(inicio)
            for i in range(1, p + 1):
                E -= self.phi[i - 1] * np.vstack([np.zeros((i, 3)), X[:-i]])
                valido[i:] &= seg[i:] == seg[:-i]
            valido[:p] = False
        return E / self.sigma_e, valido

    # ------------------------------------------------------------ treino
    def treinar(self, dfs, alpha=1e-3):
        dfs = [dfs] if isinstance(dfs, pd.DataFrame) else list(dfs)
        self.coef = {}
        for r in ROLOS:
            X = np.vstack([self._x(d, r)[self.portao(d)] for d in dfs])
            y = np.concatenate([d[f"tq_{r}"].to_numpy(float)[self.portao(d)] for d in dfs])
            self.coef[r] = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)
        partes = [self._blocos(d) for d in dfs]
        R = np.vstack([p[0] for p in partes])
        inicio = np.concatenate([p[2] for p in partes])
        self.mu = R.mean(axis=0)
        p = self.ordem_ar
        self.phi = np.zeros((p, 3))
        if p:
            seg = np.cumsum(inicio)
            ok = np.ones(len(R), bool)
            ok[:p] = False
            for i in range(1, p + 1):
                ok[i:] &= seg[i:] == seg[:-i]
            for c in range(3):
                x = R[:, c] - self.mu[c]
                L = np.column_stack([np.concatenate([np.zeros(i), x[:-i]]) for i in range(1, p + 1)])
                self.phi[:, c] = np.linalg.lstsq(L[ok], x[ok], rcond=None)[0]
        self.sigma_e = np.ones(3)
        E, val = self._branquear(R, inicio)
        self.sigma_e = E[val].std(axis=0)
        self.S = np.corrcoef((E[val] / self.sigma_e).T)
        self._montar_filtro()
        return self

    def _montar_filtro(self):
        ganho = (1 - self.phi.sum(axis=0)) / self.sigma_e
        Si = np.linalg.inv(self.S)
        H = matriz_assinaturas()
        n_h = len(HIPOTESES) if self.mecanicas else 3
        W = []
        for j in range(n_h):
            s = H[:, j] * ganho
            W.append(Si @ s / np.sqrt(s @ Si @ s))
        self.W = np.array(W).T                                    # 3 x n_h

    def _z(self, df):
        R, idx, inicio = self._blocos(df)
        E, val = self._branquear(R, inicio)
        return E[val] @ self.W, idx[val], R[val]

    # ------------------------------------------------------------ estatistica e regras
    def _passo(self, Zb, c0):
        """Estatistica num trecho a partir do estado c0. CUSUM de Page vetorizada:
        C_t = max(0, C_{t-1} + z_t - k) = S_t - min(-c0, min_{s<=t} S_s)."""
        if self.estatistica == "cusum":
            S = np.cumsum(Zb - self.k, axis=0)
            return S - np.minimum(np.minimum.accumulate(S, axis=0), -c0)
        g = np.sqrt(self.lam / (2 - self.lam))
        out, _ = lfilter([self.lam], [1, -(1 - self.lam)], Zb, axis=0,
                         zi=(c0 * g * (1 - self.lam))[None, :])
        return out / g                                            # EWMA em unidades de sigma

    def _alarmes(self, Z, h, bloco=2000):
        """Varre Z (um bloco por linha) com estado; devolve [(indice, hipotese, sinal)].
        Alarme quando max > h por `persist` blocos seguidos e fora do intervalo minimo;
        depois do alarme a estatistica volta a zero. Linhas nao finitas de Z sao puladas (estatistica
        congelada nelas, como fora do regime) - um NaN nunca cala a CUSUM."""
        fin = np.isfinite(Z).all(axis=1) if len(Z) else np.ones(0, bool)
        if not fin.all():
            pos_f = np.flatnonzero(fin)
            return [(int(pos_f[i]), j, s) for i, j, s in self._alarmes(Z[fin], h, bloco)]
        Zb_all = np.hstack([Z, -Z])
        nh = Z.shape[1]
        out, pos, c, run, min_prox = [], 0, np.zeros(2 * nh), 0, 0
        while pos < len(Z):
            C = self._passo(Zb_all[pos:pos + bloco], c)
            acima = C.max(axis=1) > h
            ar = np.arange(len(C))
            cont = ar - np.maximum.accumulate(np.where(~acima, ar, -1 - run))
            cand = np.flatnonzero((cont >= self.persist) & (ar + pos >= min_prox))
            if len(cand):
                i = int(cand[0])
                j = int(C[i].argmax())
                out.append((pos + i, j % nh, 1 if j < nh else -1))
                pos, min_prox = pos + i + 1, pos + i + self.holdoff
                c, run = np.zeros_like(c), 0
                continue
            c, run = C[-1], int(cont[-1])
            pos += len(C)
        return out

    # ------------------------------------------------------------ calibracao por meta de FA
    def calibrar(self, dfs, h_max=1e4):
        """h tal que a taxa de falso alarme na reamostragem em blocos = meta por semana
        (168 h de calendario; a fracao do tempo em regime vem dos proprios dados)."""
        dfs = [dfs] if isinstance(dfs, pd.DataFrame) else list(dfs)
        Zs = [self._z(d)[0] for d in dfs]
        Z = np.vstack(Zs)
        if len(Z) <= self.bloco:
            # a reamostragem sorteia blocos de bloco_h horas de regime: precisa de mais de um bloco
            raise ValueError(
                f"calibracao curta demais: {len(Z) * self.media_s / 3600:.2f} h em regime (setpoint >= "
                f"{self.v_min:.0f} e estavel); o minimo e mais de {self.bloco * self.media_s / 3600:.2f} h "
                f"(bloco_h). Junte mais trechos normais ou reduza bloco_h.")
        horas_cal = sum(len(d) for d in dfs) / self.fs / 3600
        self.horas_regime_semana = 168 * len(Z) * self.media_s / 3600 / horas_cal
        n_sem = int(self.horas_regime_semana * 3600 / self.media_s)
        rng = np.random.default_rng(self.seed_boot)
        nb = int(np.ceil(n_sem / self.bloco))
        semanas = [np.vstack([Z[i:i + self.bloco] for i in rng.integers(0, len(Z) - self.bloco, nb)])[:n_sem]
                   for _ in range(self.semanas)]

        def taxa(h):
            return sum(len(self._alarmes(S, h)) for S in semanas) / self.semanas

        lo, hi = 0.1, h_max
        while hi / lo > 1.02:
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if taxa(mid) > self.meta else (lo, mid)
        self.h = float(hi)
        self.fa_boot_semana = taxa(self.h)
        return self

    # ------------------------------------------------------------ uso
    def analisar(self, df):
        faltando = [c for c in ["t_s", "v_setpoint"] + [f"{p}_{r}" for r in ROLOS for p in ("v", "tq")]
                    + (["tensao_entrada", "tensao_saida"] if self.usar_tensao else [])
                    if c not in df.columns]
        if faltando:
            raise ValueError(f"colunas ausentes no arquivo: {faltando}")
        Z, idx, R = self._z(df)
        n = len(df)
        stat = np.zeros(n)
        if len(Z):                                                # congelada fora do regime
            full = np.full(n, np.nan)
            full[idx] = self._passo(np.hstack([Z, -Z]), np.zeros(2 * Z.shape[1])).max(axis=1)
            stat = pd.Series(full).ffill().fillna(0.0).to_numpy()
        desvio = _ewma(R - self.mu, 0.3) if len(R) else R         # ~3 blocos, para a mensagem
        t = df["t_s"].to_numpy()
        eventos = []
        for i, j, sinal in (self._alarmes(Z, self.h) if len(Z) else []):
            tipo, rolo = HIPOTESES[j]
            eventos.append({"t_s": float(t[idx[i]]), "indice": int(idx[i]), "indicador": "torque_v2",
                            "tipo": tipo, "rolo": rolo, "sinal": sinal,
                            "desvio_nm": dict(zip(ROLOS, np.round(desvio[i], 2).tolist())),
                            "mensagem": self._mensagem(tipo, rolo, sinal, desvio[i])})
        # isolacao confirmada: maioria (>= 2) dos ultimos 3 alarmes dentro de 90 min
        for n_e, e in enumerate(eventos):
            janela = [x for x in eventos[max(0, n_e - 2): n_e + 1] if e["t_s"] - x["t_s"] <= 5400]
            votos = [(x["tipo"], x["rolo"], x["sinal"]) for x in janela]
            melhor = max(set(votos), key=votos.count)
            e["confirmada"] = melhor if votos.count(melhor) >= 2 else None
        return {"estatistica": stat, "regime": self.portao(df), "limiar": self.h}, eventos

    @staticmethod
    def _mensagem(tipo, rolo, sinal, d):
        txt = ", ".join(f"{r.upper()} {v:+.1f}" for r, v in zip(ROLOS, d))
        if tipo == "encoder":
            leitura = ("a menos (rolo girando mais rápido que o devido)" if sinal > 0 else
                       "a mais (rolo girando mais devagar que o devido)")
            return f"Suspeita no ENCODER do {rolo.upper()}: lendo {leitura}. Desvio de torque (N.m): {txt}"
        lado = "acima" if sinal > 0 else "abaixo"
        return (f"Torque do {rolo.upper()} {lado} do esperado sem efeito nos vizinhos: suspeita "
                f"MECÂNICA (atrito, rolamento, redutor), não encoder. Desvio de torque (N.m): {txt}")

    # ------------------------------------------------------------ exportar
    _PARAMS = ("fs", "k", "h", "media_s", "ordem_ar", "lags_acel", "janela_s", "v_min",
               "estatistica", "lam", "usar_tensao_pontas", "hipoteses_mecanicas")
    _PARAMS_CAL = ("meta_fa_semana", "bloco_h", "semanas_boot", "seed_boot")

    def exportar(self):
        d = {"fs": self.fs, "k": self.k, "h": self.h, "media_s": self.media_s,
             "persist_s": self.persist * self.media_s, "holdoff_min": self.holdoff * self.media_s / 60,
             "ordem_ar": self.ordem_ar, "lags_acel": self.lags, "janela_s": self.janela_s,
             "v_min": self.v_min, "estatistica": self.estatistica, "lam": self.lam,
             "usar_tensao_pontas": self.usar_tensao, "hipoteses_mecanicas": self.mecanicas,
             # parametros da calibracao: sem eles, recalibrar um modelo carregado voltava a blocos de 1 h
             "meta_fa_semana": self.meta, "bloco_h": self.bloco * self.media_s / 3600,
             "semanas_boot": self.semanas, "seed_boot": self.seed_boot}
        d.update({"coef": {r: c.tolist() for r, c in self.coef.items()}, "mu": self.mu.tolist(),
                  "phi": self.phi.tolist(), "sigma_e": self.sigma_e.tolist(), "S": self.S.tolist()})
        return d

    @classmethod
    def carregar(cls, dados):
        if isinstance(dados, str):
            dados = json.loads(dados)
        m = cls(**{k: dados[k] for k in cls._PARAMS + ("persist_s", "holdoff_min")},
                **{k: dados[k] for k in cls._PARAMS_CAL if k in dados})   # JSON antigo: padroes
        m.coef = {r: np.array(c) for r, c in dados["coef"].items()}
        m.mu, m.sigma_e, m.S = np.array(dados["mu"]), np.array(dados["sigma_e"]), np.array(dados["S"])
        m.phi = np.array(dados["phi"], dtype=float).reshape(m.ordem_ar, 3)
        m._montar_filtro()
        return m
