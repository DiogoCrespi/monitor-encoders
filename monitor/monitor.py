"""
Monitor de saude de encoder - treina so com operacao normal.

Formato de dados de entrada (uma linha por amostra, uma linha de papel):
    t_s              tempo em segundos
    v_setpoint       setpoint de velocidade da linha (m/min)
    v_<encoder>      velocidade lida de cada encoder (m/min), ex.: v_festoon, v_pb1, v_pb2
    erros_<encoder>  contador ACUMULADO de erros de comunicacao do encoder (opcional),
                     ex.: contadores de erro do barramento do encoder ou bits de alarme
    tq_<encoder>     torque do motor de cada rolo (N.m) (opcional; liga o indicador 'torque')

Indicadores (cada um com sua linha de corte = maior valor visto na calibracao + margem):
    vizinhos  EWMA do residuo entre o encoder e a estimativa feita pelos vizinhos
              (redundancia analitica: todos puxam o mesmo papel) -> deriva, perda de pulsos
    proprio   EWMA do |residuo| de um preditor do proprio sinal (AR)       -> picos, ruido
    erros     erros de comunicacao na ultima hora                          -> cabo/conector
    torque    EWMA do residuo entre o torque do motor e a estimativa cinematica
              (setpoint, aceleracao e velocidade do proprio rolo)          -> deriva em malha fechada

Em malha fechada o drive corrige a velocidade que o encoder MEDE: um encoder que erra
devagar nao aparece na velocidade lida, mas muda a tensao no papel e o torque.
Por isso, com torque disponivel, o indicador 'torque' e o principal para deriva.
O torque dos vizinhos NAO entra na estimativa deste indicador: uma mudanca de tensao num
vao move o torque dos dois rolos do vao em sentidos opostos, e o modelo aprenderia a
explicar a falha como oscilacao normal de tensao. Este indicador olha um rolo so; QUAL
encoder falhou sai do padrao de torque nos TRES rolos (residuos estruturados,
monitor/torque_v2.py): cada encoder errado tem uma assinatura propria, porque a tensao se
propaga para o vao seguinte. Em simulacao, a isolacao acertou 29/30 derivas (30/30 usando
tambem a tensao das pontas).
"""
import json

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import V_RODANDO   # m/min, media dos vizinhos (ver monitor/__init__.py)
LAGS_VIZ = 3
LAGS_AR = 5
LAGS_TQ = 2
INDICADORES = ("vizinhos", "proprio", "torque", "erros")


def validar_formato(df, alvo, vizinhos):
    faltando = [c for c in ["t_s", "v_setpoint", f"v_{alvo}"] + [f"v_{v}" for v in vizinhos]
                if c not in df.columns]
    if faltando:
        raise ValueError(f"colunas ausentes no arquivo: {faltando}")


def _atraso(x, k):
    return np.concatenate([np.full(k, x[0]), x[:-k]])


def _ewma(x, lam):
    return lfilter([lam], [1, -(1 - lam)], x)


def _mq(X, y, alpha=1e-3):
    """Minimos quadrados com regularizacao leve; devolve coeficientes com intercepto no fim."""
    A = np.column_stack([X, np.ones(len(X))])
    return np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y)


def _prever(coef, X):
    return X @ coef[:-1] + coef[-1]


class MonitorEncoder:
    def __init__(self, alvo, vizinhos, fs=10, lam=0.05, persist_s=2.0, holdoff_min=10.0,
                 margem=1.1, lam_tq=0.005):
        self.alvo, self.vizinhos, self.fs = alvo, list(vizinhos), fs
        self.lam, self.lam_tq, self.persist = lam, lam_tq, max(1, int(persist_s * fs))
        self.holdoff, self.margem = int(holdoff_min * 60 * fs), margem

    # ------------------------------------------------------------ atributos
    def _rodando(self, df):
        return df[[f"v_{v}" for v in self.vizinhos]].mean(axis=1).to_numpy() > V_RODANDO

    def _x_viz(self, df):
        cols = [df["v_setpoint"].to_numpy()]
        for v in self.vizinhos:
            x = df[f"v_{v}"].to_numpy()
            cols += [x] + [_atraso(x, k) for k in range(1, LAGS_VIZ + 1)]
        return np.column_stack(cols)

    def _x_ar(self, df):
        y = df[f"v_{self.alvo}"].to_numpy()
        return np.column_stack([_atraso(y, k) for k in range(1, LAGS_AR + 1)])

    def _tem_torque(self, df):
        return all(f"tq_{e}" in df.columns for e in [self.alvo] + self.vizinhos)

    def _x_tq(self, df):
        sp = df["v_setpoint"].to_numpy()
        v = df[f"v_{self.alvo}"].to_numpy()
        acel = np.gradient(v) * self.fs
        cols = [sp, np.gradient(sp) * self.fs, v, acel] + [_atraso(acel, k) for k in range(1, LAGS_TQ + 1)]
        return np.column_stack(cols)

    def _erros_1h(self, df):
        col = f"erros_{self.alvo}"
        if col not in df.columns:
            return np.zeros(len(df))
        c = df[col].to_numpy(float)
        return c - _atraso(c, int(3600 * self.fs))

    # ------------------------------------------------------------ treino
    def treinar(self, df):
        validar_formato(df, self.alvo, self.vizinhos)
        m = self._rodando(df)
        y = df[f"v_{self.alvo}"].to_numpy()
        Xv, Xa = self._x_viz(df), self._x_ar(df)
        self.coef_viz = _mq(Xv[m], y[m])
        self.coef_ar = _mq(Xa[m], y[m])
        self.sigma_viz = float((y[m] - _prever(self.coef_viz, Xv[m])).std())
        self.sigma_ar = float((y[m] - _prever(self.coef_ar, Xa[m])).std())
        self.coef_tq = self.sigma_tq = None
        if self._tem_torque(df):
            Xt, yt = self._x_tq(df), df[f"tq_{self.alvo}"].to_numpy()
            self.coef_tq = _mq(Xt[m], yt[m])
            self.sigma_tq = float((yt[m] - _prever(self.coef_tq, Xt[m])).std())
        return self

    def _indicadores(self, df):
        m = self._rodando(df)
        y = df[f"v_{self.alvo}"].to_numpy()
        est = _prever(self.coef_viz, self._x_viz(df))
        z_viz = np.where(m, (y - est) / self.sigma_viz, 0.0)
        z_ar = np.where(m, np.abs(y - _prever(self.coef_ar, self._x_ar(df))) / self.sigma_ar, 0.0)
        tq = np.zeros(len(df))
        desvio_tq = np.zeros(len(df))
        if self.coef_tq is not None and self._tem_torque(df):
            r = df[f"tq_{self.alvo}"].to_numpy() - _prever(self.coef_tq, self._x_tq(df))
            tq = np.abs(_ewma(np.where(m, r / self.sigma_tq, 0.0), self.lam_tq))
            desvio_tq = _ewma(np.where(m, r, 0.0), self.lam_tq)
        return {
            "rodando": m, "medida": y, "estimada": est,
            "torque": tq, "desvio_tq": desvio_tq,
            "vizinhos": np.abs(_ewma(z_viz, self.lam)),
            "desvio": _ewma(np.where(m, y - est, 0.0), self.lam),
            # linha de base do |z| em operacao normal ~0,8; so interessa o excesso
            "proprio": _ewma(z_ar, self.lam),
            "erros": self._erros_1h(df),
        }

    def calibrar(self, df):
        ind = self._indicadores(df)
        m = ind["rodando"]
        self.limiar = {
            "vizinhos": float(self.margem * ind["vizinhos"][m].max()),
            "proprio": float(self.margem * ind["proprio"][m].max()),
            "torque": float(self.margem * ind["torque"][m].max()) if self.coef_tq is not None else float("inf"),
            "erros": float(ind["erros"].max() + 2),
        }
        return self

    # ------------------------------------------------------------ uso
    def analisar(self, df):
        validar_formato(df, self.alvo, self.vizinhos)
        ind = self._indicadores(df)
        t = df["t_s"].to_numpy()
        eventos = []
        for nome in INDICADORES:
            flag = ind[nome] > self.limiar[nome]
            if nome != "erros":
                flag &= ind["rodando"]
            for i in self._eventos(flag, 1 if nome == "erros" else self.persist):
                eventos.append({"t_s": float(t[i]), "indice": int(i), "indicador": nome,
                                "valor": float(ind[nome][i]), "limiar": self.limiar[nome],
                                "mensagem": self._mensagem(nome, ind, i)})
        eventos.sort(key=lambda e: e["t_s"])
        return ind, eventos

    def _eventos(self, flag, persist):
        cheio = np.convolve(flag.astype(int), np.ones(persist, int), "full")[:len(flag)] >= persist
        out, ultimo = [], -self.holdoff
        for i in np.flatnonzero(cheio):
            if i - ultimo >= self.holdoff:
                out.append(i)
                ultimo = i
        return out

    def _mensagem(self, nome, ind, i):
        a = self.alvo.upper()
        if nome == "vizinhos":
            d = ind["desvio"][i]
            lado = "abaixo" if d < 0 else "acima"
            return f"{a} lendo {abs(d):.1f} m/min {lado} do esperado pelos encoders vizinhos"
        if nome == "proprio":
            return (f"{a}: variação brusca/ruído no próprio sinal "
                    f"(indicador {ind['proprio'][i]:.2f}, linha de corte {self.limiar['proprio']:.2f})")
        if nome == "torque":
            d = ind["desvio_tq"][i]
            lado = "acima" if d > 0 else "abaixo"
            return (f"{a}: torque {abs(d):.1f} N.m {lado} do esperado para a velocidade "
                    f"(tensão no papel alterada; qual encoder: ver o padrão de torque nos três rolos, torque_v2)")
        return f"{a}: {ind['erros'][i]:.0f} erros de comunicação na última hora (normal até {self.limiar['erros'] - 2:.0f})"

    # ------------------------------------------------------------ exportar
    def exportar(self):
        """Coeficientes em JSON - bastam multiplicacoes e somas para rodar em CLP/microcontrolador."""
        return {"alvo": self.alvo, "vizinhos": self.vizinhos, "fs": self.fs, "lam": self.lam,
                "lam_tq": self.lam_tq,
                "persist_s": self.persist / self.fs, "holdoff_min": self.holdoff / 60 / self.fs,
                "margem": self.margem, "coef_viz": self.coef_viz.tolist(),
                "coef_ar": self.coef_ar.tolist(), "sigma_viz": self.sigma_viz,
                "sigma_ar": self.sigma_ar, "limiar": self.limiar,
                "coef_tq": None if self.coef_tq is None else self.coef_tq.tolist(),
                "sigma_tq": self.sigma_tq}

    @classmethod
    def carregar(cls, dados):
        if isinstance(dados, str):
            dados = json.loads(dados)
        m = cls(dados["alvo"], dados["vizinhos"], dados["fs"], dados["lam"], dados["persist_s"],
                dados["holdoff_min"], dados["margem"], dados.get("lam_tq", dados["lam"]))
        m.coef_viz, m.coef_ar = np.array(dados["coef_viz"]), np.array(dados["coef_ar"])
        m.sigma_viz, m.sigma_ar, m.limiar = dados["sigma_viz"], dados["sigma_ar"], dados["limiar"]
        m.coef_tq = None if dados.get("coef_tq") is None else np.array(dados["coef_tq"])
        m.sigma_tq = dados.get("sigma_tq")
        return m


def de_simulacao(df_sim):
    """Converte a saida de poc.poc_deteccao.simular para o formato de dados do monitor."""
    out = pd.DataFrame({"t_s": df_sim["t_s"], "v_setpoint": df_sim["setpoint"]})
    for e in [r for r in ("festoon", "pb1", "pb2") if r in df_sim.columns]:
        out[f"v_{e}"] = df_sim[e]
        if f"{e}_erros_com" in df_sim.columns:            # contador acumulado de cada rolo que o tiver
            out[f"erros_{e}"] = df_sim[f"{e}_erros_com"].cumsum()
    return out
