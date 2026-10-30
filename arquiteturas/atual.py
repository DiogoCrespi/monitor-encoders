"""
Adaptadores dos monitores atuais para a interface comum (referencia do benchmark).

    Conjunto  monitor/conjunto.py (Modelos 1 + 2 + 3), como em validacao_conjunta.py: o Modelo 1 e
              retreinado em cada arquivo com o historico normal da mesma planta que vem no log
              (t_s < inicio de df), e o de treino so e usado se o log nao tiver historico.
    Modelo2   monitor/torque_v2.py sozinho, com os parametros do conjunto (PARAM_TORQUE) e a
              tensao das pontas (como dentro do conjunto).
"""
import pandas as pd

from monitor.conjunto import PARAM_TORQUE, MonitorConjunto
from monitor.torque_v2 import MonitorTorqueV2

from .base import Arquitetura

_CAUSA = {"encoder": "encoder", "mecanica": "mecanica", "cabo": "encoder", "sinal": "encoder",
          "ruido": "encoder", "picos": "encoder"}


def _encoders(ev):
    return ev.attrs.get("encoders") if ev is not None else None


class Conjunto(Arquitetura):
    nome = "conjunto_atual"

    def __init__(self, **kw):
        self.kw = kw
        self.mc = MonitorConjunto(**kw)

    def treinar(self, normais, eventos_normais=None, duracao_eventos_s=None):
        self.mc.treinar(normais)
        if eventos_normais is not None and duracao_eventos_s:
            self.mc.treinar_avisos(eventos_normais, duracao_eventos_s, _encoders(eventos_normais),
                                   eventos_normais.attrs.get("contadores"))
        return self

    def calibrar(self, normais, meta_fa_semana=1.0):
        if self.mc.torque is not None:
            self.mc.torque.meta = meta_fa_semana
        self.mc.calibrar(normais)
        return self

    def analisar(self, df, eventos=None):
        t0 = float(df["t_s"].iloc[0])
        dt = float(df["t_s"].iloc[1] - df["t_s"].iloc[0]) if len(df) > 1 else 0.1
        t_fim = float(df["t_s"].iloc[-1]) + dt
        cont = None
        if eventos is not None:
            cont = eventos.attrs.get("contadores")
            hist = eventos[eventos["t_s"] < t0]
            if t0 > 0:                                   # historico normal da propria planta
                hc = None if cont is None else cont[cont["t_s"] < t0]
                self.mc.avisos = None
                self.mc.treinar_avisos(hist, t0, _encoders(eventos), hc)
        saida = self.mc.analisar(df, eventos, cont, t_ini=t0, t_fim=t_fim)
        return [{"t_s": e["t_s"], "rolo": e["rolo"], "causa": _CAUSA.get(e["tipo"]),
                 "severidade": e["severidade"], "modelo": e["modelo"],
                 "mensagem": e["mensagem"]} for e in saida]


class Modelo2(Arquitetura):
    nome = "modelo2_torque"

    def __init__(self, usar_tensao_pontas=True, **kw):
        self.m = MonitorTorqueV2(usar_tensao_pontas=usar_tensao_pontas, **{**PARAM_TORQUE, **kw})

    def treinar(self, normais, eventos_normais=None, duracao_eventos_s=None):
        self.m.treinar(list(normais))
        return self

    def calibrar(self, normais, meta_fa_semana=1.0):
        self.m.meta = meta_fa_semana
        self.m.calibrar(list(normais))
        return self

    def analisar(self, df, eventos=None):
        return [{"t_s": e["t_s"], "rolo": e["rolo"], "causa": e["tipo"],
                 "severidade": "alta" if e.get("confirmada") else "media", "modelo": "torque",
                 "mensagem": e["mensagem"]} for e in self.m.analisar(df)[1]]


__all__ = ["Conjunto", "Modelo2", "pd"]
