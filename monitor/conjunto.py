"""
Monitor conjunto - os tres modelos v2 num monitor so, com eventos unificados para o tecnico.

    Modelo 1  avisos e contadores do drive (monitor/avisos.py, CUSUM de Poisson por canal)
              -> cabo/conector e sinal do encoder degradando; nao precisa de velocidade
    Modelo 2  torque esperado x real com isolacao (monitor/torque_v2.py, v2.1)
              -> deriva lenta escondida pela malha de velocidade; diz QUAL encoder (ou se e
                 mecanica) pelo padrao de torque nos tres rolos
    Modelo 3  prever o proximo passo v2 (monitor/proximo_passo_v2.py), um por rolo
              -> picos, perda de pulsos e ruido no proprio sinal do encoder

Cada modelo fica ligado so se os dados dele existirem: Modelo 2 precisa de tq_<rolo> dos tres
rolos (e usa tensao_entrada/tensao_saida se houver); Modelo 1 precisa de um log de eventos no
treino; Modelo 3 precisa so de v_setpoint e v_<rolo>.

Cada evento:
    t_s          instante do alarme (mesmo relogio do CSV e do log de eventos)
    modelo       'avisos' | 'torque' | 'proximo_passo'
    encoder      encoder suspeito (None quando o torque aponta causa mecanica)
    rolo         rolo envolvido
    tipo         'cabo' | 'sinal' (avisos), 'encoder' | 'mecanica' (torque), 'ruido' | 'picos'
    severidade   'baixa' | 'media' | 'alta'
    mensagem     texto para o tecnico (o que foi visto e o que verificar)
    corroborado_por  outros modelos que apontaram o mesmo encoder na janela anterior
    origem       o alarme original do modelo (para auditoria)

Severidade: 'media' por padrao; 'alta' quando a isolacao do torque esta confirmada (maioria de
3 alarmes), quando o drive ja registrou falha transitoria ou quando outro modelo apontou o
mesmo encoder nas ultimas `janela_corrob_min`; 'baixa' quando varios encoders alarmam juntos
nos avisos (provavel causa comum: EMC, alimentacao, rede) - nao e defeito de um encoder.
Os modelos continuam independentes: a correlacao so muda a severidade e a mensagem, nunca
apaga nem cria alarme.

A correlacao compara ROLOS. Os Modelos 2 e 3 ja falam em rolo; o Modelo 1 fala no nome do
encoder do log de avisos, traduzido por `mapa_encoders` ({encoder do log: rolo}; o padrao e o
proprio nome). Se nenhum encoder do log casar com um rolo, a corroboracao fica desligada e o
treino dos avisos emite um aviso (warnings).
"""
import json
import warnings

import numpy as np
import pandas as pd

from .avisos import MonitorAvisos
from .proximo_passo_v2 import ProximoPassoV2, de_dataframe
from .torque_v2 import ROLOS, MonitorTorqueV2

MODELOS = ("avisos", "torque", "proximo_passo")
SEVERIDADES = ("baixa", "media", "alta")
NOME_MODELO = {"avisos": "Avisos do drive", "torque": "Torque", "proximo_passo": "Próprio sinal"}

# configuracoes escolhidas no desenvolvimento de cada modelo
PARAM_AVISOS = {"razao": 8.0, "fator_seguranca": 3.0}                       # validacao_avisos.md
PARAM_TORQUE = {"k": 3.0, "media_s": 60, "ordem_ar": 0, "bloco_h": 2.0}      # v2.1, validacao_torque_v2.md
PARAM_PROXIMO = {}                                                          # padrao = configuracao final


class MonitorConjunto:
    def __init__(self, rolos=ROLOS, fs=10, usar_tensao_pontas=None, janela_corrob_min=120.0,
                 param_avisos=None, param_torque=None, param_proximo=None, mapa_encoders=None):
        """usar_tensao_pontas: None = usa se as colunas tensao_entrada/tensao_saida existirem no treino.
        mapa_encoders: {encoder no log de avisos: rolo}; quem nao estiver no mapa usa o proprio nome."""
        self.rolos, self.fs = tuple(rolos), fs
        self.mapa_encoders = dict(mapa_encoders or {})
        self.usar_tensao = usar_tensao_pontas
        self.janela_corrob_s = janela_corrob_min * 60.0
        self.param_avisos = {**PARAM_AVISOS, **(param_avisos or {})}
        self.param_torque = {**PARAM_TORQUE, **(param_torque or {})}
        self.param_proximo = {**PARAM_PROXIMO, **(param_proximo or {})}
        self.avisos, self.torque, self.proximo = None, None, {}

    # ------------------------------------------------------------ dados
    def _vizinhos(self, rolo):
        return [r for r in self.rolos if r != rolo]

    def _tem_torque(self, df):
        return set(self.rolos) == set(ROLOS) and all(f"tq_{r}" in df.columns for r in ROLOS)

    def _serie(self, df, rolo):
        return de_dataframe(df, rolo, self._vizinhos(rolo))

    def rolo_do_encoder(self, encoder):
        return self.mapa_encoders.get(encoder, encoder)

    def _conferir_mapa(self):
        """Avisa quando nenhum encoder do Modelo 1 corresponde a um rolo (corroboracao desligada)."""
        if self.avisos is None:
            return
        encs = sorted({e for e, _ in self.avisos.canais})
        if encs and not {self.rolo_do_encoder(e) for e in encs} & set(self.rolos):
            warnings.warn(f"nenhum encoder do log de avisos {encs} corresponde aos rolos {list(self.rolos)}: "
                          "a corroboracao entre o Modelo 1 e os Modelos 2/3 fica desligada. Informe "
                          "mapa_encoders={encoder: rolo}.", stacklevel=3)

    @property
    def componentes(self):
        return {"avisos": self.avisos is not None, "torque": self.torque is not None,
                "proximo_passo": sorted(self.proximo)}

    # ------------------------------------------------------------ treino e calibracao
    def treinar(self, dfs, eventos=None, duracao_eventos_s=None, encoders=None, contadores=None):
        """dfs: um ou mais trechos NORMAIS no formato do MonitorEncoder (com tq_<rolo> para o Modelo 2).
        eventos/contadores: log NORMAL de avisos que cobre duracao_eventos_s segundos (Modelo 1);
        encoders: {encoder: drive} para o Modelo 1 (ex.: {'pb1': 'tipo_a', 'pb2': 'tipo_b'})."""
        dfs = [dfs] if isinstance(dfs, pd.DataFrame) else list(dfs)
        if all(self._tem_torque(d) for d in dfs):
            tensao = all({"tensao_entrada", "tensao_saida"} <= set(d.columns) for d in dfs)
            usar = tensao if self.usar_tensao is None else bool(self.usar_tensao)
            self.torque = MonitorTorqueV2(fs=self.fs, usar_tensao_pontas=usar, **self.param_torque).treinar(dfs)
        for r in self.rolos:                          # trechos separados: os atrasos nao cruzam a juncao
            self.proximo[r] = ProximoPassoV2(fs=self.fs, **self.param_proximo).treinar_trechos(
                [self._serie(d, r) for d in dfs])
        if eventos is not None:
            self.treinar_avisos(eventos, duracao_eventos_s, encoders, contadores)
        return self

    def treinar_avisos(self, eventos, duracao_s, encoders=None, contadores=None):
        """So o Modelo 1 (o historico de avisos costuma ser bem mais longo que o de sinais)."""
        if duracao_s is None:
            raise ValueError("informe a duracao do historico normal de avisos (s)")
        self.avisos = MonitorAvisos(**self.param_avisos).treinar(eventos, duracao_s, encoders, contadores)
        self._conferir_mapa()
        return self

    def calibrar(self, dfs):
        """Trechos NORMAIS separados do treino: limites dos Modelos 2 e 3 pela meta de falso alarme.
        (O Modelo 1 calibra o limite no proprio treino, pela cadeia de Markov.)"""
        dfs = [dfs] if isinstance(dfs, pd.DataFrame) else list(dfs)
        if self.torque is not None:
            self.torque.calibrar(dfs)
        for r, m in self.proximo.items():
            m.calibrar([self._serie(d, r) for d in dfs])
        return self

    # ------------------------------------------------------------ uso
    def analisar(self, df=None, eventos=None, contadores=None, t_ini=None, t_fim=None):
        """Roda os modelos ligados e devolve a lista de eventos (ordenada no tempo).

        df: CSV de producao (Modelos 2 e 3). eventos/contadores: log de avisos no MESMO relogio
        de df['t_s'] (Modelo 1), analisado em [t_ini, t_fim) - por padrao o intervalo de df."""
        saida = []
        if df is not None:
            t = df["t_s"].to_numpy(float)
            dt = t[1] - t[0] if len(t) > 1 else 1.0 / self.fs
            t_ini = float(t[0]) if t_ini is None else t_ini
            t_fim = float(t[-1] + dt) if t_fim is None else t_fim
            if self.torque is not None and self._tem_torque(df):
                saida += [self._de_torque(e) for e in self.torque.analisar(df)[1]]
            for r, m in self.proximo.items():
                est, idx, alarmes = m.analisar(*self._serie(df, r))
                saida += [self._de_proximo(r, m, est, idx, a, t) for a in alarmes]
        if self.avisos is not None and eventos is not None:
            if t_ini is None or t_fim is None:
                raise ValueError("sem df, informe t_ini e t_fim para os avisos")
            alarmes, _ = self.avisos.analisar(eventos, t_ini, t_fim, contadores)
            saida += [self._de_avisos(a) for a in alarmes]
        saida.sort(key=lambda e: (e["t_s"], MODELOS.index(e["modelo"])))
        return self.correlacionar(saida, self.janela_corrob_s)

    # ------------------------------------------------------------ traducao dos alarmes
    @staticmethod
    def _evento(t_s, modelo, encoder, rolo, tipo, severidade, mensagem, origem):
        return {"t_s": float(t_s), "modelo": modelo, "encoder": encoder, "rolo": rolo, "tipo": tipo,
                "severidade": severidade, "mensagem": f"[{NOME_MODELO[modelo]}] {mensagem}",
                "corroborado_por": [], "origem": origem}

    def _de_torque(self, e):
        conf = e.get("confirmada")
        if e["tipo"] == "encoder":
            confirmada = conf is not None and conf[:2] == ("encoder", e["rolo"])
            acao = (f". Ação: conferir o encoder do {e['rolo'].upper()} (fixação, acoplamento, diâmetro "
                    "efetivo do rolo) e comparar com um tacômetro.")
            return self._evento(e["t_s"], "torque", e["rolo"], e["rolo"], "encoder",
                                "alta" if confirmada else "media",
                                e["mensagem"] + (" Isolação confirmada (maioria de 3 alarmes)." if confirmada else "")
                                + acao, e)
        acao = f". Ação: verificar rolamento, redutor e lubrificação do {e['rolo'].upper()}."
        return self._evento(e["t_s"], "torque", None, e["rolo"], "mecanica", "media", e["mensagem"] + acao, e)

    def _de_proximo(self, rolo, m, est, idx, a, t):
        nome = a["estatistica"]
        pos = int(np.searchsorted(idx, a["indice"]))
        valor = float(est[nome][min(pos, len(est[nome]) - 1)])
        W = m.estatisticas[nome].get("W", 0) / self.fs / 60
        R = rolo.upper()
        if nome == "contagem":
            tipo = "picos"
            lim = m.limiar[nome]                       # alarma com contagem > lim
            normal = "nenhum" if lim < 1 else f"até {lim:.0f}"
            msg = (f"{R}: {valor:.0f} picos no sinal do encoder em {W:.0f} min (normal: {normal}). Quedas "
                   f"curtas de leitura ou perda de pulsos. Ação: verificar conector, cabo e fixação do "
                   f"encoder do {R}.")
        else:
            tipo = "ruido"
            msg = (f"{R}: ruído no sinal do encoder - variância do erro de previsão {valor:.2f}× a normal "
                   f"em {W:.0f} min (limite {m.limiar[nome]:.2f}). Ação: verificar sujeira no disco, cabo, "
                   f"blindagem e aterramento do encoder do {R}.")
        return self._evento(t[a["indice"]], "proximo_passo", rolo, rolo, tipo, "media", msg, dict(a, rolo=rolo))

    def _de_avisos(self, a):
        sev = "media"
        if a.get("causa_comum"):
            sev = "baixa"
        elif "falha(s) transitoria" in a["mensagem"]:
            sev = "alta"
        rolo = self.rolo_do_encoder(a["encoder"])     # nome do log -> rolo (para a corroboracao)
        return self._evento(a["t_s"], "avisos", rolo, rolo, a["familia"], sev, a["mensagem"], a)

    @staticmethod
    def correlacionar(eventos, janela_s):
        """Sobe para 'alta' o evento cujo encoder ja foi apontado por OUTRO modelo nos ultimos janela_s.
        Nao corrobora evento de causa comum (severidade 'baixa') nem causa mecanica (encoder None).
        Os nomes comparados sao os de rolo (eventos do Modelo 1 ja traduzidos por mapa_encoders)."""
        for i, e in enumerate(eventos):
            if e["encoder"] is None or e["severidade"] == "baixa":
                continue
            outros = []
            for a in eventos[:i]:
                if (a["encoder"] == e["encoder"] and a["modelo"] != e["modelo"] and a["severidade"] != "baixa"
                        and e["t_s"] - a["t_s"] <= janela_s and a["modelo"] not in outros):
                    outros.append(a["modelo"])
            if outros:
                e["corroborado_por"] = outros
                e["severidade"] = "alta"
                e["mensagem"] += (" Corroborado por: " + ", ".join(NOME_MODELO[m] for m in outros)
                                  + f" (mesmo encoder nas últimas {janela_s / 3600:.0f} h).")
        return eventos

    # ------------------------------------------------------------ exportar
    def exportar(self):
        return {"rolos": list(self.rolos), "fs": self.fs, "janela_corrob_min": self.janela_corrob_s / 60,
                "mapa_encoders": self.mapa_encoders,
                "avisos": None if self.avisos is None else self.avisos.exportar(),
                "torque": None if self.torque is None else self.torque.exportar(),
                "proximo_passo": {r: json.loads(m.exportar()) for r, m in self.proximo.items()}}

    @classmethod
    def carregar(cls, dados):
        if isinstance(dados, str):
            dados = json.loads(dados)
        m = cls(rolos=dados["rolos"], fs=dados["fs"], janela_corrob_min=dados["janela_corrob_min"],
                mapa_encoders=dados.get("mapa_encoders"))
        m.avisos = None if dados["avisos"] is None else MonitorAvisos.carregar(dados["avisos"])
        m.torque = None if dados["torque"] is None else MonitorTorqueV2.carregar(dados["torque"])
        m.proximo = {r: ProximoPassoV2.carregar(json.dumps(d)) for r, d in dados["proximo_passo"].items()}
        return m
