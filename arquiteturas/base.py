"""
Interface comum das arquiteturas candidatas (usada por benchmark/avaliar.py).

Uma arquitetura e treinada UMA vez por planta (conjunto de treino) e depois analisa cada arquivo
de forma independente: o avaliador faz copy.deepcopy da arquitetura treinada antes de cada
`analisar`, entao adaptacao online e permitida dentro de um arquivo, mas nao vaza entre arquivos.

DADOS (formato do simulador, poc/malha_fechada_v2.py, 10 Hz):
    t_s, v_setpoint (m/min), v_<rolo> (m/min, lida pelo drive), tq_<rolo> (N.m),
    tensao_entrada, tensao_saida (N, pontas), receita (id da receita em vigor, inteiro),
    draw_pb1, draw_pb2 (setpoint de draw, fracao), trim_pb2 (so com malha de tensao no pb2).
    rolos = 'festoon', 'pb1', 'pb2'. As colunas de verdade de campo do simulador (v_real_*,
    tensao_vao1/2, severidade) sao REMOVIDAS antes de chegar na arquitetura.
    Nos arquivos analisados, t_s esta no relogio do log de avisos (comeca em 14 dias = 1 209 600 s);
    nos de treino/calibracao comeca em 0. Use diferencas de tempo, nao o valor absoluto.

LOG DE AVISOS (Modelo 1; colunas t_s, drive, encoder, codigo, tipo, valor):
    treinar(..., eventos_normais, duracao_eventos_s): historico NORMAL de duracao_eventos_s
        segundos da planta de treino.
    analisar(df, eventos): o log cobre [0, fim de df). A parte com t_s < df['t_s'].iloc[0] e
        historico NORMAL da MESMA planta (14 dias) - pode ser usada para (re)treinar taxas por
        encoder, como faz validacao_conjunta.py. So conta alarme em [inicio de df, fim de df).
    eventos.attrs: 'contadores' (DataFrame t_s, encoder, contador, valor: contador de erros acumulado a cada
        60 s), 'encoders' ({encoder: drive}), 't_ini_s', 't_fim_s' (intervalo de df).
    eventos pode ser None (arquivo sem log).

SAIDA de analisar: lista de dicts, um por alarme, com pelo menos
    t_s        float, instante do alarme no relogio de df
    rolo       'festoon' | 'pb1' | 'pb2' | None (nao sabe)
    causa      'encoder' (encoder, cabo ou sinal do encoder) | 'mecanica' (atrito, rolamento,
               redutor) | 'processo' (mudanca de planta/receita, nao e defeito) | None
    severidade 'baixa' | 'media' | 'alta' | None
    modelo     nome do sub-modelo que gerou o alarme (texto livre)
    (outras chaves sao livres: mensagem, origem, ...)
Todo alarme conta: em arquivo normal, qualquer alarme e falso alarme (inclusive causa='processo').
"""
import pandas as pd

CAUSAS = ("encoder", "mecanica", "processo", None)
ROLOS = ("festoon", "pb1", "pb2")


class Arquitetura:
    nome: str = "base"

    def treinar(self, normais: "list[pd.DataFrame]", eventos_normais: "pd.DataFrame | None" = None,
                duracao_eventos_s: "float | None" = None) -> "Arquitetura":
        raise NotImplementedError

    def calibrar(self, normais: "list[pd.DataFrame]", meta_fa_semana: float = 1.0) -> "Arquitetura":
        raise NotImplementedError

    def analisar(self, df: pd.DataFrame, eventos: "pd.DataFrame | None" = None) -> "list[dict]":
        raise NotImplementedError

    def __repr__(self):
        return f"<{type(self).__name__} {self.nome}>"


def alarme(t_s, rolo=None, causa=None, severidade=None, modelo="", **extra):
    """Atalho para montar um alarme no formato da interface."""
    if causa not in CAUSAS:
        raise ValueError(f"causa invalida: {causa}")
    return {"t_s": float(t_s), "rolo": rolo, "causa": causa, "severidade": severidade, "modelo": modelo, **extra}
