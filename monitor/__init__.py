"""
Pacote do monitor de encoders (v1 em monitor.py; modelos v2 em avisos.py, torque_v2.py,
proximo_passo_v2.py, reunidos em conjunto.py).

Limiares de "linha rodando" (m/min). Sao TRES de proposito, cada um com um papel diferente;
unificar mudaria os resultados ja validados (validacao_*.md), entao ficam separados e so aqui:

    V_RODANDO = 50       velocidade LIDA, media dos encoders vizinhos (monitor.py e
                         proximo_passo_v2.de_dataframe). Separa linha parada/arrastando de linha
                         em movimento. Usa os vizinhos, e nao o proprio encoder, para que uma
                         falha no encoder monitorado nao desligue o proprio monitoramento.
    SP_AQUECIMENTO = 30  SETPOINT (comando), so no Modelo 2 (torque_v2._x): marca "motor girando"
                         para as entradas F15/F45 de aquecimento do redutor. Qualquer movimento
                         aquece, entao o corte e baixo; nao decide quando monitorar.
    V_MIN_REGIME = 200   SETPOINT, portao de regime do Modelo 2 (torque_v2.portao, parametro
                         v_min). Abaixo disso o torque e dominado por atrito seco e transitorio de
                         tensao e a regressao do torque nao vale; e o mais alto dos tres porque o
                         Modelo 2 precisa de producao estavel, nao so de linha andando.
"""
# definidos ANTES dos imports abaixo: os modulos do pacote importam estas constantes
V_RODANDO = 50.0
SP_AQUECIMENTO = 30.0
V_MIN_REGIME = 200.0

from .conjunto import MonitorConjunto  # noqa: E402
from .monitor import MonitorEncoder, de_simulacao, validar_formato  # noqa: E402

__all__ = ["MonitorEncoder", "de_simulacao", "validar_formato", "MonitorConjunto"]
