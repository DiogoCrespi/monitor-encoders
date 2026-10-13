# Monitor de saúde de encoders

Detecção de degradação de encoders de velocidade numa linha de papel, treinada só com operação normal. Não é preciso histórico de falhas.

## Método

Quatro indicadores sobre o encoder monitorado:

1. Vizinhos: estima a velocidade do encoder a partir dos encoders vizinhos da mesma linha e do setpoint (mínimos quadrados). Indicador = |EWMA do resíduo padronizado|.
2. Próprio sinal: AR(5) com os valores passados do próprio encoder. Indicador = EWMA do |erro de previsão|.
3. Erros de comunicação: erros do drive na última hora.
4. Torque (opcional, colunas `tq_<encoder>`): resíduo entre o torque do motor e uma estimativa cinemática (setpoint, velocidade e aceleração do próprio rolo). Indicador = |EWMA do resíduo padronizado|, com memória de ~20 s.

Em malha fechada, o drive corrige a velocidade que o encoder mede. Um erro lento do encoder some da velocidade lida e aparece na tensão do papel e no torque. Os indicadores 1 e 2 só pegam falhas mais rápidas que a malha; para deriva lenta, o indicador é o 4. O torque aponta o vão com tensão alterada, não qual dos dois encoders do vão está errado.

Linha de corte: 1,1 × o maior valor observado num trecho normal de calibração (contador: máximo + 2). O alarme dispara se o indicador passar do corte por 2 s, e cada indicador tem 10 min de intervalo entre alarmes.

## Dados

- Simulação sem malha fechada (`poc/poc_deteccao.py`): 3 encoders a 10 Hz com paradas, rampas e emendas. O erro do encoder aparece direto na velocidade lida. Falhas injetadas: perda de pulsos, ruído, deriva de ganho e erros de comunicação. A severidade vai de 0 a 100% em 3 h.
- Simulação com malha fechada (`poc/malha_fechada.py`): mesmo perfil, com malha de velocidade em cada drive (τ = 0,15 s), tensão nos vãos (EA = 1,2·10⁶ N) e torque de cada motor. Magnitudes de falha menores, para manter a tensão do papel possível (deriva máxima de 0,03%).
- Nas duas simulações, os parâmetros são hipóteses, não medições.
- SKAB, bancada de bomba d'água, 1 Hz, 34 experimentos rotulados: https://github.com/waico/SKAB (GPL-3.0)
- PMSM com inversor, bancada, 10 Hz, 8 falhas: https://doi.org/10.5281/zenodo.14482932 (CC-BY-4.0)

## Resultados

Simulação sem malha fechada, teste em cenários não usados no desenvolvimento: 48 h normais e 10 cenários por falha. Estes resultados não valem para deriva lenta numa máquina real (ver a tabela seguinte).

| Detector | Falsos alarmes / 48 h | Perda de pulsos | Ruído | Deriva | Erros com. |
|---|---|---|---|---|---|
| Referência: só um vizinho | 0 | 10/10, 23 min | 0/10 | 10/10, 79 min | 0/10 |
| Só próprio sinal | 0 | 10/10, 5 min | 10/10, 48 min | 0/10 | 0/10 |
| Monitor completo | 0 | 10/10, 5 min | 10/10, 48 min | 10/10, 10 min | 10/10, 59 min |

Tempo = atraso mediano desde o início da falha. O monitor completo também teve 0 falsos alarmes em 240 h normais.

Simulação com malha fechada, teste em cenários não usados no desenvolvimento: 240 h normais e 10 cenários por falha.

| Detector | Falsos alarmes / 240 h | Perda de pulsos | Ruído | Deriva | Erros com. |
|---|---|---|---|---|---|
| Só vizinhos (velocidade lida) | 0 | 5/10, 161 min | 0/10 | 0/10 | 0/10 |
| Só próprio sinal | 0 | 0/10 | 4/10, 192 min | 0/10 | 0/10 |
| Só torque | 3 | 10/10, 127 min | 10/10, 85 min | 10/10, 56 min | 0/10 |
| Monitor sem torque | 0 | 5/10 | 4/10 | 0/10 | 10/10 |
| Monitor com torque | 3 | 10/10 | 10/10 | 10/10 | 10/10 |

Na deriva (severidade > 90%), a velocidade lida muda +0,06 m/min, a velocidade real +0,23 m/min, a tensão do vão vai de 270 N para 628 N e o torque do rolo de −9,7 para +25,7 N·m. Usar o torque dos vizinhos na estimativa do torque deu 0/40: a falha se confunde com a oscilação normal de tensão do vão.

SKAB, protocolo oficial (treino = 400 primeiras amostras de cada arquivo):

| Detector | F1 | Falso alarme % | Alarme perdido % |
|---|---|---|---|
| Próprio sinal + EWMA | 0,78 | 9,3 | 30,2 |
| Vizinhos (redundância entre sensores) + EWMA | 0,76 | 57,3 | 8,3 |
| Conv-AE (melhor do leaderboard oficial) | 0,78 | 13,55 | 28,02 |

PMSM, treino com metade da operação normal:

| Detector | Falso alarme | Falhas detectadas |
|---|---|---|
| Vizinhos (redundância entre sensores) + EWMA | 0% | 8/8 |
| Próprio sinal + EWMA | 0% | 4/8 |

Comparações adicionais (`comparacoes.py`):
- Isolation Forest nos sinais brutos: 0/40.
- Isolation Forest no resíduo dos vizinhos: 30/40.
- Random Forest treinado com falhas simuladas: 13 a 28 falsos alarmes em 240 h, e detecção mais lenta de falhas não vistas no treino.

## Limitações

- Os dados de encoder são simulados. As bases públicas são reais, mas não têm falha de encoder.
- A simulação em malha fechada é linear e considera os rolos só com controle de velocidade, sem malha de tensão local.
- O torque não distingue qual dos dois encoders de um vão está errado. Para isso é preciso uma referência de velocidade independente.
- Falsos alarmes: 0 em 240 h (sem malha fechada) equivale a uma taxa < ~0,3/dia (95%), não a zero; com malha fechada e torque foram 3 em 240 h (~0,3/dia).
- A simulação assume draw constante entre os rolos. Com dado real, o setpoint de tensão e o torque devem entrar como entradas.

## Uso

Requisitos: `numpy pandas scipy scikit-learn pytest tabulate`.

```
python -m pytest -q
python poc/poc_deteccao.py
python validacao.py
python validacao_malha_fechada.py
python comparacoes.py
python -m monitor.gerar_dashboard
```

Bases públicas, antes de `bases/testar_skab.py` e `bases/testar_pmsm.py`:

```
cd bases
git clone --depth 1 https://github.com/waico/SKAB.git
curl -L -o pmsm.zip https://zenodo.org/api/records/14482932/files/PMSM-inverter-fault-diagnosis-3.0.zip/content
python -c "import zipfile; zipfile.ZipFile('pmsm.zip').extractall('PMSM')"
```

Entrada para dado real: CSV com `t_s, v_setpoint, v_<encoder>..., erros_<encoder>, tq_<encoder>...` (velocidade em m/min, torque em N·m; `erros_` e `tq_` opcionais).

```python
from monitor import MonitorEncoder
m = MonitorEncoder("pb1", ["festoon", "pb2"], fs=10).treinar(df_normal_1).calibrar(df_normal_2)
indicadores, alarmes = m.analisar(df)
```
