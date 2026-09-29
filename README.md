# Monitor de saúde de encoders

Detecção de degradação de encoders de velocidade numa linha de papel, treinada só com operação normal. Não é preciso histórico de falhas.

## Método

Três indicadores sobre o encoder monitorado:

1. Vizinhos: estima a velocidade do encoder a partir dos encoders vizinhos da mesma linha e do setpoint (mínimos quadrados). Indicador = |EWMA do resíduo padronizado|.
2. Próprio sinal: AR(5) com os valores passados do próprio encoder. Indicador = EWMA do |erro de previsão|.
3. Erros de comunicação: erros do drive na última hora.

Linha de corte: 1,1 × o maior valor observado num trecho normal de calibração (contador: máximo + 2). O alarme dispara se o indicador passar do corte por 2 s, e cada indicador tem 10 min de intervalo entre alarmes.

## Dados

- Simulação (`poc/poc_deteccao.py`): 3 encoders a 10 Hz com paradas, rampas e emendas. Falhas injetadas: perda de pulsos, ruído, deriva de ganho e erros de comunicação. A severidade vai de 0 a 100% em 3 h. Os parâmetros são hipóteses, não medições.
- SKAB, bancada de bomba d'água, 1 Hz, 34 experimentos rotulados: https://github.com/waico/SKAB (GPL-3.0)
- PMSM com inversor, bancada, 10 Hz, 8 falhas: https://doi.org/10.5281/zenodo.14482932 (CC-BY-4.0)

## Resultados

Simulação, teste em cenários não usados no desenvolvimento: 48 h normais e 10 cenários por falha.

| Detector | Falsos alarmes / 48 h | Perda de pulsos | Ruído | Deriva | Erros com. |
|---|---|---|---|---|---|
| Referência: só um vizinho | 0 | 10/10, 23 min | 0/10 | 10/10, 79 min | 0/10 |
| Só próprio sinal | 0 | 10/10, 5 min | 10/10, 48 min | 0/10 | 0/10 |
| Monitor completo | 0 | 10/10, 5 min | 10/10, 48 min | 10/10, 10 min | 10/10, 59 min |

Tempo = atraso mediano desde o início da falha. O monitor completo também teve 0 falsos alarmes em 240 h normais.

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
- Zero falso alarme em 240 h equivale a uma taxa < ~0,3/dia (95%), não a uma taxa zero.
- A simulação assume draw constante entre os rolos. Com dado real, o setpoint de tensão e o torque devem entrar como entradas.

## Uso

Requisitos: `numpy pandas scipy scikit-learn pytest tabulate`.

```
python -m pytest -q
python poc/poc_deteccao.py
python validacao.py
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

Entrada para dado real: CSV com `t_s, v_setpoint, v_<encoder>..., erros_<encoder>`.

```python
from monitor import MonitorEncoder
m = MonitorEncoder("pb1", ["festoon", "pb2"], fs=10).treinar(df_normal_1).calibrar(df_normal_2)
indicadores, alarmes = m.analisar(df)
```
