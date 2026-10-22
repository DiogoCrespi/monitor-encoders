# Monitor de saúde de encoders

Detecção de degradação de encoders de velocidade numa linha de papel, treinada só com operação normal. Não é preciso histórico de falhas.

Relatório técnico completo (ABNT): `Relatorio-tecnico-ABNT-monitor-encoders.pdf`.

## Método

Quatro indicadores sobre o encoder monitorado:

1. Vizinhos: estima a velocidade do encoder a partir dos encoders vizinhos da mesma linha e do setpoint (mínimos quadrados). Indicador = |EWMA do resíduo padronizado|.
2. Próprio sinal: AR(5) com os valores passados do próprio encoder. Indicador = EWMA do |erro de previsão|.
3. Erros de comunicação: erros do drive na última hora.
4. Torque (opcional, colunas `tq_<encoder>`): resíduo entre o torque do motor e uma estimativa cinemática (setpoint, velocidade e aceleração do próprio rolo). Indicador = |EWMA do resíduo padronizado|, com memória de ~20 s.

Em malha fechada, o drive corrige a velocidade que o encoder mede. Um erro lento do encoder some da velocidade lida e aparece na tensão do papel e no torque. Os indicadores 1 e 2 só pegam falhas mais rápidas que a malha; para deriva lenta, o indicador é o 4. O indicador 4 olha um rolo só; qual encoder falhou sai do padrão de torque nos três rolos (`monitor/torque_v2.py`, abaixo).

Linha de corte: 1,1 × o maior valor observado num trecho normal de calibração (contador: máximo + 2). O alarme dispara se o indicador passar do corte por 2 s, e cada indicador tem 10 min de intervalo entre alarmes.

## Modelos v2 e monitor conjunto

Três modelos novos, em módulos separados (o `monitor/monitor.py` original continua igual), reunidos em `monitor/conjunto.py`:

1. Avisos e contadores do drive (`monitor/avisos.py`): cada aviso ou falha do histórico do drive vira um incidente de um canal (encoder, família `cabo` ou `sinal`). CUSUM de Poisson por canal, com a taxa normal estimada no histórico e a linha de corte pela meta de 1 falso alarme por semana na linha inteira (ARL por cadeia de Markov). Os códigos são genéricos (`aviso_cabo`, `aviso_sinal`, `falha_cabo`, `contador_erros`...); para dado real, `configurar_codigos()` troca a tabela pelos códigos do fabricante.
2. Torque v2.1 com isolação (`monitor/torque_v2.py`): regressão com dinâmica por rolo, portão de regime (suspende em parada, rampa e emenda), médias de 1 min e resíduos estruturados. Um encoder lendo a menos muda os três torques com uma assinatura própria: Festoon (+1, 0, −1), P&B1 (−1, +1, 0), P&B2 (0, −1, +1); causa mecânica muda só o torque do próprio rolo. CUSUM bilateral em cada hipótese, limite pela meta de falso alarme via reamostragem em blocos. Usa a tensão das pontas (`tensao_entrada`, `tensao_saida`) se houver.
3. Próximo passo v2 (`monitor/proximo_passo_v2.py`): ARX com o setpoint (ordem por BIC) e branqueamento do resíduo; alarme por variância do resíduo em 5 min ou contagem de picos em 10 min, com limite pela meta de falso alarme.

`MonitorConjunto` liga cada modelo conforme os dados disponíveis e emite eventos com o modelo que disparou, o encoder suspeito, a mensagem para o técnico e a severidade (`alta` com isolação confirmada ou com dois modelos apontando o mesmo encoder em 2 h; `baixa` quando vários encoders avisam juntos, provável causa comum).

Simulador v2 (`poc/malha_fechada_v2.py`): malha de velocidade de 10 ms, atraso da tensão L/v, EA = 2·10⁶ N, tensão de 700–900 N/m, deriva de 0,005 %, falha injetável em qualquer rolo, atrito e aquecimento do redutor variando, opção de malha de tensão no P&B2. Avisos sintéticos em `poc/simulador_avisos.py`, com acoplamento aos cenários do simulador v2 (`simular_acoplado`). Todos os parâmetros são hipóteses.

Resultados de cada modelo no seu próprio teste (sementes não usadas no desenvolvimento):

- Avisos: 10/10 degradações detectadas, antecedência mediana de 29 h antes da falha final, 0,60 falso alarme por semana numa linha de 10 encoders. A regra "máximo + 2 na última hora" deu 5/10 e 1,5 por semana.
- Torque v2.1: deriva 10/10 nos três rolos (25–42 min), 1 falso alarme em 240 h. Isolação confirmada em 29/30 derivas só com torque e setpoint, 30/30 com a tensão das pontas.
- Próximo passo v2, simulador com malha lenta: perda de pulsos e ruído 10/10 (7 e 14 min), 0 falso alarme em 240 h.

Validação conjunta (`validacao_conjunta.py`): os três modelos juntos no simulador v2, com avisos acoplados. Teste rodado uma vez em sementes novas: falhas 211–220 (12 casos × 10 cenários de 6 h, falha a partir de 1 h, rampa de 4 h), normais 191–195 (5 × 48 h).

| Caso | Detecção | Primeiro a disparar | Isolação no 1º evento |
|---|---|---|---|
| Deriva Festoon / P&B1 / P&B2 | 10/10, 35 / 27 / 31 min | torque | 30/30 |
| Atrito Festoon / P&B1 / P&B2 | 10/10, 91 / 63 / 78 min | torque | 27/30 (confirmada 30/30) |
| Perda de pulsos P&B1 / P&B2 | 10/10, 143 / 139 min | avisos 9, torque 11 | 18/20 |
| Ruído P&B1 / P&B2 | 9/10, 147 / 158 min | avisos | 18/18 |
| Erros de comunicação P&B1 | 10/10, 163 min | avisos | 10/10 |
| Degradação de sinal Festoon | 9/10, 232 min | avisos | 9/9 |

Falsos alarmes do conjunto: 0 em 240 h normais e 0 na primeira hora dos 120 cenários (360 h; a 95 %, menos de ~1,4 por semana). O modelo de avisos sozinho, em 20 semanas de log normal da mesma linha, deu 0,45 por semana. Total: 117/120 cenários detectados; o primeiro evento apontou o encoder (ou a causa mecânica) certo em 112/117.

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
- Um indicador de torque por rolo não diz qual encoder falhou; o padrão nos três rolos diz (29/30 em simulação, 30/30 com a tensão das pontas). Exige torque dos três rolos e hipótese de falha única.
- Na validação conjunta, ruído, erros de comunicação e degradação de sinal só foram detectados porque a simulação supõe que o drive emite avisos (2 incidentes por hora na severidade máxima). Sem aviso, ruído fica 0/10 no simulador v2.
- O próximo passo v2 não disparou em nenhum cenário do simulador v2: a malha de 10 ms corrige o erro antes da amostra de 10 Hz e a velocidade lida fica igual à referência. Ele precisa de velocidade registrada mais rápido que a malha, ou de malha mais lenta.
- Rolo com malha de tensão local: a deriva do encoder vai para a correção de velocidade, não para o torque (1/10 no P&B2 nesse modo).
- Falsos alarmes: 0 em 240 h (sem malha fechada) equivale a uma taxa < ~0,3/dia (95%), não a zero; com malha fechada e torque foram 3 em 240 h (~0,3/dia).
- A simulação assume draw constante entre os rolos. Com dado real, o setpoint de tensão e o torque devem entrar como entradas.

## Uso

Requisitos: Python 3 com `numpy pandas scipy scikit-learn pytest tabulate` (versões mínimas em `requirements.txt`):

```
pip install -r requirements.txt
```

Os scripts `validacao*.py` e `comparacoes.py` gravam arquivos `.md` de resultado; para só conferir o código, use o pytest.

```
python -m pytest -q                       # 80 testes (~2 min)
python poc/poc_deteccao.py
python validacao.py
python validacao_malha_fechada.py
python comparacoes.py
python validacao_avisos.py
python validacao_torque_v2.py
python validacao_proximo_passo_v2.py teste
python validacao_conjunta.py teste
python -m monitor.gerar_dashboard         # dashboard do monitor conjunto (~2 min)
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

Monitor conjunto, com o histórico de avisos do drive (`t_s, encoder, codigo, tipo`) no mesmo relógio do CSV:

```python
from monitor import MonitorConjunto
mc = MonitorConjunto().treinar([df_normal_1, df_normal_2], eventos=avisos_normais,
                               duracao_eventos_s=14 * 86400,
                               encoders={"festoon": "tipo_a", "pb1": "tipo_a", "pb2": "tipo_b"})
mc.calibrar([df_normal_3, df_normal_4])
for e in mc.analisar(df, eventos=avisos):
    print(e["t_s"], e["modelo"], e["encoder"], e["severidade"], e["mensagem"])
```

A corroboração entre modelos compara rolos. Se o log de avisos usa outros nomes de encoder, passe `MonitorConjunto(mapa_encoders={"eixo_3": "pb1", ...})`; sem nenhum nome casando, o treino dos avisos emite um aviso.

Dashboard (`dashboard/monitor-encoders.html`, gerado por `monitor/gerar_dashboard.py`): o monitor conjunto com os três modelos v2 no simulador v2 com avisos acoplados, alarmes com severidade e isolação. Abre offline no navegador.
