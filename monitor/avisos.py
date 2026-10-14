"""
Modelo 1 - avisos e contadores do proprio drive.

Os drives costumam avisar antes de falhar. Dois tipos de drive (nomes genericos):
    tipo_a   aviso_cabo a cada incremento de contador_erros (erro no cabo do encoder) -> falha_cabo
             aviso_sinal / aviso_encoder (amplitude, iluminacao)                    -> falha_sinal
    tipo_b   aviso_sinal_b (sinais do encoder perturbados)                          -> falha_sinal_b
Os codigos reais de cada fabricante entram pela configuracao CODIGOS (codigo -> familia), que
pode ser trocada com configurar_codigos(). Este modulo transforma o historico desses avisos num alarme com base estatistica.

ENTRADA 1 - LOG DE EVENTOS (uma linha por aviso/falha registrado):
    t_s      tempo em segundos (horimetro ou carimbo do historico do drive, ou relogio
             do coletor)
    encoder  identificacao do encoder/eixo, ex.: 'rewinder', 'pb1'
    codigo   codigo da mensagem, ex.: 'aviso_cabo', 'aviso_sinal', 'aviso_sinal_b', 'falha_sinal_b'
    tipo     'aviso' ou 'falha'
    drive    (opcional) 'tipo_a' ou 'tipo_b'; usado para agrupar encoders parecidos
    valor    (opcional) valor da mensagem: n. do componente (aviso_cabo), amplitude em mV (aviso_sinal)...

ENTRADA 2 - CONTADORES amostrados (opcional; uma linha por leitura):
    t_s, encoder, contador ('contador_erros' ou 'contador_erros_2'), valor (ACUMULADO; uma queda = contador zerado)

METODO
    1. Cada mensagem vira um "incidente" de um CANAL = (encoder, familia). Familias:
       'cabo' (comunicacao) e 'sinal' (amplitude/iluminacao/sinal perturbado). Mensagens a
       menos de `declust_s` uma da outra no mesmo canal contam como 1 incidente: uma rajada
       de EMC, o aviso_cabo repetido e o incremento de contador_erros do mesmo evento nao viram 10 sinais.
    2. Taxa normal lambda0 de cada canal estimada no historico normal, encolhida para a taxa
       media dos encoders do mesmo drive (com poucos eventos, a taxa de 1 encoder e ruidosa).
    3. CUSUM de Poisson (Page 1954; Lucas 1985) nas contagens por intervalo de `bin_min`:
           S = max(0, S + x - k),   k = mu0 (r - 1) / ln r,   mu0 = lambda0 * bin
       otima para detectar a taxa passar de lambda0 para r*lambda0.
    4. Linha de corte h escolhida para a META de falso alarme em tempo: o tempo medio ate
       um falso alarme (ARL0) da linha inteira = 1 semana / meta. Com n canais, cada canal
       recebe ARL0 = n semanas / meta. O ARL0 e calculado por cadeia de Markov (Brook &
       Evans 1972) com a distribuicao de Poisson de mu0 - nao pelo maximo de um trecho.
    5. Alarme quando S > h; repete no mesmo canal so depois de `holdoff_h`. A mensagem diz
       qual encoder, qual aviso, quantos incidentes nas ultimas 24 h e o normal.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import poisson

SEMANA_S = 7 * 86400.0
FAMILIAS = ("cabo", "sinal")
# familias que cada drive PODE emitir, mesmo sem nenhum caso no historico normal
FAMILIAS_DRIVE = {"tipo_a": ("cabo", "sinal"), "tipo_b": ("sinal",)}

DESCRICAO = {
    "aviso_cabo": "erro no cabo do encoder (contador de erros incrementou)",
    "contador_erros": "contador de erros de comunicacao do encoder incrementou",
    "contador_erros_2": "segundo contador de erros de comunicacao incrementou",
    "falha_cabo": "falha de comunicacao com o encoder",
    "aviso_encoder": "encoder sinaliza alarme interno (amplitude/iluminacao/temperatura)",
    "aviso_sinal": "amplitude do sinal fora da tolerancia",
    "falha_sinal": "nivel de sinal muito baixo",
    "falha_encoder": "encoder sinaliza erro interno",
    "aviso_sinal_b": "sinais do encoder perturbados",
    "falha_sinal_b": "sinais do encoder incorretos",
}
# codigo -> familia. Para dado real, troque pelos codigos do fabricante com configurar_codigos().
CODIGOS = {"aviso_cabo": "cabo", "contador_erros": "cabo", "contador_erros_2": "cabo", "falha_cabo": "cabo",
           "aviso_encoder": "sinal", "aviso_sinal": "sinal", "falha_sinal": "sinal", "falha_encoder": "sinal",
           "aviso_sinal_b": "sinal", "falha_sinal_b": "sinal"}
CONTADORES = ("contador_erros", "contador_erros_2")    # nomes de contador (nao entram na mensagem)
AMPLITUDE = ("aviso_sinal",)                          # avisos cujo valor e a amplitude em mV
ACAO = {"cabo": "verificar cabo, conector e blindagem do encoder deste eixo",
        "sinal": "verificar sujeira/umidade no encoder e amplitude do sinal; planejar a troca"}


def configurar_codigos(codigos, descricao=None):
    """Troca a tabela codigo -> familia ('cabo' ou 'sinal') pela do fabricante usado.
    descricao (opcional): codigo -> texto para a mensagem do tecnico."""
    invalidas = set(codigos.values()) - set(FAMILIAS)
    if invalidas:
        raise ValueError(f"familia deve ser 'cabo' ou 'sinal'; encontrado {sorted(invalidas)}")
    CODIGOS.clear()
    CODIGOS.update(codigos)
    if descricao:
        DESCRICAO.update(descricao)


def familia(codigo):
    """Familia de um codigo de mensagem ('cabo', 'sinal') ou None se nao for de encoder."""
    return CODIGOS.get(str(codigo).strip())


def validar_eventos(df):
    faltando = [c for c in ("t_s", "encoder", "codigo", "tipo") if c not in df.columns]
    if faltando:
        raise ValueError(f"colunas ausentes no log de eventos: {faltando}")
    tipos = set(df["tipo"].dropna().unique()) - {"aviso", "falha"}
    if tipos:
        raise ValueError(f"tipo deve ser 'aviso' ou 'falha'; encontrado {sorted(tipos)}")


def contadores_para_eventos(cont, declust_s=60.0):
    """Converte leituras de contador ACUMULADO em eventos (codigo = nome do contador).

    Incremento d entre duas leituras separadas por dt vira min(d, max(1, dt // declust_s))
    eventos espalhados no intervalo: com amostragem lenta nao da para saber se foram
    d incidentes ou 1 rajada, entao contamos no maximo 1 por janela de agrupamento.
    Valor que cai = contador zerado: o incremento e o proprio valor lido.
    """
    if cont is None or len(cont) == 0:
        return pd.DataFrame(columns=["t_s", "encoder", "codigo", "tipo", "valor"])
    faltando = [c for c in ("t_s", "encoder", "contador", "valor") if c not in cont.columns]
    if faltando:
        raise ValueError(f"colunas ausentes nos contadores: {faltando}")
    linhas = []
    for (enc, nome), g in cont.sort_values("t_s").groupby(["encoder", "contador"], sort=False):
        t = g["t_s"].to_numpy(float)
        v = g["valor"].to_numpy(float)
        d = np.diff(v)
        d = np.where(d < 0, v[1:], d)
        dt = np.diff(t)
        for i in np.flatnonzero(d > 0):
            n = int(min(d[i], max(1, dt[i] // declust_s)))
            for tt in t[i] + dt[i] * (np.arange(n) + 1) / n:
                linhas.append((float(tt), enc, str(nome), "aviso", float(d[i])))
    return pd.DataFrame(linhas, columns=["t_s", "encoder", "codigo", "tipo", "valor"])


def _juntar(eventos, contadores, declust_s):
    validar_eventos(eventos)
    ev = eventos.copy()
    if "drive" not in ev.columns:
        ev["drive"] = "todos"
    if "valor" not in ev.columns:
        ev["valor"] = np.nan
    if contadores is not None and len(contadores):
        c = contadores_para_eventos(contadores, declust_s)
        drv = ev.drop_duplicates("encoder").set_index("encoder")["drive"]
        c["drive"] = c["encoder"].map(drv).fillna("todos")
        ev = pd.concat([ev, c], ignore_index=True)
    ev["familia"] = ev["codigo"].map(familia)
    return ev[ev["familia"].notna()].sort_values("t_s", kind="stable").reset_index(drop=True)


def _incidentes(t, declust_s):
    """Agrupa mensagens proximas (ligacao simples): devolve o indice da 1a de cada incidente."""
    if len(t) == 0:
        return np.array([], int)
    novo = np.concatenate([[True], np.diff(t) > declust_s])
    return np.flatnonzero(novo)


# ------------------------------------------------------------------ ARL por cadeia de Markov
def arl_cusum_poisson(mu0, k, h, mu=None, n_grade=300):
    """Tempo medio ate o alarme (em intervalos) da CUSUM de Poisson S=max(0,S+x-k), alarme S>h.

    Cadeia de Markov (Brook & Evans 1972) numa grade de [0, h], com arredondamento
    estocastico para a grade (preserva a deriva media -k mesmo com k << passo da grade).
    mu = media verdadeira por intervalo (default mu0, ou seja, ARL0).
    """
    mu = mu0 if mu is None else mu
    g = h / n_grade
    s = np.arange(n_grade + 1) * g
    xmax = int(poisson.ppf(1 - 1e-12, mu)) + 2
    px = poisson.pmf(np.arange(xmax + 1), mu)
    Q = np.zeros((n_grade + 1, n_grade + 1))
    for x, p in enumerate(px):
        sn = s + x - k
        dentro = sn <= h
        sn = np.clip(sn, 0, None)
        pos = sn / g
        lo = np.floor(pos).astype(int)
        fr = pos - lo
        for i in np.flatnonzero(dentro):
            Q[i, lo[i]] += p * (1 - fr[i])
            if fr[i] > 0:
                Q[i, min(lo[i] + 1, n_grade)] += p * fr[i]
    arl = np.linalg.solve(np.eye(n_grade + 1) - Q, np.ones(n_grade + 1))
    return float(arl[0])


def limiar_para_arl(mu0, k, arl_alvo, h_max=60.0):
    """Menor h com ARL0 >= arl_alvo (bissecao; o ARL cresce com h)."""
    lo, hi = 0.0, 1.0
    while arl_cusum_poisson(mu0, k, hi) < arl_alvo:
        lo, hi = hi, hi * 2
        if hi > h_max:
            return h_max
    for _ in range(30):
        mid = (lo + hi) / 2
        if arl_cusum_poisson(mu0, k, mid) >= arl_alvo:
            hi = mid
        else:
            lo = mid
    return hi


# ------------------------------------------------------------------ monitor
class MonitorAvisos:
    def __init__(self, bin_min=10.0, declust_s=60.0, meta_fa_semana=1.0, razao=4.0,
                 holdoff_h=12.0, prior_h=168.0, fator_seguranca=1.0):
        self.bin_s = bin_min * 60.0
        self.declust_s = declust_s
        self.meta = meta_fa_semana
        self.razao = razao
        self.holdoff_s = holdoff_h * 3600.0
        self.prior_h = prior_h
        self.fator_seguranca = fator_seguranca
        self.canais = {}

    # -------------------------------------------------------- treino (Fase I)
    def treinar(self, eventos, duracao_s, encoders=None, contadores=None):
        """eventos/contadores de um periodo NORMAL de duracao_s segundos (sem falha conhecida).

        encoders: dict {encoder: drive} ou lista; default = os que aparecem no log.
        Cria um canal (encoder, familia) para cada familia que o drive pode emitir
        (FAMILIAS_DRIVE) ou que algum encoder do mesmo drive ja apresentou no historico.
        Um tipo_b (sem contador de erros de comunicacao) nao ganha canal 'cabo'.
        """
        ev = _juntar(eventos, contadores, self.declust_s)
        if encoders is None:
            encoders = ev.drop_duplicates("encoder").set_index("encoder")["drive"].to_dict()
        elif not isinstance(encoders, dict):
            drv = ev.drop_duplicates("encoder").set_index("encoder")["drive"]
            encoders = {e: drv.get(e, "todos") for e in encoders}
        horas = duracao_s / 3600.0
        n_inc = {}
        for enc, drive in encoders.items():
            for fam in FAMILIAS:
                t = ev.loc[(ev["encoder"] == enc) & (ev["familia"] == fam), "t_s"].to_numpy()
                n_inc[(enc, fam)] = len(_incidentes(t, self.declust_s))
        # familias por drive (conhecidas + vistas no historico) e taxa media por (drive, familia)
        canais = {}
        for drive in set(encoders.values()):
            encs = [e for e, d in encoders.items() if d == drive]
            for fam in FAMILIAS:
                total = sum(n_inc[(e, fam)] for e in encs)
                if total == 0 and fam not in FAMILIAS_DRIVE.get(drive, ()):
                    continue
                # sem nenhum caso no grupo: assume 1 incidente no historico (cota pequena, mas um
                # aviso isolado nunca visto nao basta para alarmar; dois proximos, sim)
                pool = max(total, 1) / (horas * len(encs))
                for e in encs:
                    lam = (n_inc[(e, fam)] + self.prior_h * pool) / (horas + self.prior_h)
                    canais[(e, fam)] = {"drive": drive, "lambda0_h": float(lam),
                                        "n_treino": int(n_inc[(e, fam)])}
        if not canais:
            raise ValueError("sem canais: informe o drive ('tipo_a'/'tipo_b') ou um historico com avisos")
        arl_alvo_s = len(canais) * SEMANA_S / self.meta * self.fator_seguranca
        cache = {}
        for c in canais.values():
            mu0 = c["lambda0_h"] * self.bin_s / 3600.0
            k = mu0 * (self.razao - 1) / np.log(self.razao)
            chave = round(mu0, 10)
            if chave not in cache:
                cache[chave] = limiar_para_arl(mu0, k, arl_alvo_s / self.bin_s)
            c.update(mu0=float(mu0), k=float(k), h=float(cache[chave]))
        self.canais = canais
        self.horas_treino = horas
        return self

    # -------------------------------------------------------- uso (Fase II)
    def analisar(self, eventos, t_ini, t_fim, contadores=None):
        """Roda a CUSUM de cada canal em [t_ini, t_fim). Devolve (alarmes, trajetorias).

        trajetorias[(encoder, familia)] = DataFrame t_s (fim de cada intervalo), S.
        """
        ev = _juntar(eventos, contadores, self.declust_s)
        ev = ev[(ev["t_s"] >= t_ini) & (ev["t_s"] < t_fim)]
        nb = int(np.ceil((t_fim - t_ini) / self.bin_s))
        alarmes, traj = [], {}
        for (enc, fam), c in self.canais.items():
            sub = ev[(ev["encoder"] == enc) & (ev["familia"] == fam)].reset_index(drop=True)
            t = sub["t_s"].to_numpy()
            idx = _incidentes(t, self.declust_s)
            ti = t[idx]
            b = ((ti - t_ini) // self.bin_s).astype(int)
            S, ultimo, Ss = 0.0, -np.inf, np.zeros(nb)
            j0 = 0
            for i in range(nb):
                j1 = j0
                while j1 < len(b) and b[j1] == i:
                    j1 += 1
                for m, j in enumerate(range(j0, j1), start=1):
                    if S + m - c["k"] > c["h"] and ti[j] - ultimo >= self.holdoff_s:
                        alarmes.append(self._alarme(enc, fam, c, sub, ti[j], S + m - c["k"]))
                        ultimo = ti[j]
                        S = -(m - c["k"])  # zera: o que ja contou neste intervalo sai
                S = max(0.0, S + (j1 - j0) - c["k"])
                Ss[i] = S
                j0 = j1
            traj[(enc, fam)] = pd.DataFrame({"t_s": t_ini + (np.arange(nb) + 1) * self.bin_s, "S": Ss})
        alarmes.sort(key=lambda a: a["t_s"])
        self._marcar_causa_comum(alarmes)
        return alarmes, traj

    def _alarme(self, enc, fam, c, sub, t, s):
        janela = sub[(sub["t_s"] > t - 86400) & (sub["t_s"] <= t)]
        n24 = len(_incidentes(janela["t_s"].to_numpy(), self.declust_s))
        cod = janela["codigo"].value_counts()
        sem_contador = cod.drop(list(CONTADORES), errors="ignore")
        cod = sem_contador if len(sem_contador) else cod
        codigo = str(cod.index[0])
        normal_dia = c["lambda0_h"] * 24
        extra = ""
        amp = janela.loc[janela["codigo"].isin(list(AMPLITUDE)), "valor"].dropna()
        if len(amp):
            extra = f"; ultima amplitude {amp.iloc[-1]:.0f} mV (nominal 375-600 mV)"
        falhas = int((janela["tipo"] == "falha").sum())
        if falhas:
            extra += f"; {falhas} falha(s) transitoria(s) nas ultimas 24 h"
        msg = (f"{enc.upper()} ({c['drive']}): {codigo} - {DESCRICAO.get(codigo, codigo)}. "
               f"{n24} ocorrencias nas ultimas 24 h (normal: {normal_dia:.2f} por dia){extra}. "
               f"Acao: {ACAO[fam]}.")
        return {"t_s": float(t), "encoder": enc, "familia": fam, "drive": c["drive"],
                "codigo": codigo, "n_24h": n24, "S": float(s), "h": c["h"],
                "mensagem": msg, "causa_comum": False}

    @staticmethod
    def _marcar_causa_comum(alarmes, janela_s=600.0):
        """Varios encoders alarmando juntos na mesma familia = provavel causa comum (EMC, rede)."""
        for a in alarmes:
            outros = {b["encoder"] for b in alarmes if b["familia"] == a["familia"]
                      and abs(b["t_s"] - a["t_s"]) <= janela_s}
            if len(outros) >= 2:
                a["causa_comum"] = True
                if "causa comum" not in a["mensagem"]:
                    a["mensagem"] += (f" Atencao: {len(outros)} encoders alarmaram juntos - "
                                      "possivel causa comum (EMC, alimentacao, rede).")

    # -------------------------------------------------------- exportar
    def exportar(self):
        return {"bin_min": self.bin_s / 60, "declust_s": self.declust_s, "meta_fa_semana": self.meta,
                "razao": self.razao, "holdoff_h": self.holdoff_s / 3600, "prior_h": self.prior_h,
                "fator_seguranca": self.fator_seguranca,
                "canais": [{"encoder": e, "familia": f, **c} for (e, f), c in self.canais.items()]}

    @classmethod
    def carregar(cls, dados):
        if isinstance(dados, str):
            dados = json.loads(dados)
        m = cls(dados["bin_min"], dados["declust_s"], dados["meta_fa_semana"], dados["razao"],
                dados["holdoff_h"], dados["prior_h"], dados["fator_seguranca"])
        m.canais = {(c["encoder"], c["familia"]): {k: v for k, v in c.items() if k not in ("encoder", "familia")}
                    for c in dados["canais"]}
        return m


# ------------------------------------------------------------------ regra atual (comparacao)
class RegraAtual:
    """A regra de monitor/monitor.py: mensagens na ultima 1 h > (maximo no trecho normal + 2),
    reavisa a cada 10 min. familias=('cabo',) reproduz o indicador 'erros' atual
    (so erros de comunicacao); familias=FAMILIAS e a mesma regra aplicada a todos os avisos."""

    def __init__(self, familias=("cabo",), janela_s=3600.0, margem=2, holdoff_min=10.0):
        self.familias, self.janela_s, self.margem = tuple(familias), janela_s, margem
        self.holdoff_s = holdoff_min * 60

    def _brutos(self, eventos, contadores):
        """Contagem bruta: cada incremento do contador conta 1 (como o 'erros_<encoder>' atual);
        com contador, o aviso_cabo nao e somado de novo."""
        ev = _juntar(eventos, None, 1.0)
        linhas = [ev[["t_s", "encoder", "familia"]]]
        if contadores is not None and len(contadores):
            com_cont = set(contadores["encoder"])
            linhas[0] = ev[~((ev["codigo"] == "aviso_cabo") & ev["encoder"].isin(com_cont))][["t_s", "encoder", "familia"]]
            for enc, g in contadores.sort_values("t_s").groupby("encoder"):
                v = g["valor"].to_numpy(float)
                d = np.diff(v)
                d = np.where(d < 0, v[1:], d).astype(int)
                t = np.repeat(g["t_s"].to_numpy()[1:], np.clip(d, 0, None))
                linhas.append(pd.DataFrame({"t_s": t, "encoder": enc, "familia": "cabo"}))
        ev = pd.concat(linhas, ignore_index=True)
        return ev[ev["familia"].isin(self.familias)].sort_values("t_s")

    def _contagem(self, t):
        return np.arange(1, len(t) + 1) - np.searchsorted(t, t - self.janela_s, side="right")

    def treinar(self, eventos, encoders, contadores=None):
        ev = self._brutos(eventos, contadores)
        self.limiar = {}
        for enc in encoders:
            t = ev.loc[ev["encoder"] == enc, "t_s"].to_numpy()
            self.limiar[enc] = int(self._contagem(t).max() if len(t) else 0) + self.margem
        return self

    def analisar(self, eventos, t_ini, t_fim, contadores=None):
        ev = self._brutos(eventos, contadores)
        alarmes = []
        for enc, lim in self.limiar.items():
            t = ev.loc[(ev["encoder"] == enc) & (ev["t_s"] < t_fim), "t_s"].to_numpy()
            n = self._contagem(t)
            ultimo = -np.inf
            for ti, ni in zip(t, n):
                if ti >= t_ini and ni > lim and ti - ultimo >= self.holdoff_s:
                    alarmes.append({"t_s": float(ti), "encoder": enc, "n_1h": int(ni), "limiar": lim,
                                    "mensagem": f"{enc.upper()}: {ni} erros/avisos na ultima hora "
                                                f"(normal ate {lim - self.margem})"})
                    ultimo = ti
        alarmes.sort(key=lambda a: a["t_s"])
        return alarmes
