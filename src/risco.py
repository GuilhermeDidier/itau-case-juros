"""VaR, Expected Shortfall e estrutura de movimento da curva.

Três métodos, de propósito. Não por indecisão: eles discordam de um jeito
informativo.

  histórico    reprecifica a carteira sob cada movimento de curva que de fato
               aconteceu. Não assume distribuição nenhuma, e por isso é o
               único que enxerga cauda gorda e assimetria. Preso ao tamanho
               da amostra.

  paramétrico  delta-normal: DV01 por vértice contra a matriz de covariância.
               Rápido e analítico. A literatura espera que subestime cauda,
               porque a curva de juros tem cauda mais gorda que a normal.
               ATENÇÃO: com a amostra atual isso NÃO está demonstrado — e o
               observado é o contrário. Ver nota sobre tamanho de amostra
               abaixo.

  Monte Carlo  amostra da normal multivariada e reprecifica de verdade. Herda
               a hipótese gaussiana do paramétrico, mas captura a não
               linearidade que o delta-normal perde.

Um VaR sem o tamanho da amostra ao lado é um número sem sentido. Toda saída
aqui carrega `n_observacoes` e um aviso explícito quando a amostra é curta
demais para a cauda que se está estimando.

NOTA SOBRE A AMOSTRA ATUAL
Com os ~19 movimentos que a B3 disponibiliza, o quantil de 99% se apoia em
0,2 observação: o "VaR histórico" é, na prática, o pior dia da amostra, e a
distribuição não tem cauda para ser vista. Medido hoje, o paramétrico sai
MAIOR que o histórico — o oposto do esperado na literatura. Isso não é
evidência contra a cauda gorda; é evidência de que 19 pontos não estimam
cauda nenhuma. A comparação entre os três métodos só passa a significar
alguma coisa com o histórico longo da extração externa.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from calendario import pares_consecutivos
from curvas import Curva, FonteCurva

# Abaixo disso, estimar um quantil de 99% é ilusão: o 1% da cauda cai em
# menos de 3 observações.
MINIMO_CONFIAVEL = 250


@dataclass
class ResultadoRisco:
    metodo: str
    confianca: float
    var: float  # perda em reais, positiva
    es: float  # Expected Shortfall, positiva
    n_observacoes: int
    horizonte_du: int = 1
    aviso: str = ""

    @property
    def confiavel(self) -> bool:
        return not self.aviso

    def __str__(self) -> str:
        linha = (
            f"{self.metodo:<14} VaR {self.confianca:.0%}: R$ {self.var:>14,.2f}   "
            f"ES: R$ {self.es:>14,.2f}   n={self.n_observacoes}"
        )
        return linha + (f"\n  ⚠ {self.aviso}" if self.aviso else "")


def _avaliar_amostra(n: int, confianca: float) -> str:
    if n == 0:
        return "sem observações"
    cauda = n * (1 - confianca)
    if n < MINIMO_CONFIAVEL:
        return (
            f"amostra de {n} observações é curta — o quantil de "
            f"{confianca:.0%} se apoia em {cauda:.1f} ponto(s). "
            f"Número ilustrativo, não utilizável para limite de risco."
        )
    if cauda < 5:
        return f"apenas {cauda:.1f} observações na cauda de {confianca:.0%}"
    return ""


# --- movimentos observados ------------------------------------------------


def variacoes_historicas(
    fonte: FonteCurva,
    vertices: list[int],
    datas: list[date] | None = None,
    horizonte_du: int = 1,
) -> pd.DataFrame:
    """Matriz de variações diárias, em bps, por vértice.

    Uma linha por movimento observado, uma coluna por vértice. É a matéria
    prima dos três métodos.
    """
    datas = sorted(datas or fonte.datas_disponiveis())
    curvas = {}
    for d in datas:
        try:
            curvas[d] = fonte.curva(d)
        except Exception:  # noqa: BLE001
            continue

    disponiveis = sorted(curvas)
    linhas, indice = [], []
    for anterior, atual in pares_consecutivos(disponiveis, horizonte_du):
        c0, c1 = curvas[anterior], curvas[atual]
        linhas.append([(c1.taxa(v) - c0.taxa(v)) * 10_000 for v in vertices])
        indice.append(atual)

    return pd.DataFrame(linhas, index=pd.Index(indice, name="data"), columns=vertices)


def _curva_chocada(curva: Curva, vertices: list[int], deltas: np.ndarray) -> Curva:
    return curva.com_choques({v: float(d) for v, d in zip(vertices, deltas)})


# --- os três métodos ------------------------------------------------------


def var_historico(
    carteira,
    curva: Curva,
    variacoes: pd.DataFrame,
    confianca: float = 0.99,
) -> tuple[ResultadoRisco, pd.Series]:
    """Simulação histórica com full repricing.

    Cada movimento observado é reaplicado sobre a curva de hoje e a carteira
    é reprecificada inteira — não estimada por DV01. Captura convexidade e
    movimento não paralelo sem hipótese adicional.
    """
    vertices = list(variacoes.columns)
    base = carteira.valor(curva)

    pl = pd.Series(
        [
            carteira.valor(_curva_chocada(curva, vertices, linha.to_numpy())) - base
            for _, linha in variacoes.iterrows()
        ],
        index=variacoes.index,
        name="PL",
    )

    n = len(pl)
    if n == 0:
        return (
            ResultadoRisco("histórico", confianca, 0.0, 0.0, 0, aviso="sem observações"),
            pl,
        )

    corte = float(np.quantile(pl, 1 - confianca))
    cauda = pl[pl <= corte]
    return (
        ResultadoRisco(
            metodo="histórico",
            confianca=confianca,
            var=-corte,
            es=-float(cauda.mean()) if len(cauda) else -corte,
            n_observacoes=n,
            aviso=_avaliar_amostra(n, confianca),
        ),
        pl,
    )


def var_parametrico(
    carteira,
    curva: Curva,
    variacoes: pd.DataFrame,
    confianca: float = 0.99,
) -> ResultadoRisco:
    """Delta-normal: KRD contra a matriz de covariância dos vértices.

    Linear e gaussiano nas duas pontas. Fica aqui para ser comparado com o
    histórico: a distância entre os dois É o resultado, porque mede quanto a
    hipótese normal custa nesta carteira.
    """
    from scipy.stats import norm

    vertices = list(variacoes.columns)
    krd = carteira.dv01_por_vertice(curva, vertices)
    v = np.array([krd[x] for x in vertices], dtype=float)

    n = len(variacoes)
    if n < 2:
        return ResultadoRisco(
            "paramétrico", confianca, 0.0, 0.0, n, aviso="sem observações suficientes"
        )

    cov = np.cov(variacoes.to_numpy(), rowvar=False)
    desvio = float(np.sqrt(v @ np.atleast_2d(cov) @ v))

    z = norm.ppf(confianca)
    # ES da normal: densidade no quantil dividida pela cauda.
    fator_es = float(norm.pdf(z) / (1 - confianca))

    return ResultadoRisco(
        metodo="paramétrico",
        confianca=confianca,
        var=z * desvio,
        es=fator_es * desvio,
        n_observacoes=n,
        aviso=_avaliar_amostra(n, confianca),
    )


def var_monte_carlo(
    carteira,
    curva: Curva,
    variacoes: pd.DataFrame,
    confianca: float = 0.99,
    simulacoes: int = 10_000,
    semente: int = 42,
) -> tuple[ResultadoRisco, np.ndarray]:
    """Normal multivariada com reprecificação completa.

    Mantém a hipótese gaussiana do paramétrico, mas reprecifica de verdade em
    vez de linearizar. A diferença entre os dois isola o efeito da
    convexidade; a diferença para o histórico isola o efeito da hipótese
    distribucional. Separar as duas é o que permite dizer de onde vem o erro.
    """
    vertices = list(variacoes.columns)
    n = len(variacoes)
    if n < 2:
        return (
            ResultadoRisco(
                "Monte Carlo", confianca, 0.0, 0.0, n, aviso="sem observações suficientes"
            ),
            np.array([]),
        )

    media = variacoes.mean().to_numpy()
    cov = np.atleast_2d(np.cov(variacoes.to_numpy(), rowvar=False))

    rng = np.random.default_rng(semente)
    amostras = rng.multivariate_normal(media, cov, size=simulacoes, method="eigh")

    base = carteira.valor(curva)
    pl = np.array(
        [carteira.valor(_curva_chocada(curva, vertices, a)) - base for a in amostras]
    )

    corte = float(np.quantile(pl, 1 - confianca))
    cauda = pl[pl <= corte]

    aviso = _avaliar_amostra(n, confianca)
    if aviso:
        aviso = f"covariância estimada em {n} observações — {aviso}"

    return (
        ResultadoRisco(
            metodo="Monte Carlo",
            confianca=confianca,
            var=-corte,
            es=-float(cauda.mean()) if len(cauda) else -corte,
            n_observacoes=simulacoes,
            aviso=aviso,
        ),
        pl,
    )


# --- estrutura do movimento ----------------------------------------------


def decompor_pca(variacoes: pd.DataFrame) -> pd.DataFrame:
    """Componentes principais das variações da curva.

    Na curva de juros os três primeiros componentes costumam ser
    reconhecíveis: nível (toda a curva junto), inclinação (pontas em sentidos
    opostos) e curvatura (miolo contra as pontas). Se saírem assim, a matriz
    de covariância está descrevendo mercado e não ruído — e serve de
    sanidade antes de confiar no VaR paramétrico.
    """
    x = variacoes.to_numpy()
    if len(x) < 3:
        return pd.DataFrame()

    centrado = x - x.mean(axis=0)
    cov = np.cov(centrado, rowvar=False)
    autovalores, autovetores = np.linalg.eigh(np.atleast_2d(cov))

    ordem = np.argsort(autovalores)[::-1]
    autovalores, autovetores = autovalores[ordem], autovetores[:, ordem]
    explicada = autovalores / autovalores.sum()

    n = min(3, len(autovalores))
    dados = {
        "variancia_explicada_%": [round(explicada[i] * 100, 2) for i in range(n)],
        "acumulada_%": [round(explicada[: i + 1].sum() * 100, 2) for i in range(n)],
    }
    df = pd.DataFrame(dados, index=[f"PC{i+1}" for i in range(n)])

    for j, v in enumerate(variacoes.columns):
        df[f"{v}du"] = [round(float(autovetores[j, i]), 3) for i in range(n)]

    df["interpretação"] = [_nomear_componente(autovetores[:, i]) for i in range(n)]
    return df


def _nomear_componente(carga: np.ndarray) -> str:
    """Nomeia o componente pelo padrão de sinais das cargas."""
    sinais = np.sign(carga)
    trocas = int(np.sum(sinais[:-1] != sinais[1:]))
    if trocas == 0:
        return "nível"
    if trocas == 1:
        return "inclinação"
    if trocas == 2:
        return "curvatura"
    return f"{trocas} trocas de sinal"


# --- backtest ---------------------------------------------------------------


def backtest_var(
    carteira,
    curva: Curva,
    variacoes: pd.DataFrame,
    confianca: float = 0.99,
    janela: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Testa se o VaR erra na frequência que promete.

    Para cada dia t, estima o VaR só com as `janela` variações anteriores a t
    e o compara com o P&L que a carteira de hoje teria no movimento real de
    t. Um VaR de 99% bem calibrado é rompido em ~1% dos dias — nem mais
    (subestima risco), nem muito menos (capital parado à toa).

    A carteira fica fixa (a de hoje) de propósito: o que se testa é o
    MÉTODO, não o histórico de gestão de uma carteira que não existiu.

    Devolve a série dia a dia e um resumo por método, com o teste de Kupiec
    (a taxa de exceções é compatível com 1 - confiança?) e o semáforo de
    Basileia nas últimas 250 observações.
    """
    from scipy.stats import chi2

    if len(variacoes) <= janela + 50:
        raise ValueError(
            f"backtest precisa de mais que {janela + 50} variações "
            f"(janela de {janela} + dias testados); há {len(variacoes)}."
        )
    vertices = list(variacoes.columns)
    base = carteira.valor(curva)
    pl = np.array(
        [carteira.valor(_curva_chocada(curva, vertices, linha)) - base
         for linha in variacoes.to_numpy()]
    )
    krd = carteira.dv01_por_vertice(curva, vertices)
    v = np.array([krd[x] for x in vertices], dtype=float)
    z = _quantil_normal(confianca)
    x = variacoes.to_numpy()

    linhas = []
    for t in range(janela, len(pl)):
        passado = pl[t - janela:t]
        var_hist = -float(np.quantile(passado, 1 - confianca))
        cov = np.cov(x[t - janela:t], rowvar=False)
        var_par = z * float(np.sqrt(v @ cov @ v))
        linhas.append(
            {
                "data": variacoes.index[t],
                "PL": pl[t],
                "VaR_historico": var_hist,
                "VaR_parametrico": var_par,
            }
        )
    serie = pd.DataFrame(linhas).set_index("data")

    p = 1 - confianca
    resumo = []
    for metodo, coluna in (("histórico", "VaR_historico"), ("paramétrico", "VaR_parametrico")):
        excecao = serie["PL"] < -serie[coluna]
        serie[f"excecao_{coluna[4:]}"] = excecao
        n, k = len(excecao), int(excecao.sum())
        taxa = k / n if n else 0.0
        # Kupiec (POF): razão de verossimilhança contra a taxa prometida.
        if 0 < k < n:
            lr = -2 * ((n - k) * np.log(1 - p) + k * np.log(p)
                       - (n - k) * np.log(1 - taxa) - k * np.log(taxa))
        else:
            lr = -2 * n * np.log(1 - p) if k == 0 else float("inf")
        p_valor = float(1 - chi2.cdf(lr, 1))
        ultimas = int(excecao.iloc[-250:].sum())
        resumo.append(
            {
                "método": metodo,
                "dias testados": n,
                "exceções": k,
                "esperadas": round(n * p, 1),
                "taxa_%": round(taxa * 100, 2),
                "Kupiec p-valor": round(p_valor, 3),
                "veredito": "calibrado" if p_valor >= 0.05 else (
                    "subestima risco" if taxa > p else "superestima risco"),
                "exceções 250d": ultimas,
                "Basileia": "verde" if ultimas <= 4 else ("amarelo" if ultimas <= 9 else "vermelho"),
            }
        )
    return serie, pd.DataFrame(resumo)


def _quantil_normal(confianca: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(confianca))
