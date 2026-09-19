"""Cenários de curva: os nomeados da mesa e os extraídos de dias reais.

Cenário inventado com número redondo — "bull steepening de 20bps na curta e
5 na longa" — é conveniente e não acontece. A curva real não se move em
linha reta entre dois vértices, e o dia de Copom com surpresa tem uma
assinatura própria: a ponta curta salta, o miolo acompanha em parte, a longa
quase não se mexe.

Por isso existem dois tipos aqui. Os nomeados servem para responder "e se",
com forma controlada. Os históricos pegam a variação observada vértice a
vértice num dia real e reaplicam sobre a curva de hoje — e esses são os que
um trader reconhece.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

from curvas import Curva, FonteCurva

# Âncoras em dias úteis: 6 meses, 1, 2, 5 e 10 anos.
CURTA, UM_ANO, DOIS_ANOS, CINCO_ANOS, DEZ_ANOS = 126, 252, 504, 1260, 2520

# Malha para cenário histórico. Começa em 1 dia útil de propósito: num dia de
# Copom o overnight se move os 25bps cheios enquanto o resto da curva mal
# reage. O impacto em P&L é desprezível — o vértice de 1 du quase não tem
# DV01 — mas descartá-lo apagaria justamente a assinatura que identifica o
# dia como decisão de política monetária.
MALHA_HISTORICA = [1, 5, 21, 63, CURTA, UM_ANO, DOIS_ANOS, 756, CINCO_ANOS, DEZ_ANOS]


@dataclass(frozen=True)
class Cenario:
    """Uma transformação de curva, com nome e justificativa."""

    nome: str
    descricao: str
    aplicar: Callable[[Curva], Curva]
    origem: str = "nomeado"

    def __call__(self, curva: Curva) -> Curva:
        return self.aplicar(curva)


def _por_ancoras(nome: str, descricao: str, ancoras: dict[int, float]) -> Cenario:
    return Cenario(
        nome=nome,
        descricao=descricao,
        aplicar=lambda c, a=ancoras: c.com_choques(a),
    )


def _paralelo(bps: float) -> Cenario:
    return Cenario(
        nome=f"Paralelo {bps:+.0f}bps",
        descricao="Toda a curva se desloca igualmente. É o cenário que menos "
        "acontece, e o único que o DV01 sozinho descreve bem.",
        aplicar=lambda c, b=bps: c.deslocar(b),
    )


def catalogo_nomeado(intensidade: float = 50.0) -> dict[str, Cenario]:
    """Cenários padrão de mesa, parametrizados pela intensidade em bps.

    Bull = taxa caindo, bear = taxa subindo. Steepening = a curva se
    inclina (a longa sobe mais que a curta, ou a curta cai mais que a longa);
    flattening = o contrário.
    """
    i = intensidade
    return {
        c.nome: c
        for c in [
            _paralelo(+i),
            _paralelo(-i),
            _por_ancoras(
                "Bull steepening",
                "Queda de juros concentrada na ponta curta — típico de "
                "início de ciclo de corte já precificado.",
                {CURTA: -i, UM_ANO: -i * 0.85, DOIS_ANOS: -i * 0.6,
                 CINCO_ANOS: -i * 0.3, DEZ_ANOS: -i * 0.15},
            ),
            _por_ancoras(
                "Bear flattening",
                "Alta de juros concentrada na ponta curta — susto "
                "inflacionário de curto prazo com a longa ancorada.",
                {CURTA: +i, UM_ANO: +i * 0.85, DOIS_ANOS: +i * 0.6,
                 CINCO_ANOS: +i * 0.3, DEZ_ANOS: +i * 0.15},
            ),
            _por_ancoras(
                "Bear steepening",
                "Alta concentrada na ponta longa — deterioração fiscal ou "
                "prêmio de risco, sem mudança no juro de curto prazo.",
                {CURTA: +i * 0.15, UM_ANO: +i * 0.3, DOIS_ANOS: +i * 0.6,
                 CINCO_ANOS: +i * 0.85, DEZ_ANOS: +i},
            ),
            _por_ancoras(
                "Bull flattening",
                "Queda concentrada na ponta longa — fuga para qualidade ou "
                "revisão de crescimento para baixo.",
                {CURTA: -i * 0.15, UM_ANO: -i * 0.3, DOIS_ANOS: -i * 0.6,
                 CINCO_ANOS: -i * 0.85, DEZ_ANOS: -i},
            ),
            _por_ancoras(
                "Surpresa de Copom",
                "Choque agudo na ponta curta que se dissipa em dois anos. "
                "A longa não se move porque o juro terminal não mudou.",
                {21: +i, CURTA: +i * 0.75, UM_ANO: +i * 0.4,
                 DOIS_ANOS: +i * 0.1, CINCO_ANOS: 0.0, DEZ_ANOS: 0.0},
            ),
        ]
    }


# --- cenários históricos -------------------------------------------------


def variacao_entre(inicial: Curva, final: Curva, vertices: list[int]) -> dict[int, float]:
    """Variação observada, em bps, vértice a vértice."""
    return {
        v: float((final.taxa(v) - inicial.taxa(v)) * 10_000) for v in vertices
    }


def cenario_historico(
    inicial: Curva,
    final: Curva,
    vertices: list[int] | None = None,
    rotulo: str | None = None,
) -> Cenario:
    """Transforma o movimento observado entre duas datas em cenário reaplicável.

    Reaplica a VARIAÇÃO sobre a curva de hoje, não substitui a curva pela
    antiga. A pergunta que interessa não é "quanto a carteira valia naquele
    dia", é "quanto ela perderia se aquele dia se repetisse agora".
    """
    vertices = vertices or MALHA_HISTORICA
    deltas = variacao_entre(inicial, final, vertices)

    maior = max(deltas.values(), key=abs)
    nome = rotulo or f"Repetir {final.data:%d/%m/%Y}"

    return Cenario(
        nome=nome,
        descricao=(
            f"Variação observada entre {inicial.data:%d/%m/%Y} e "
            f"{final.data:%d/%m/%Y}: máximo de {maior:+.1f}bps. "
            f"Forma real da curva naquele dia, não choque estilizado."
        ),
        aplicar=lambda c, d=deltas: c.com_choques(d),
        origem=f"histórico {inicial.data}→{final.data}",
    )


def catalogar_movimentos(
    fonte: FonteCurva,
    datas: list[date] | None = None,
    vertices: list[int] | None = None,
    janela: int = 1,
) -> pd.DataFrame:
    """Varre o histórico disponível e ordena os dias por tamanho do movimento.

    Serve para achar os dias que valem virar cenário sem precisar consultar
    calendário de Copom: um dia de decisão com surpresa aparece sozinho no
    topo, e a forma da variação por vértice revela se foi choque de curta ou
    reprecificação de toda a curva.
    """
    vertices = vertices or MALHA_HISTORICA
    datas = sorted(datas or fonte.datas_disponiveis())

    linhas = []
    for anterior, atual in zip(datas, datas[janela:]):
        try:
            c0, c1 = fonte.curva(anterior), fonte.curva(atual)
        except Exception:  # noqa: BLE001
            continue

        d = variacao_entre(c0, c1, vertices)
        valores = np.array(list(d.values()))
        curta = d[CURTA]
        longa = d[DEZ_ANOS]

        linhas.append(
            {
                "de": anterior,
                "para": atual,
                "max_abs_bps": round(float(np.abs(valores).max()), 2),
                "medio_bps": round(float(valores.mean()), 2),
                "curta_bps": round(curta, 2),
                "longa_bps": round(longa, 2),
                "inclinacao_bps": round(longa - curta, 2),
                "formato": _classificar(curta, longa),
            }
        )

    df = pd.DataFrame(linhas)
    return df.sort_values("max_abs_bps", ascending=False).reset_index(drop=True)


def ranking_por_impacto(
    carteira,
    curva_hoje: Curva,
    fonte: FonteCurva,
    datas: list[date] | None = None,
    janela: int = 1,
    vertices: list[int] | None = None,
) -> pd.DataFrame:
    """Ordena os dias históricos pelo estrago que fariam NESTA carteira.

    Ranquear por variação em bps mede o que a curva fez; ranquear por P&L
    mede o que importa. Um dia de Copom move 25bps no overnight e quase nada
    em P&L, porque o vértice de 1 dia útil não carrega DV01. Um dia sonolento
    que move 8bps em toda a parte longa pode doer dez vezes mais.

    A distinção é a diferença entre um relatório de mercado e uma ferramenta
    de risco.
    """
    datas = sorted(datas or fonte.datas_disponiveis())
    base = carteira.valor(curva_hoje)

    linhas = []
    for anterior, atual in zip(datas, datas[janela:]):
        try:
            c0, c1 = fonte.curva(anterior), fonte.curva(atual)
        except Exception:  # noqa: BLE001
            continue

        cenario = cenario_historico(c0, c1, vertices)
        pl = carteira.valor(cenario(curva_hoje)) - base
        d = variacao_entre(c0, c1, vertices or MALHA_HISTORICA)

        linhas.append(
            {
                "de": anterior,
                "para": atual,
                "PL": round(pl, 2),
                "PL_%": round(pl / base * 100, 4),
                "max_abs_bps": round(float(np.abs(list(d.values())).max()), 2),
                "curta_bps": round(d[CURTA], 2),
                "longa_bps": round(d[DEZ_ANOS], 2),
                "formato": _classificar(d[CURTA], d[DEZ_ANOS]),
            }
        )

    df = pd.DataFrame(linhas)
    return df.sort_values("PL").reset_index(drop=True)


def _classificar(curta: float, longa: float, limiar: float = 1.0) -> str:
    """Nomeia o movimento pela combinação de direção e inclinação."""
    if abs(curta) < limiar and abs(longa) < limiar:
        return "parado"

    subindo = (curta + longa) / 2 > 0
    inclinando = longa - curta > limiar
    achatando = longa - curta < -limiar

    if not inclinando and not achatando:
        return "paralelo de alta" if subindo else "paralelo de baixa"

    # Steepening é a inclinação AUMENTANDO — longa subindo mais que a curta,
    # ou curta caindo mais que a longa. O teste é o mesmo nos dois lados; o
    # que bull/bear muda é só a direção do nível.
    if subindo:
        return "bear steepening" if inclinando else "bear flattening"
    return "bull steepening" if inclinando else "bull flattening"
