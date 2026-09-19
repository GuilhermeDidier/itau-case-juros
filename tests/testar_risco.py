"""Verifica as propriedades que os números de risco têm que respeitar.

Com a amostra atual nenhum VaR aqui é utilizável como limite de risco — e é
justamente por isso que o teste checa ESTRUTURA, não valor. Um VaR de R$ 71
mil não pode ser validado contra nada hoje; mas ES ≥ VaR, monotonicidade na
confiança e reprodutibilidade sob semente têm que valer em qualquer amostra.

Quando o histórico longo chegar, estes testes continuam valendo e passam a
sustentar números que significam alguma coisa.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from carteira import Carteira, VERTICES_PADRAO  # noqa: E402
from curvas import FonteB3  # noqa: E402
from risco import (  # noqa: E402
    MINIMO_CONFIAVEL,
    decompor_pca,
    var_historico,
    var_monte_carlo,
    var_parametrico,
    variacoes_historicas,
)
from titulos import Titulo  # noqa: E402


def montar():
    fonte = FonteB3("PRE")
    curva = fonte.curva(sorted(fonte.datas_disponiveis())[-1])
    carteira = (
        Carteira("book")
        .adicionar(Titulo.ltn(date(2028, 1, 1)), 50_000)
        .adicionar(Titulo.ltn(date(2031, 1, 1)), -30_000)
        .adicionar(Titulo.ntnf(date(2035, 1, 1)), 20_000)
    )
    return fonte, curva, carteira


def main() -> int:
    fonte, curva, carteira = montar()
    variacoes = variacoes_historicas(fonte, VERTICES_PADRAO)
    n = len(variacoes)
    ok = []

    def checa(nome: str, condicao: bool, detalhe: str = "") -> None:
        ok.append(condicao)
        print(f"  {'OK  ' if condicao else 'FALHOU'} {nome}" + (f"  — {detalhe}" if detalhe else ""))

    print(f"Amostra: {n} movimentos observados\n")

    print("Propriedades estruturais:")
    hist, pl = var_historico(carteira, curva, variacoes)
    par = var_parametrico(carteira, curva, variacoes)
    mc, plmc = var_monte_carlo(carteira, curva, variacoes, simulacoes=5_000)

    for r in (hist, par, mc):
        checa(
            f"{r.metodo}: ES ≥ VaR",
            r.es >= r.var - 1e-6,
            f"ES {r.es:,.0f} vs VaR {r.var:,.0f}",
        )

    hist99, _ = var_historico(carteira, curva, variacoes, confianca=0.99)
    hist95, _ = var_historico(carteira, curva, variacoes, confianca=0.95)
    checa(
        "VaR cresce com a confiança",
        hist99.var >= hist95.var - 1e-6,
        f"99%: {hist99.var:,.0f} ≥ 95%: {hist95.var:,.0f}",
    )

    checa(
        "simulação histórica usa toda a amostra",
        len(pl) == n,
        f"{len(pl)} cenários de {n} movimentos",
    )

    mc_a, _ = var_monte_carlo(carteira, curva, variacoes, simulacoes=3_000, semente=7)
    mc_b, _ = var_monte_carlo(carteira, curva, variacoes, simulacoes=3_000, semente=7)
    checa("Monte Carlo reprodutível sob semente", abs(mc_a.var - mc_b.var) < 1e-9)

    print("\nHonestidade sobre a amostra:")
    checa(
        "amostra curta dispara aviso em todos os métodos",
        all(r.aviso for r in (hist, par, mc)) and n < MINIMO_CONFIAVEL,
        f"n={n} < {MINIMO_CONFIAVEL}",
    )
    checa(
        "resultado se declara não confiável",
        not any(r.confiavel for r in (hist, par, mc)),
    )

    print("\nEstrutura da curva (PCA):")
    pca = decompor_pca(variacoes)
    nomes = list(pca["interpretação"])
    checa(
        "três primeiros componentes são nível, inclinação e curvatura",
        nomes == ["nível", "inclinação", "curvatura"],
        " / ".join(nomes),
    )
    acumulada = float(pca["acumulada_%"].iloc[-1])
    checa(
        "três componentes explicam mais de 95% da variância",
        acumulada > 95,
        f"{acumulada:.2f}%",
    )
    primeiro = float(pca["variancia_explicada_%"].iloc[0])
    checa(
        "nível domina, como esperado em curva de juros",
        primeiro > 70,
        f"PC1 = {primeiro:.2f}%",
    )

    print()
    if all(ok):
        print("APROVADO — as propriedades valem; os valores ainda não significam nada")
        return 0
    print("REPROVADO")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
