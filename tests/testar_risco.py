"""Verifica as propriedades que os números de risco têm que respeitar.

Checa ESTRUTURA, não valor: ES ≥ VaR, monotonicidade na confiança,
reprodutibilidade sob semente e a PCA canônica valem em qualquer amostra. Se o
número está calibrado é o que o backtest do VaR responde, no app.

Também confere a honestidade sobre a amostra: um recorte curto tem que se
declarar ilustrativo, e o histórico inteiro do DI1 não.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from carteira import Carteira, VERTICES_PADRAO  # noqa: E402
from curvas import FonteDI1Historico  # noqa: E402
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
    fonte = FonteDI1Historico()
    curva = fonte.curva(fonte.datas_disponiveis()[-1])
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
    curta = variacoes.iloc[-20:]
    h_c, _ = var_historico(carteira, curva, curta)
    p_c = var_parametrico(carteira, curva, curta)
    checa(
        "recorte de 20 dias se declara ilustrativo",
        bool(h_c.aviso) and bool(p_c.aviso),
        f"n={len(curta)} < {MINIMO_CONFIAVEL}",
    )
    checa(
        "histórico inteiro não dispara aviso",
        n >= MINIMO_CONFIAVEL and hist.confiavel and par.confiavel,
        f"n={n}",
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
        print("APROVADO — as propriedades valem")
        return 0
    print("REPROVADO")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
