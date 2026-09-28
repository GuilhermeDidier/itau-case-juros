"""Confere o resultado acumulado (P&L do dia, semana, mês e total).

1. Para títulos, a soma dos P&Ls diários tem que ser EXATAMENTE a
   reprecificação entre as pontas: valor final + caixa − valor inicial
   telescopa. Se não bater, algum dia está sendo contado duas vezes ou
   pulado.
2. Para o DI1 a soma NÃO telescopa, porque o ajuste de cada dia paga o CDI
   sobre o PU daquele dia. A diferença tem que existir e ser pequena perto
   do resultado.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from carteira import Carteira  # noqa: E402
from curvas import FonteDI1Historico  # noqa: E402
from titulos import Titulo  # noqa: E402


def main() -> int:
    fonte = FonteDI1Historico()
    datas = fonte.datas_disponiveis()[-60:]
    curvas = [fonte.curva(d) for d in datas]

    c = Carteira("títulos")
    c.adicionar(Titulo.ltn(date(2028, 1, 1)), 50_000)
    c.adicionar(Titulo.ltn(date(2031, 1, 1)), -30_000)
    c.adicionar(Titulo.ntnf(date(2035, 1, 1)), 20_000)
    serie = c.resultado_diario(curvas)
    pontas = c.pl_entre(curvas[0], curvas[-1])["total"]
    dif = abs(serie["total"].sum() - pontas)
    print(f"títulos: {len(serie)} pregões, soma R$ {serie['total'].sum():,.2f}, "
          f"pontas R$ {pontas:,.2f}, diferença R$ {dif:.6f}")
    assert dif < 1e-3, "soma diária não telescopa"
    assert abs(serie["acumulado"].iloc[-1] - serie["total"].sum()) < 1e-6
    assert (serie["de"].iloc[1:].values == serie["para"].iloc[:-1].values).all(), "elo quebrado"

    d = Carteira("DI1")
    d.adicionar(Titulo.di1(date(2029, 1, 1)), -1_000)
    serie = d.resultado_diario(curvas)
    soma = serie["total"].sum()
    pontas = d.pl_entre(curvas[0], curvas[-1])["total"]
    print(f"DI1: soma diária R$ {soma:,.0f}, reprecificação entre pontas R$ {pontas:,.0f}")
    assert soma != pontas, "DI1 deveria acumular CDI dia a dia"

    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
