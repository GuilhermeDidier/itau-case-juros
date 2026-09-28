"""Confere a curva ao vivo do DI1 contra a curva oficial da B3.

A curva referencial PRE da B3 é construída a partir dos ajustes do DI1. Então
a curva que montamos com o ajuste do pregão anterior, vencimento a vencimento,
tem que reproduzir a PRE publicada para aquela data. Se bater, o dado ao vivo
está entrando no mesmo lugar (prazo em dias úteis, base 252, unidade).

Precisa de rede — o dado é o de hoje.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from curvas import FonteB3, FonteDI1AoVivo  # noqa: E402

TOLERANCIA_BPS = 1.0  # ajuste vem com 3 casas: arredondamento de até 0,5bp


def main() -> int:
    fonte = FonteDI1AoVivo()
    df = fonte.contratos()
    ao_vivo = fonte.curva_ao_vivo(df)
    ajuste = fonte.curva_fechamento_anterior(df)

    print(f"B3 às {df.attrs['horario']}: {len(df)} contratos, "
          f"origem {df['origem'].value_counts().to_dict()}")
    assert len(ao_vivo.dias_uteis) >= 20, "poucos vencimentos na cotação"

    if df.attrs.get("ajuste_de_hoje"):
        # Depois do fechamento a B3 troca o "ajuste anterior" pelo de hoje; a
        # fonte percebe (não bate com a PRE de ontem) e usa a PRE oficial.
        print(f"pregão encerrado: a cotação já traz o ajuste de hoje; ponto de "
              f"partida do P&L = {ajuste.fonte} de {ajuste.data}. A conferência "
              f"campo a campo só é possível durante o pregão.")
    else:
        oficial = FonteB3("PRE", usar_cache=False).curva(ajuste.data)
        dif = (ajuste.taxas - oficial.taxa(ajuste.dias_uteis)) * 10_000
        pior = np.abs(dif).max()
        print(f"ajuste DI1 x PRE oficial em {ajuste.data}: {len(dif)} vértices, "
              f"maior diferença {pior:.2f}bp")
        assert pior < TOLERANCIA_BPS, f"curva do DI1 descolou da PRE: {pior:.2f}bp"

    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
