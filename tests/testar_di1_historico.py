"""Confere o histórico de ajustes do DI1 (data/di1_ajustes.csv).

Duas provas independentes, nenhuma contra expectativa própria:

1. Encadeamento. Cada arquivo de pregão traz o ajuste do dia E o do pregão
   anterior. O "anterior" de hoje tem que bater com o "ajuste" de ontem,
   contrato a contrato. Se as datas estivessem deslocadas ou o parser lendo o
   campo errado, isso quebraria. A diferença esperada é arredondamento: a B3
   publica o ajuste anterior corrigido pelo CDI e reconvertido em taxa com o
   prazo de hoje.

2. Curva oficial. Nos dias em que a B3 ainda publica a curva referencial PRE
   (~20 dias úteis), a curva montada dos ajustes tem que reproduzi-la.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from calendario import proximo_dia_util  # noqa: E402
from curvas import FonteB3, FonteDI1Historico  # noqa: E402


def main() -> int:
    fonte = FonteDI1Historico()
    datas = fonte.datas_disponiveis()
    print(f"{len(datas)} pregões, {datas[0]} a {datas[-1]}")
    ok = True

    df = pd.read_csv(fonte.caminho)
    df["data"] = pd.to_datetime(df["data"]).dt.date
    df["anterior"] = df["data"].map(lambda d: proximo_dia_util(d, -1))
    ontem = df[["data", "contrato", "ajuste"]].rename(
        columns={"data": "anterior", "ajuste": "ajuste_ontem"}
    )
    par = df.merge(ontem, on=["anterior", "contrato"]).dropna(subset=["ajuste_anterior"])
    # Contrato a menos de 1 mês do vencimento: poucos dias úteis amplificam o
    # arredondamento da conversão PU→taxa. Fica fora da régua.
    par = par[par.apply(lambda r: (fonte.vencimento(r.contrato) - r.data).days > 30, axis=1)]
    dif = (par["ajuste_anterior"] - par["ajuste_ontem"]).abs() * 100
    print(f"encadeamento: {len(par):,} pares, |dif| p99 {dif.quantile(.99):.2f}bp, "
          f"máx {dif.max():.2f}bp")
    if dif.quantile(0.99) > 1.0:
        print("  FALHOU: ajuste anterior não encadeia com o do dia anterior")
        ok = False

    b3 = FonteB3("PRE")
    comuns = [d for d in b3.datas_disponiveis() if d in set(datas)]
    piores = []
    for d in comuns:
        c = fonte.curva(d)
        piores.append(np.abs((c.taxas - b3.curva(d).taxa(c.dias_uteis)) * 10_000).max())
    if piores:
        print(f"curva oficial: {len(comuns)} dias em comum, maior diferença "
              f"{max(piores):.2f}bp")
        if max(piores) > 1.0:
            print("  FALHOU: curva dos ajustes descolou da PRE oficial")
            ok = False
    else:
        print("curva oficial: sem dias em comum (histórico ainda não chegou às últimas semanas)")

    print("ok" if ok else "FALHOU")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
