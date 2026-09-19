"""Confronta o apreçador contra os PUs oficiais do Tesouro Nacional.

Não é teste inventado: o Tesouro Transparente publica, para cada dia e cada
título, a taxa E o PU. Se o apreçador reconstrói o PU a partir da taxa e bate
ao centavo, então convenção de 252 dias úteis, calendário de feriados,
capitalização exponencial e montagem do fluxo de cupom estão todos corretos.

Dois lados, dois significados:

  VENDA   preço em que o investidor compra o título. É o par taxa/PU
          internamente consistente e serve de referência de apreçamento.

  COMPRA  preço de recompra pelo Tesouro. Medido em 17/09/2026, carrega um
          desconto uniforme de 0,051% sobre o preço (desvio de 0,0012% entre
          11 títulos de 72 a 2.577 dias úteis). É política comercial de
          recompra, não divergência de cálculo — por isso é verificado como
          spread constante, e não como erro.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from calendario import proximo_dia_util  # noqa: E402
from titulos import Titulo  # noqa: E402

ARQ = RAIZ / "data" / "raw" / "PrecoTaxaTesouroDireto.csv"

MAPA = {
    "Tesouro Prefixado": "LTN",
    "Tesouro Prefixado com Juros Semestrais": "NTN-F",
}

TOLERANCIA_CENTAVOS = 0.01
SPREAD_RECOMPRA_ESPERADO = -0.051  # em %, medido
SPREAD_TOLERANCIA = 0.01  # em %

# Três regimes, medidos e não supostos:
#
#  até 2013     liquidação D+1. Par taxa/PU consistente: reconstrói ao centavo.
#  2015 a 2021  NÃO RECONCILIA sob D+0 nem D+1. Resíduo de 0,02% a 0,12% do PU,
#               espalhado nos dois sinais e crescendo com o tempo. Arredondamento
#               de meia casa na taxa explicaria até ~0,03%; o resto está em
#               aberto. Pode ser mudança de convenção de liquidação no meio do
#               período, ou PU publicado a partir de taxa com outra precisão.
#               NÃO investigado a fundo — não é o regime que o simulador usa.
#  2022 adiante liquidação D+0. Par taxa/PU consistente: reconstrói ao centavo.
#
# A tolerância NÃO é alargada para a janela do meio passar. Ela é reportada
# como pendência aberta: se o fallback histórico do Tesouro for usado nesse
# período, o erro de apreçamento é conhecido e precisa ser declarado.
ANO_REGIME_ATUAL = 2022
ANO_FIM_REGIME_ANTIGO = 2014
JANELA_ABERTA = (2015, 2021)

_CACHE: pd.DataFrame | None = None


def _base() -> pd.DataFrame:
    global _CACHE
    if _CACHE is None:
        df = pd.read_csv(ARQ, sep=";", decimal=",", encoding="latin-1")
        df["Data Base"] = pd.to_datetime(df["Data Base"], dayfirst=True)
        df["Data Vencimento"] = pd.to_datetime(df["Data Vencimento"], dayfirst=True)
        _CACHE = df[df["Tipo Titulo"].isin(MAPA)].copy()
    return _CACHE


def carregar(data_base: date) -> pd.DataFrame:
    df = _base()
    return df[df["Data Base"] == pd.Timestamp(data_base)]


def avaliar(data_base: date, defasagem: int = 0) -> pd.DataFrame:
    """Reprecifica todos os prefixados de uma data, nos dois lados."""
    liquidacao = proximo_dia_util(data_base, defasagem)
    linhas = []

    for _, r in carregar(data_base).iterrows():
        venc = r["Data Vencimento"].date()
        especie = MAPA[r["Tipo Titulo"]]
        papel = Titulo.ltn(venc) if especie == "LTN" else Titulo.ntnf(venc)

        for lado in ("Compra", "Venda"):
            taxa = r[f"Taxa {lado} Manha"]
            pu_oficial = r[f"PU {lado} Manha"]
            if pd.isna(taxa) or pd.isna(pu_oficial) or taxa <= 0:
                continue
            pu_calc = papel.pu_por_taxa(taxa / 100.0, liquidacao)
            linhas.append(
                {
                    "data_base": data_base,
                    "papel": papel.nome,
                    "especie": especie,
                    "lado": lado,
                    "taxa_%": taxa,
                    "pu_oficial": pu_oficial,
                    "pu_calculado": round(pu_calc, 2),
                    "erro": round(pu_calc - pu_oficial, 4),
                    "erro_%": (pu_calc / pu_oficial - 1) * 100,
                }
            )

    return pd.DataFrame(linhas)


def datas_de_teste(quantidade: int = 8) -> list[date]:
    """Datas espalhadas pelo histórico, para não validar só o regime de hoje."""
    todas = sorted(_base()["Data Base"].unique())
    if len(todas) <= quantidade:
        return [pd.Timestamp(d).date() for d in todas]
    passo = len(todas) // quantidade
    escolhidas = todas[::passo][-quantidade:]
    return [pd.Timestamp(d).date() for d in escolhidas]


def regime_de(data_base: date) -> str:
    if data_base.year >= ANO_REGIME_ATUAL:
        return "atual"
    if data_base.year <= ANO_FIM_REGIME_ANTIGO:
        return "antigo"
    return "aberto"


def defasagem_de(data_base: date) -> int:
    """Liquidação D+0 no regime atual, D+1 nos demais (o melhor ajuste
    medido para a janela aberta também é D+1 em metade dos casos)."""
    return 0 if regime_de(data_base) == "atual" else 1


def main() -> int:
    datas = datas_de_teste()
    print("Validação do apreçador contra os PUs oficiais do Tesouro Nacional")
    print(f"{len(datas)} datas, de {datas[0]:%d/%m/%Y} a {datas[-1]:%d/%m/%Y}\n")

    partes = [avaliar(d, defasagem_de(d)) for d in datas]
    df = pd.concat([p for p in partes if not p.empty], ignore_index=True)
    df["regime"] = df["data_base"].map(regime_de)

    venda = df[df.lado == "Venda"]
    compra = df[df.lado == "Compra"]

    print("LADO VENDA — reconstrução do PU a partir da taxa publicada")
    print(f"{'data':<12}{'regime':>9}{'liq':>6}{'obs':>5}"
          f"{'erro máx (R$)':>16}{'erro máx (%)':>15}")
    for d, g in venda.groupby("data_base"):
        print(f"{d:%d/%m/%Y}  {g['regime'].iloc[0]:>8}{'D+'+str(defasagem_de(d)):>6}"
              f"{len(g):>5}{g['erro'].abs().max():>16.4f}"
              f"{g['erro_%'].abs().max():>15.4f}")

    atual = venda[venda.regime == "atual"]
    antigo = venda[venda.regime == "antigo"]
    aberto = venda[venda.regime == "aberto"]

    pior_atual = atual["erro"].abs().max()
    print(f"\n  regime atual  n={len(atual):<4} erro máx R$ {pior_atual:.4f}   "
          f"dentro de R$ 0,01: "
          f"{(atual['erro'].abs() <= TOLERANCIA_CENTAVOS).mean()*100:.0f}%")
    for esp, g in atual.groupby("especie"):
        print(f"      {esp:<6} n={len(g):<4} erro máx R$ {g['erro'].abs().max():.4f}")

    pior_antigo = antigo["erro"].abs().max()
    print(f"  regime antigo n={len(antigo):<4} erro máx R$ {pior_antigo:.4f}   "
          f"dentro de R$ 0,01: "
          f"{(antigo['erro'].abs() <= TOLERANCIA_CENTAVOS).mean()*100:.0f}%")

    print(f"\n  JANELA ABERTA {JANELA_ABERTA[0]}-{JANELA_ABERTA[1]}  n={len(aberto)}")
    print(f"    erro máx {aberto['erro_%'].abs().max():.4f}% do PU "
          f"(R$ {aberto['erro'].abs().max():.4f})")
    print("    não reconcilia sob D+0 nem D+1 — pendência conhecida, declarada.")

    print("\nLADO COMPRA — spread de recompra no regime atual")
    compra_atual = compra[compra.data_base.map(lambda d: d.year >= ANO_REGIME_ATUAL)]
    media = compra_atual["erro_%"].mean()
    print(f"  spread médio       : {media:.4f}%")
    print(f"  desvio             : {compra_atual['erro_%'].std():.4f}%")

    ok_atual = pior_atual <= TOLERANCIA_CENTAVOS
    ok_antigo = pior_antigo <= TOLERANCIA_CENTAVOS
    ok_compra = abs(media - SPREAD_RECOMPRA_ESPERADO) <= SPREAD_TOLERANCIA

    print()
    print(f"  apreçamento, regime atual  : {'OK' if ok_atual else 'FALHOU'}")
    print(f"  apreçamento, regime antigo : {'OK' if ok_antigo else 'FALHOU'}")
    print(f"  spread de recompra         : {'OK' if ok_compra else 'FALHOU'}")
    print(f"  janela {JANELA_ABERTA[0]}-{JANELA_ABERTA[1]}          : "
          f"PENDENTE (não bloqueia — regime não usado)")

    tudo = ok_atual and ok_antigo and ok_compra
    print("\n" + ("APROVADO" if tudo else "REPROVADO — investigar"))
    return 0 if tudo else 1


if __name__ == "__main__":
    raise SystemExit(main())
