"""Títulos públicos prefixados e o apreçamento em base 252.

Convenção local: capitalização exponencial, 252 dias úteis, fluxo de caixa
descontado dia útil a dia útil. Pagamento que cai em feriado rola para o
próximo dia útil.

LTN    (Tesouro Prefixado)                  zero cupom, principal 1.000
NTN-F  (Tesouro Prefixado c/ Juros Semest.) cupom de 10% a.a. em jan e jul
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from calendario import BASE_252, dias_uteis, proximo_dia_util

PRINCIPAL = 1000.0
CUPOM_NTNF_AA = 0.10
DI1_NOCIONAL = 100_000.0


def cupom_semestral(taxa_anual: float, principal: float = PRINCIPAL) -> float:
    """Cupom semestral na convenção brasileira: juro anual em base composta,
    convertido para o semestre pela raiz — não por divisão por dois."""
    return round(principal * ((1.0 + taxa_anual) ** 0.5 - 1.0), 6)


@dataclass(frozen=True)
class Fluxo:
    data: date
    valor: float


@dataclass
class Titulo:
    """Um título prefixado, definido pelo seu fluxo de caixa."""

    nome: str
    vencimento: date
    fluxos: list[Fluxo]
    principal: float = PRINCIPAL
    # Futuro não tem desembolso: o ajuste diário corrige o preço de ontem pelo
    # CDI. Muda o carrego e o "valor" da carteira, não o risco de taxa.
    futuro: bool = False

    # Contar dia útil é caro e o fluxo não muda dentro de uma data de
    # liquidação. Sem esse cache, uma bisseção de 200 passos recontava o
    # calendário 200 vezes por título — e o Monte Carlo seria inviável.
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def _fluxo_em(self, liquidacao: date) -> tuple[np.ndarray, np.ndarray]:
        if liquidacao not in self._cache:
            vivos = self.fluxos_futuros(liquidacao)
            self._cache[liquidacao] = (
                np.array([dias_uteis(liquidacao, f.data) for f in vivos], dtype=float),
                np.array([f.valor for f in vivos], dtype=float),
            )
        return self._cache[liquidacao]

    @classmethod
    def ltn(cls, vencimento: date, principal: float = PRINCIPAL) -> "Titulo":
        return cls(
            nome=f"LTN {vencimento:%d/%m/%Y}",
            vencimento=vencimento,
            fluxos=[Fluxo(vencimento, principal)],
            principal=principal,
        )

    @classmethod
    def di1(cls, vencimento: date) -> "Titulo":
        """Contrato futuro de DI1: R$ 100 mil no primeiro dia útil do mês de
        vencimento, PU descontado pela taxa do contrato — um zero-cupom."""
        mes = date(vencimento.year, vencimento.month, 1)
        letra = "FGHJKMNQUVXZ"[mes.month - 1]
        return cls(
            nome=f"DI1{letra}{mes:%y}",
            vencimento=mes,
            fluxos=[Fluxo(mes, DI1_NOCIONAL)],
            principal=DI1_NOCIONAL,
            futuro=True,
        )

    @classmethod
    def ntnf(
        cls,
        vencimento: date,
        principal: float = PRINCIPAL,
        taxa_cupom: float = CUPOM_NTNF_AA,
        emissao: date | None = None,
    ) -> "Titulo":
        """NTN-F paga cupom em 1º de janeiro e 1º de julho, e o principal no
        vencimento junto com o último cupom."""
        cupom = cupom_semestral(taxa_cupom, principal)
        inicio = emissao or date(2000, 1, 1)

        datas = [
            d
            for ano in range(inicio.year, vencimento.year + 1)
            for d in (date(ano, 1, 1), date(ano, 7, 1))
            if inicio < d <= vencimento
        ]

        fluxos = [Fluxo(d, cupom) for d in datas]
        if fluxos and fluxos[-1].data == vencimento:
            fluxos[-1] = Fluxo(vencimento, cupom + principal)
        else:
            fluxos.append(Fluxo(vencimento, principal))

        return cls(
            nome=f"NTN-F {vencimento:%d/%m/%Y}",
            vencimento=vencimento,
            fluxos=fluxos,
            principal=principal,
        )

    # --- apreçamento ---------------------------------------------------

    def fluxos_futuros(self, liquidacao: date) -> list[Fluxo]:
        """Fluxos ainda a receber, com a data de pagamento já rolada para
        dia útil."""
        vivos = []
        for f in self.fluxos:
            pagamento = proximo_dia_util(f.data)
            if pagamento > liquidacao:
                vivos.append(Fluxo(pagamento, f.valor))
        return vivos

    def prazos_du(self, liquidacao: date) -> np.ndarray:
        return self._fluxo_em(liquidacao)[0]

    def valores(self, liquidacao: date) -> np.ndarray:
        return self._fluxo_em(liquidacao)[1]

    def pu_por_taxa(self, taxa: float, liquidacao: date) -> float:
        """PU descontando todo o fluxo por uma única taxa — a taxa cotada
        no mercado (TIR do papel)."""
        du, vl = self._fluxo_em(liquidacao)
        return float(np.sum(vl * (1.0 + taxa) ** (-du / BASE_252)))

    def pu_por_curva(self, curva, liquidacao: date | None = None) -> float:
        """PU descontando cada fluxo pelo vértice correspondente da curva.

        É o `full repricing`: a verdade contra a qual DV01 e convexidade são
        apenas aproximações."""
        liquidacao = liquidacao or curva.data
        du, vl = self._fluxo_em(liquidacao)
        return float(np.sum(vl * curva.desconto(du)))

    def taxa_por_pu(self, pu: float, liquidacao: date) -> float:
        """Taxa interna de retorno implícita no PU (busca por bisseção —
        a função é monótona, então converge sempre)."""
        baixo, alto = -0.99, 10.0
        for _ in range(200):
            meio = (baixo + alto) / 2.0
            if self.pu_por_taxa(meio, liquidacao) > pu:
                baixo = meio
            else:
                alto = meio
        return (baixo + alto) / 2.0

    # --- sensibilidades -------------------------------------------------

    def duration_macaulay(self, taxa: float, liquidacao: date) -> float:
        du, vl = self._fluxo_em(liquidacao)
        vp = vl * (1.0 + taxa) ** (-du / BASE_252)
        anos = du / BASE_252
        return float(np.sum(vp * anos) / np.sum(vp))

    def dv01(self, curva, liquidacao: date | None = None, bump_bps: float = 1.0) -> float:
        """Variação de PU para 1bp de choque paralelo na curva.

        Calculado por diferença central — mais preciso que diferença para a
        frente e imune ao sinal do choque. Retorna valor negativo para posição
        comprada (taxa sobe, preço cai)."""
        liquidacao = liquidacao or curva.data
        cima = self.pu_por_curva(curva.deslocar(+bump_bps), liquidacao)
        baixo = self.pu_por_curva(curva.deslocar(-bump_bps), liquidacao)
        return (cima - baixo) / 2.0

    def gamma_pu(self, curva, liquidacao: date | None = None,
                 bump_bps: float = 1.0) -> float:
        """Segunda diferença do PU para um choque de `bump_bps`, em reais.

        É o termo de segunda ordem cru — o que entra na aproximação
        `ΔPU ≈ DV01·Δbps + ½·gamma·Δbps²`. Não confundir com convexidade."""
        liquidacao = liquidacao or curva.data
        base = self.pu_por_curva(curva, liquidacao)
        cima = self.pu_por_curva(curva.deslocar(+bump_bps), liquidacao)
        baixo = self.pu_por_curva(curva.deslocar(-bump_bps), liquidacao)
        return cima - 2.0 * base + baixo

    def convexidade(self, curva, liquidacao: date | None = None,
                    bump_bps: float = 1.0) -> float:
        """Convexidade na definição de mercado: (d²PU/dy²) / PU, em anos².

        Normalizada pelo preço e pelo quadrado do choque em decimal, que é a
        unidade em que a mesa cota convexidade. A segunda diferença crua em
        reais está em `gamma_pu`."""
        liquidacao = liquidacao or curva.data
        base = self.pu_por_curva(curva, liquidacao)
        dy = bump_bps / 10_000.0
        return self.gamma_pu(curva, liquidacao, bump_bps) / (base * dy**2)

    def dv01_por_vertice(self, curva, vertices: list[int],
                         liquidacao: date | None = None,
                         bump_bps: float = 1.0) -> dict[int, float]:
        """Key rate duration: sensibilidade a um bump localizado em cada
        vértice-chave, com os demais ancorados em zero.

        Usa o bump em tenda de `com_choque_local`. Com choque interpolado a
        partir de uma âncora só, todo vértice devolveria o DV01 paralelo e a
        soma daria N vezes o valor correto — erro silencioso, porque os
        números continuam plausíveis.

        Sem isso, `bear flattening` não é implementável — e flattening é o
        que a mesa negocia, não choque paralelo."""
        liquidacao = liquidacao or curva.data
        resultado = {}
        for v in vertices:
            cima = self.pu_por_curva(
                curva.com_choque_local(v, +bump_bps, vertices), liquidacao
            )
            baixo = self.pu_por_curva(
                curva.com_choque_local(v, -bump_bps, vertices), liquidacao
            )
            resultado[v] = (cima - baixo) / 2.0
        return resultado
