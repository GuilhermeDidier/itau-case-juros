"""Carteira de renda fixa, marcação e decomposição de P&L.

A pergunta que a mesa faz não é "quanto ganhei", é "de onde veio". Um P&L de
+R$ 180 mil não significa nada até separar quanto veio do tempo passar e
quanto veio da curva mexer — e, dentro da curva, de qual pedaço dela.

A decomposição aqui é exata por construção: carrego e variação de taxa somam
o P&L total sem resíduo, porque são definidos como as duas metades da mesma
diferença. O resíduo aparece só um nível abaixo, quando a variação de taxa é
atribuída por vértice via key rate duration — e aí ele é reportado, não
escondido.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from curvas import Curva
from titulos import Titulo

# Vértices-chave padrão, em dias úteis: 6 meses a 10 anos.
VERTICES_PADRAO = [126, 252, 504, 756, 1260, 2520]


@dataclass
class Posicao:
    """Uma posição: título e quantidade. Quantidade negativa é posição vendida."""

    titulo: Titulo
    quantidade: float

    @property
    def nome(self) -> str:
        return self.titulo.nome

    def valor(self, curva: Curva, liquidacao: date | None = None) -> float:
        return self.quantidade * self.titulo.pu_por_curva(curva, liquidacao)

    def caixa_entre(self, inicio: date, fim: date) -> float:
        """Cupom e principal recebidos em (inicio, fim].

        Sem isso a decomposição acusaria prejuízo no dia do cupom: o PU cai
        justamente porque o dinheiro saiu do título e entrou no caixa."""
        from calendario import proximo_dia_util

        total = 0.0
        for f in self.titulo.fluxos:
            pagamento = proximo_dia_util(f.data)
            if inicio < pagamento <= fim:
                total += f.valor
        return self.quantidade * total


@dataclass
class Carteira:
    nome: str
    posicoes: list[Posicao] = field(default_factory=list)

    def adicionar(self, titulo: Titulo, quantidade: float) -> "Carteira":
        self.posicoes.append(Posicao(titulo, quantidade))
        return self

    # --- marcação --------------------------------------------------------

    def valor(self, curva: Curva, liquidacao: date | None = None) -> float:
        return sum(p.valor(curva, liquidacao) for p in self.posicoes)

    def dv01(self, curva: Curva, liquidacao: date | None = None) -> float:
        return sum(
            p.quantidade * p.titulo.dv01(curva, liquidacao) for p in self.posicoes
        )

    def dv01_por_vertice(
        self,
        curva: Curva,
        vertices: list[int] | None = None,
        liquidacao: date | None = None,
    ) -> dict[int, float]:
        vertices = vertices or VERTICES_PADRAO
        total = {v: 0.0 for v in vertices}
        for p in self.posicoes:
            krd = p.titulo.dv01_por_vertice(curva, vertices, liquidacao)
            for v, x in krd.items():
                total[v] += p.quantidade * x
        return total

    def resumo(self, curva: Curva, liquidacao: date | None = None) -> pd.DataFrame:
        liquidacao = liquidacao or curva.data
        linhas = []
        for p in self.posicoes:
            pu = p.titulo.pu_por_curva(curva, liquidacao)
            linhas.append(
                {
                    "papel": p.nome,
                    "quantidade": p.quantidade,
                    "PU": round(pu, 2),
                    "valor": round(p.quantidade * pu, 2),
                    "taxa_%": round(p.titulo.taxa_por_pu(pu, liquidacao) * 100, 4),
                    "duration": round(
                        p.titulo.duration_macaulay(
                            p.titulo.taxa_por_pu(pu, liquidacao), liquidacao
                        ),
                        3,
                    ),
                    "DV01": round(p.quantidade * p.titulo.dv01(curva, liquidacao), 2),
                    "convexidade": round(p.titulo.convexidade(curva, liquidacao), 2),
                }
            )
        return pd.DataFrame(linhas)

    # --- decomposição de P&L ---------------------------------------------

    def decompor_pl(
        self,
        curva_inicial: Curva,
        curva_final: Curva,
        liquidacao_inicial: date | None = None,
        liquidacao_final: date | None = None,
        vertices: list[int] | None = None,
    ) -> dict:
        """Separa o P&L entre passagem de tempo e movimento de curva.

        carrego  reprecifica na curva ANTIGA com o prazo NOVO: é o que a
                 carteira renderia se a curva não tivesse mexido. Engloba
                 carrego e roll-down, que só se separam sob hipótese sobre a
                 forward — e essa hipótese seria escondida no número.

        taxa     reprecifica na curva NOVA com o mesmo prazo novo: é o que o
                 movimento de mercado fez, limpo do efeito do tempo.

        As duas somam o P&L total exatamente, porque são as duas metades da
        mesma diferença. Cupom pago no intervalo entra como caixa, senão o dia
        do cupom apareceria como prejuízo.
        """
        vertices = vertices or VERTICES_PADRAO
        t0 = liquidacao_inicial or curva_inicial.data
        t1 = liquidacao_final or curva_final.data

        valor_inicial = self.valor(curva_inicial, t0)
        valor_carregado = self.valor(curva_inicial, t1)  # curva velha, prazo novo
        valor_final = self.valor(curva_final, t1)
        caixa = sum(p.caixa_entre(t0, t1) for p in self.posicoes)

        carrego = valor_carregado - valor_inicial + caixa
        efeito_taxa = valor_final - valor_carregado
        total = valor_final + caixa - valor_inicial

        return {
            "valor_inicial": valor_inicial,
            "valor_final": valor_final,
            "caixa_recebido": caixa,
            "carrego": carrego,
            "efeito_taxa": efeito_taxa,
            "total": total,
            "por_vertice": self._atribuir_por_vertice(
                curva_inicial, curva_final, t1, vertices, efeito_taxa
            ),
            "por_papel": self._atribuir_por_papel(
                curva_inicial, curva_final, t0, t1
            ),
        }

    def _atribuir_por_vertice(
        self,
        curva_inicial: Curva,
        curva_final: Curva,
        liquidacao: date,
        vertices: list[int],
        efeito_taxa: float,
    ) -> pd.DataFrame:
        """Atribui o efeito de taxa aos vértices via key rate duration.

        É aproximação de primeira ordem: KRD vezes a variação observada em
        cada vértice. O que sobra é convexidade e movimento entre vértices, e
        aparece explicitamente como 'não linear' — porque uma decomposição que
        fecha por construção esconde o erro em vez de mostrá-lo.
        """
        krd = self.dv01_por_vertice(curva_inicial, vertices, liquidacao)

        linhas = []
        explicado = 0.0
        for v in vertices:
            variacao_bps = (curva_final.taxa(v) - curva_inicial.taxa(v)) * 10_000
            contribuicao = krd[v] * variacao_bps
            explicado += contribuicao
            linhas.append(
                {
                    "vertice_du": v,
                    "taxa_inicial_%": round(curva_inicial.taxa(v) * 100, 4),
                    "taxa_final_%": round(curva_final.taxa(v) * 100, 4),
                    "variacao_bps": round(variacao_bps, 2),
                    "DV01": round(krd[v], 2),
                    "PL": round(contribuicao, 2),
                }
            )

        linhas.append(
            {
                "vertice_du": None,
                "taxa_inicial_%": None,
                "taxa_final_%": None,
                "variacao_bps": None,
                "DV01": None,
                "PL": round(efeito_taxa - explicado, 2),
            }
        )
        df = pd.DataFrame(linhas)
        df["vertice"] = [f"{v} du" for v in vertices] + ["não linear"]
        return df[["vertice", "taxa_inicial_%", "taxa_final_%", "variacao_bps", "DV01", "PL"]]

    def _atribuir_por_papel(
        self, curva_inicial: Curva, curva_final: Curva, t0: date, t1: date
    ) -> pd.DataFrame:
        linhas = []
        for p in self.posicoes:
            inicial = p.valor(curva_inicial, t0)
            carregado = p.valor(curva_inicial, t1)
            final = p.valor(curva_final, t1)
            caixa = p.caixa_entre(t0, t1)
            linhas.append(
                {
                    "papel": p.nome,
                    "carrego": round(carregado - inicial + caixa, 2),
                    "efeito_taxa": round(final - carregado, 2),
                    "total": round(final + caixa - inicial, 2),
                }
            )
        return pd.DataFrame(linhas)

    # --- aproximação contra verdade ---------------------------------------

    def aproximar_vs_reprecificar(
        self,
        curva: Curva,
        choques_bps: np.ndarray,
        liquidacao: date | None = None,
    ) -> pd.DataFrame:
        """Compara DV01 (+ gamma) contra full repricing, choque a choque.

        Este é o quadro que prova entendimento de convexidade: a aproximação
        linear descola da verdade conforme o choque cresce, e o termo de
        segunda ordem recupera quase tudo. Sem o gráfico, convexidade é só uma
        fórmula decorada.
        """
        liquidacao = liquidacao or curva.data
        base = self.valor(curva, liquidacao)
        dv01 = self.dv01(curva, liquidacao)
        gamma = sum(
            p.quantidade * p.titulo.gamma_pu(curva, liquidacao) for p in self.posicoes
        )

        linhas = []
        for ch in choques_bps:
            verdade = self.valor(curva.deslocar(float(ch)), liquidacao) - base
            linear = dv01 * ch
            segunda = linear + 0.5 * gamma * ch**2
            linhas.append(
                {
                    "choque_bps": float(ch),
                    "full_repricing": round(verdade, 2),
                    "dv01": round(linear, 2),
                    "dv01_mais_convexidade": round(segunda, 2),
                    "erro_dv01": round(linear - verdade, 2),
                    "erro_com_convexidade": round(segunda - verdade, 2),
                }
            )
        return pd.DataFrame(linhas)
