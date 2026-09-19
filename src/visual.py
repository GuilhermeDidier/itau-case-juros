"""Paleta e gráficos.

A paleta é a de referência do método de visualização, validada nos seis
gates (banda de luminosidade, piso de croma, separação para daltonismo,
piso de visão normal, contraste). Não foi escolhida no olho.

Regras seguidas aqui e que valem para todo gráfico do app:
  - no máximo três séries categóricas, na ordem fixa dos slots
  - divergente é azul↔vermelho com cinza no meio; nunca arco-íris
  - um eixo por gráfico, nunca dois eixos y
  - cor segue a entidade, nunca a posição no ranking
  - texto em tinta de texto, nunca na cor da série
"""

from __future__ import annotations

import plotly.graph_objects as go

# Slots categóricos, ordem fixa.
AZUL, LARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# Par divergente: polos quente/frio com neutro no meio.
POSITIVO, NEGATIVO, NEUTRO = "#2a78d6", "#e34948", "#f0efec"

SUPERFICIE = "#fcfcfb"
PLANO = "#f9f9f7"
TINTA = "#0b0b0b"
TINTA_FRACA = "#52514e"
GRADE = "#e6e5e1"

FONTE = "ui-monospace, SFMono-Regular, Menlo, monospace"


def _base(titulo: str = "", altura: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=14, color=TINTA), x=0, xanchor="left"),
        height=altura,
        margin=dict(l=8, r=8, t=40 if titulo else 12, b=8),
        paper_bgcolor=SUPERFICIE,
        plot_bgcolor=SUPERFICIE,
        font=dict(family=FONTE, size=12, color=TINTA_FRACA),
        hoverlabel=dict(font=dict(family=FONTE, size=12)),
        showlegend=False,
        xaxis=dict(gridcolor=GRADE, zerolinecolor=GRADE, linecolor=GRADE),
        yaxis=dict(gridcolor=GRADE, zerolinecolor=GRADE, linecolor=GRADE),
    )
    return fig


def grafico_curva(curva, comparacao=None, rotulo_comparacao: str = "cenário") -> go.Figure:
    """Curva de juros por prazo. Duas séries no máximo: base e cenário."""
    fig = _base("Curva DI × pré")
    du = curva.dias_uteis
    anos = du / 252

    fig.add_trace(
        go.Scatter(
            x=anos, y=curva.taxas * 100, mode="lines", name="hoje",
            line=dict(color=AZUL, width=2),
            hovertemplate="%{x:.2f} anos<br>%{y:.3f}%<extra>hoje</extra>",
        )
    )
    if comparacao is not None:
        fig.add_trace(
            go.Scatter(
                x=comparacao.dias_uteis / 252, y=comparacao.taxas * 100,
                mode="lines", name=rotulo_comparacao,
                line=dict(color=LARANJA, width=2, dash="dot"),
                hovertemplate="%{x:.2f} anos<br>%{y:.3f}%"
                              f"<extra>{rotulo_comparacao}</extra>",
            )
        )
        fig.update_layout(showlegend=True, legend=dict(
            orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"))

    fig.update_xaxes(title="prazo (anos)", type="log")
    fig.update_yaxes(title="taxa a.a. (%)")
    return fig


def grafico_divergente(rotulos, valores, titulo: str, unidade: str = "R$") -> go.Figure:
    """Barras horizontais com polaridade: ganho e perda em polos opostos.

    Divergente porque o sinal é a informação. O rótulo fica sempre visível,
    que é a contrapartida exigida quando a cor sozinha não basta.
    """
    fig = _base(titulo, altura=max(220, 34 * len(rotulos) + 70))
    cores = [POSITIVO if v >= 0 else NEGATIVO for v in valores]

    fig.add_trace(
        go.Bar(
            x=valores, y=rotulos, orientation="h",
            marker=dict(color=cores, line=dict(color=SUPERFICIE, width=2)),
            text=[f"{unidade} {v:,.0f}" for v in valores],
            textposition="outside",
            textfont=dict(color=TINTA_FRACA, size=11, family=FONTE),
            hovertemplate="%{y}<br>" + unidade + " %{x:,.2f}<extra></extra>",
        )
    )
    # Rótulo fora da barra precisa de espaço, senão o texto é cortado na borda
    # — e texto clipado é defeito, não detalhe. A folga é proporcional ao maior
    # valor para que a escala continue honesta.
    maior = max((abs(v) for v in valores), default=1.0) or 1.0
    folga = maior * 1.55

    fig.update_xaxes(
        title=unidade, range=[-folga, folga],
        zeroline=True, zerolinewidth=2, zerolinecolor="#cfcec9",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(bargap=0.3, uniformtext=dict(mode="show", minsize=10))
    return fig


def grafico_aproximacao(df) -> go.Figure:
    """Full repricing contra as aproximações, por tamanho do choque.

    O gráfico que prova convexidade: a reta do DV01 descola da verdade
    conforme o choque cresce, e o termo de segunda ordem recupera quase tudo.
    """
    fig = _base("Aproximação contra reprecificação completa")

    series = [
        ("full_repricing", "full repricing", AZUL, "solid", 2.5),
        ("dv01_mais_convexidade", "DV01 + convexidade", AQUA, "dot", 2),
        ("dv01", "só DV01", LARANJA, "dash", 2),
    ]
    for coluna, nome, cor, traco, largura in series:
        fig.add_trace(
            go.Scatter(
                x=df["choque_bps"], y=df[coluna], mode="lines+markers", name=nome,
                line=dict(color=cor, width=largura, dash=traco),
                marker=dict(size=8, line=dict(color=SUPERFICIE, width=2)),
                hovertemplate="%{x:+.0f} bps<br>R$ %{y:,.0f}<extra>" + nome + "</extra>",
            )
        )

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(title="choque paralelo (bps)", zeroline=True,
                     zerolinewidth=2, zerolinecolor="#cfcec9")
    fig.update_yaxes(title="variação de valor (R$)")
    return fig


def grafico_distribuicao(pl, var: float, es: float, titulo: str) -> go.Figure:
    """Distribuição de P&L simulado, com VaR e ES marcados."""
    fig = _base(titulo)

    fig.add_trace(
        go.Histogram(
            x=pl, nbinsx=48,
            marker=dict(color=AZUL, line=dict(color=SUPERFICIE, width=1)),
            hovertemplate="R$ %{x:,.0f}<br>%{y} cenários<extra></extra>",
            name="P&L",
        )
    )
    # VaR e ES caem perto um do outro por construção — ES é sempre a média
    # além do VaR. Empilhar os rótulos no mesmo canto os faz colidir, então
    # cada um ganha sua própria altura.
    marcas = (
        (-var, "VaR", NEGATIVO, "top left"),
        (-es, "ES", TINTA_FRACA, "bottom left"),
    )
    for valor, nome, cor, posicao in marcas:
        fig.add_vline(
            x=valor, line=dict(color=cor, width=2, dash="dash"),
            annotation_text=f"{nome} R$ {abs(valor):,.0f}",
            annotation_position=posicao,
            annotation_font=dict(color=TINTA_FRACA, size=11, family=FONTE),
            annotation_bgcolor=SUPERFICIE,
        )

    fig.update_xaxes(title="P&L (R$)")
    fig.update_yaxes(title="cenários")
    return fig
