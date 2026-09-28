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

# Papel e tinta — mesmos tokens de .streamlit/config.toml.
SUPERFICIE = "#FFFFFF"
PLANO = "#ECEFF1"
TINTA = "#0E1C27"
TINTA_FRACA = "#5B6873"
GRADE = "#E6EAED"

FONTE = "Instrument Sans, system-ui, sans-serif"
FONTE_NUM = "Martian Mono, ui-monospace, monospace"
FONTE_TITULO = "Bricolage Grotesque, system-ui, sans-serif"

# Prazos com o nome que a mesa usa, em anos (eixo log).
PRAZOS = [(1 / 12, "1m"), (0.25, "3m"), (0.5, "6m"), (1, "1a"), (2, "2a"),
          (3, "3a"), (5, "5a"), (10, "10a"), (15, "15a")]


# Título à esquerda, legenda à direita: na mesma linha sem colidir.
_LEGENDA_DIREITA = dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right",
                        bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=TINTA))


def _legenda_direita(fig: go.Figure) -> None:
    fig.update_layout(showlegend=True, legend=_LEGENDA_DIREITA)


def _base(titulo: str = "", altura: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=titulo, font=dict(family=FONTE_TITULO, size=15, color=TINTA),
                   x=0.01, xanchor="left"),
        height=altura,
        margin=dict(l=8, r=8, t=40 if titulo else 12, b=8),
        paper_bgcolor=SUPERFICIE,
        plot_bgcolor=SUPERFICIE,
        font=dict(family=FONTE, size=12, color=TINTA_FRACA),
        hoverlabel=dict(font=dict(family=FONTE_NUM, size=11), bgcolor=SUPERFICIE,
                        bordercolor=GRADE),
        showlegend=False,
        xaxis=dict(gridcolor=GRADE, zerolinecolor=GRADE, linecolor=GRADE,
                   tickfont=dict(family=FONTE_NUM, size=11)),
        yaxis=dict(gridcolor=GRADE, zerolinecolor=GRADE, linecolor=GRADE,
                   tickfont=dict(family=FONTE_NUM, size=11)),
    )
    return fig


def grafico_curva(
    curva, comparacao=None, rotulo_comparacao: str = "cenário", rotulo_base: str = "hoje"
) -> go.Figure:
    """Curva de juros por prazo. Duas séries no máximo: base e cenário."""
    fig = _base("Curva DI × pré")
    du = curva.dias_uteis
    anos = du / 252

    fig.add_trace(
        go.Scatter(
            x=anos, y=curva.taxas * 100, mode="lines", name=rotulo_base,
            line=dict(color=TINTA, width=2),
            hovertemplate="%{x:.2f} anos<br>%{y:.3f}%"
                          f"<extra>{rotulo_base}</extra>",
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
        _legenda_direita(fig)

    _eixo_prazo(fig)
    fig.update_yaxes(title="taxa a.a. (%)")
    return fig


def _eixo_prazo(fig: go.Figure, titulo: str = "") -> None:
    fig.update_xaxes(
        type="log", title=titulo,
        tickvals=[a for a, _ in PRAZOS], ticktext=[r for _, r in PRAZOS],
        tickfont=dict(family=FONTE_NUM, size=11), showgrid=False,
    )


def grafico_hero(base, agora, rotulo_base: str, rotulo_agora: str,
                 vertices_anos: list[float]) -> go.Figure:
    """A curva de referência contra a de agora, em largura total.

    Cada ponto da curva de agora é um contrato negociado — os marcadores
    mostram onde há preço de verdade e onde a curva é interpolação. As guias
    verticais marcam os vértices de risco, os mesmos da fita logo abaixo.
    """
    fig = _base("", altura=300)
    fig.update_layout(margin=dict(l=8, r=8, t=34, b=8), plot_bgcolor=SUPERFICIE)
    for anos in vertices_anos:
        fig.add_vline(x=anos, line=dict(color=GRADE, width=1, dash="dot"))

    fig.add_trace(go.Scatter(
        x=base.dias_uteis / 252, y=base.taxas * 100, mode="lines", name=rotulo_base,
        line=dict(color="#9AA6AF", width=1.6),
        hovertemplate="%{x:.2f}a · %{y:.3f}%" f"<extra>{rotulo_base}</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=agora.dias_uteis / 252, y=agora.taxas * 100, mode="lines+markers",
        name=rotulo_agora, line=dict(color=TINTA, width=2.4),
        marker=dict(size=4.5, color=TINTA),
        hovertemplate="%{x:.2f}a · %{y:.3f}%" f"<extra>{rotulo_agora}</extra>",
    ))
    fig.update_layout(showlegend=True, legend=dict(
        orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONTE, size=12, color=TINTA)))
    _eixo_prazo(fig)
    fig.update_yaxes(ticksuffix="%", tickfont=dict(family=FONTE_NUM, size=11),
                     tickformat=".2f", nticks=6)
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
        legend=_LEGENDA_DIREITA,
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


def grafico_backtest(serie, coluna_var: str, titulo: str) -> go.Figure:
    """P&L diário contra o VaR estimado na véspera; exceções destacadas.

    Os pontos cinza são os dias comuns; os vermelhos, os dias em que a perda
    passou do VaR. Um VaR bem calibrado deixa ~1% dos pontos abaixo da linha.
    """
    fig = _base(titulo, altura=360)
    excecao = serie["PL"] < -serie[coluna_var]
    comuns, rompidos = serie[~excecao], serie[excecao]

    fig.add_trace(go.Scattergl(
        x=comuns.index, y=comuns["PL"], mode="markers", name="P&L do dia",
        marker=dict(color=TINTA_FRACA, size=3, opacity=0.45),
        hovertemplate="%{x|%d/%m/%Y}<br>R$ %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=serie.index, y=-serie[coluna_var], mode="lines", name="−VaR 99%",
        line=dict(color=AZUL, width=1.5),
        hovertemplate="%{x|%d/%m/%Y}<br>VaR R$ %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=rompidos.index, y=rompidos["PL"], mode="markers", name="exceção",
        marker=dict(color=NEGATIVO, size=7),
        hovertemplate="%{x|%d/%m/%Y}<br>R$ %{y:,.0f}<extra>exceção</extra>",
    ))
    _legenda_direita(fig)
    fig.update_yaxes(title="P&L (R$)")
    return fig
