"""Simulador de P&L e Risco — Carteira de Juros.

Rodar:  ./.venv/bin/streamlit run app.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from carteira import Carteira, VERTICES_PADRAO  # noqa: E402
from cenarios import (  # noqa: E402
    catalogo_nomeado,
    cenario_historico,
    ranking_por_impacto,
)
from curvas import FonteB3  # noqa: E402
from risco import (  # noqa: E402
    decompor_pca,
    var_historico,
    var_monte_carlo,
    var_parametrico,
    variacoes_historicas,
)
from titulos import Titulo  # noqa: E402
from visual import (  # noqa: E402
    grafico_aproximacao,
    grafico_curva,
    grafico_distribuicao,
    grafico_divergente,
)

st.set_page_config(
    page_title="Simulador de Juros", layout="wide", initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-feature-settings: "tnum" 1; }
      [data-testid="stMetricValue"] {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 1.45rem;
      }
      [data-testid="stMetricLabel"] { text-transform: uppercase; letter-spacing: .06em; }
      .stDataFrame { font-variant-numeric: tabular-nums; }
      .rodape { color: #52514e; font-size: .82rem; line-height: 1.5; }
    </style>
    """,
    unsafe_allow_html=True,
)

CARTEIRA_PADRAO = pd.DataFrame(
    [
        {"tipo": "LTN", "vencimento": date(2028, 1, 1), "quantidade": 50_000},
        {"tipo": "LTN", "vencimento": date(2031, 1, 1), "quantidade": -30_000},
        {"tipo": "NTN-F", "vencimento": date(2035, 1, 1), "quantidade": 20_000},
    ]
)


@st.cache_resource
def fonte_b3() -> FonteB3:
    return FonteB3("PRE")


@st.cache_data(ttl=3600)
def datas_disponiveis() -> list[date]:
    return sorted(fonte_b3().datas_disponiveis())


@st.cache_data(ttl=3600)
def curva_em(d: date):
    return fonte_b3().curva(d)


@st.cache_data(ttl=3600)
def variacoes():
    return variacoes_historicas(fonte_b3(), VERTICES_PADRAO)


def montar_carteira(df: pd.DataFrame) -> Carteira:
    c = Carteira("book de juros")
    for _, r in df.iterrows():
        if pd.isna(r["quantidade"]) or r["quantidade"] == 0:
            continue
        venc = pd.Timestamp(r["vencimento"]).date()
        papel = Titulo.ltn(venc) if r["tipo"] == "LTN" else Titulo.ntnf(venc)
        c.adicionar(papel, float(r["quantidade"]))
    return c


# --- barra lateral --------------------------------------------------------

with st.sidebar:
    st.markdown("### Carteira")
    editado = st.data_editor(
        CARTEIRA_PADRAO,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "tipo": st.column_config.SelectboxColumn("tipo", options=["LTN", "NTN-F"]),
            "vencimento": st.column_config.DateColumn("vencimento", format="DD/MM/YYYY"),
            "quantidade": st.column_config.NumberColumn(
                "quantidade", format="%d", help="Negativa = posição vendida"
            ),
        },
        key="carteira",
    )

    st.markdown("### Curva")
    datas = datas_disponiveis()
    data_ref = st.selectbox(
        "data de referência",
        options=list(reversed(datas)),
        format_func=lambda d: d.strftime("%d/%m/%Y"),
    )
    st.caption(
        f"Fonte: B3, taxas referenciais. Histórico disponível: {len(datas)} dias "
        f"úteis ({datas[0]:%d/%m} a {datas[-1]:%d/%m})."
    )

    st.markdown("### Cenários")
    intensidade = st.slider("intensidade (bps)", 10, 200, 50, step=5)

curva = curva_em(data_ref)
carteira = montar_carteira(editado)

if not carteira.posicoes:
    st.warning("Adicione ao menos uma posição na carteira.")
    st.stop()

# --- cabeçalho ------------------------------------------------------------

st.markdown("## Simulador de P&L e Risco — Carteira de Juros")

valor = carteira.valor(curva)
dv01 = carteira.dv01(curva)
resumo = carteira.resumo(curva)

a, b, c, d = st.columns(4)
a.metric("Valor da carteira", f"R$ {valor:,.0f}")
b.metric("DV01", f"R$ {dv01:,.0f}", help="Variação de valor por 1bp de choque paralelo")
c.metric("Duration média", f"{np.average(resumo['duration'], weights=resumo['valor'].abs()):.2f} anos")
d.metric("Posições", f"{len(carteira.posicoes)}")

marcacao, cenarios_tab, decomp, risco_tab = st.tabs(
    ["Marcação", "Cenários", "Decomposição de P&L", "Risco"]
)

# --- marcação -------------------------------------------------------------

with marcacao:
    esq, dir_ = st.columns([3, 2])
    with esq:
        st.dataframe(
            resumo,
            use_container_width=True,
            hide_index=True,
            column_config={
                "quantidade": st.column_config.NumberColumn(format="%,d"),
                "PU": st.column_config.NumberColumn(format="%.2f"),
                "valor": st.column_config.NumberColumn(format="%,.2f"),
                "taxa_%": st.column_config.NumberColumn(format="%.3f"),
                "duration": st.column_config.NumberColumn(format="%.2f"),
                "DV01": st.column_config.NumberColumn(format="%,.2f"),
                "convexidade": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.caption(
            "PU reconstruído pela curva, fluxo a fluxo, em base 252 dias úteis. "
            "O apreçador reproduz os PUs oficiais do Tesouro Nacional ao centavo."
        )
    with dir_:
        st.plotly_chart(grafico_curva(curva), use_container_width=True)

    st.plotly_chart(
        grafico_divergente(
            [f"{v} du" for v in VERTICES_PADRAO],
            [carteira.dv01_por_vertice(curva)[v] for v in VERTICES_PADRAO],
            "DV01 por vértice (key rate duration)",
        ),
        use_container_width=True,
    )
    st.caption(
        "Bump em tenda: o choque vale no vértice e zera nos vizinhos. A soma "
        "reproduz o DV01 paralelo — é o que torna flattening e steepening "
        "representáveis, em vez de só choque paralelo."
    )

# --- cenários -------------------------------------------------------------

with cenarios_tab:
    lista = list(catalogo_nomeado(float(intensidade)).values())

    ranking = ranking_por_impacto(carteira, curva, fonte_b3())
    if not ranking.empty:
        pior = ranking.iloc[0]
        lista.append(
            cenario_historico(
                curva_em(pior["de"]),
                curva_em(pior["para"]),
                rotulo=f"Repetir {pior['para']:%d/%m/%Y}",
            )
        )

    tabela = carteira.rodar_cenarios(curva, lista)

    esq, dir_ = st.columns([2, 3])
    with esq:
        st.dataframe(tabela, use_container_width=True, hide_index=True)
    with dir_:
        st.plotly_chart(
            grafico_divergente(list(tabela["cenario"]), list(tabela["PL"]), "P&L por cenário"),
            use_container_width=True,
        )
    st.caption(
        "P&L por reprecificação completa, não por DV01. A coluna erro_dv01 mostra "
        "quanto a aproximação linear teria errado em cada cenário."
    )

    st.markdown("#### Dias históricos, ordenados pelo impacto nesta carteira")
    st.dataframe(ranking.head(8), use_container_width=True, hide_index=True)
    st.caption(
        "Ordenado por P&L, não por variação em bps. O dia que mais move a curva "
        "não é o que mais dói: um choque de 25bps no overnight quase não tem DV01."
    )

    escolhido = st.selectbox("visualizar cenário", [c.nome for c in lista])
    cen = next(c for c in lista if c.nome == escolhido)
    st.plotly_chart(
        grafico_curva(curva, cen(curva), rotulo_comparacao=escolhido),
        use_container_width=True,
    )
    st.caption(cen.descricao)

# --- decomposição ---------------------------------------------------------

with decomp:
    if len(datas) < 2:
        st.info("Histórico insuficiente para decompor P&L entre duas datas.")
    else:
        col1, col2 = st.columns(2)
        d0 = col1.selectbox("de", datas[:-1], format_func=lambda d: d.strftime("%d/%m/%Y"))
        posteriores = [d for d in datas if d > d0]
        d1 = col2.selectbox(
            "para", posteriores, index=len(posteriores) - 1,
            format_func=lambda d: d.strftime("%d/%m/%Y"),
        )

        r = carteira.decompor_pl(curva_em(d0), curva_em(d1))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Carrego", f"R$ {r['carrego']:,.0f}")
        m2.metric("Efeito de taxa", f"R$ {r['efeito_taxa']:,.0f}")
        m3.metric("Caixa recebido", f"R$ {r['caixa_recebido']:,.0f}")
        m4.metric("Total", f"R$ {r['total']:,.0f}")

        st.caption(
            "Carrego reprecifica na curva antiga com o prazo novo: é o que a "
            "carteira renderia com a curva parada, e engloba roll-down. As duas "
            "parcelas somam o total exatamente, sem resíduo."
        )

        esq, dir_ = st.columns([3, 2])
        with esq:
            st.markdown("##### Efeito de taxa atribuído por vértice")
            st.dataframe(r["por_vertice"], use_container_width=True, hide_index=True)
            st.caption(
                "A linha 'não linear' é convexidade e movimento entre vértices. "
                "Aparece explícita em vez de diluída nos vértices."
            )
        with dir_:
            st.markdown("##### Por papel")
            st.dataframe(r["por_papel"], use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("##### Onde a aproximação linear quebra")
    choques = np.array([-300, -200, -150, -100, -50, -25, -10, 10, 25, 50, 100, 150, 200, 300])
    st.plotly_chart(
        grafico_aproximacao(carteira.aproximar_vs_reprecificar(curva, choques)),
        use_container_width=True,
    )
    st.caption(
        "A reta do DV01 descola da verdade conforme o choque cresce, e sempre "
        "para o mesmo lado: subestima ganho e superestima perda. O termo de "
        "segunda ordem recupera quase todo o desvio. É por isso que o simulador "
        "reprecifica em vez de multiplicar DV01."
    )

# --- risco ----------------------------------------------------------------

with risco_tab:
    var_df = variacoes()
    confianca = st.select_slider("confiança", options=[0.90, 0.95, 0.99], value=0.99)

    hist, pl_hist = var_historico(carteira, curva, var_df, confianca)
    par = var_parametrico(carteira, curva, var_df, confianca)
    mc, pl_mc = var_monte_carlo(carteira, curva, var_df, confianca)

    if hist.aviso:
        st.error(
            f"**Amostra insuficiente — os números abaixo são ilustrativos.** {hist.aviso} "
            "Com histórico longo de curva, os mesmos cálculos passam a ser utilizáveis."
        )

    tabela_risco = pd.DataFrame(
        [
            {
                "método": r.metodo,
                "VaR": round(r.var, 2),
                "ES": round(r.es, 2),
                "n": r.n_observacoes,
                "confiável": "não" if r.aviso else "sim",
            }
            for r in (hist, par, mc)
        ]
    )
    esq, dir_ = st.columns([2, 3])
    with esq:
        st.dataframe(tabela_risco, use_container_width=True, hide_index=True)
        st.caption(
            "Histórico não assume distribuição. Paramétrico é gaussiano e linear. "
            "Monte Carlo é gaussiano mas reprecifica. A distância entre eles separa "
            "o custo da hipótese de distribuição do custo da linearização."
        )
    with dir_:
        if len(pl_mc):
            st.plotly_chart(
                grafico_distribuicao(pl_mc, mc.var, mc.es, "P&L simulado (Monte Carlo)"),
                use_container_width=True,
            )

    st.markdown("##### Estrutura do movimento da curva (PCA)")
    pca = decompor_pca(var_df)
    if pca.empty:
        st.info("Observações insuficientes para decomposição em componentes.")
    else:
        st.dataframe(pca, use_container_width=True)
        st.caption(
            "A interpretação de cada componente é deduzida do padrão de trocas de "
            "sinal das cargas, não escrita à mão. Nível, inclinação e curvatura "
            "nessa ordem é o resultado canônico de curva de juros — se sair assim, "
            "a matriz de covariância está descrevendo mercado e não ruído."
        )

st.divider()
st.markdown(
    f"""<div class="rodape">
    Curva DI×pré da B3 em {curva.data:%d/%m/%Y}, {len(curva.dias_uteis)} vértices.
    Calendário de dias úteis conferido contra os 269 vértices publicados pela B3:
    divergência zero. Apreçamento conferido contra os PUs oficiais do Tesouro
    Nacional: reconstrói ao centavo em 19 anos de dados.
    </div>""",
    unsafe_allow_html=True,
)
