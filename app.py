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

from carteira import Carteira, VERTICES_PADRAO, hedge_com_di1  # noqa: E402
from cenarios import (  # noqa: E402
    catalogo_nomeado,
    cenario_historico,
    ranking_por_impacto,
)
from curvas import FonteDI1AoVivo, FonteDI1Historico  # noqa: E402
from risco import (  # noqa: E402
    backtest_var,
    decompor_pca,
    var_historico,
    var_monte_carlo,
    var_parametrico,
    variacoes_historicas,
)
from titulos import Titulo  # noqa: E402
from visual import (  # noqa: E402
    grafico_aproximacao,
    grafico_backtest,
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


ARQUIVO_HISTORICO = RAIZ / "data" / "di1_ajustes.csv"


def _versao_historico() -> float:
    """Muda quando o arquivo muda. Entra na chave dos caches: sem isso o
    servidor continua servindo o histórico da primeira carga mesmo depois
    de um arquivo novo ser publicado (visto no Streamlit Cloud em 28/09)."""
    return ARQUIVO_HISTORICO.stat().st_mtime if ARQUIVO_HISTORICO.exists() else 0.0


@st.cache_resource
def _historico(versao: float) -> FonteDI1Historico:
    return FonteDI1Historico(ARQUIVO_HISTORICO)


def historico() -> FonteDI1Historico:
    return _historico(_versao_historico())


@st.cache_data(ttl=3600)
def _datas(versao: float) -> list[date]:
    return historico().datas_disponiveis()


def datas_disponiveis() -> list[date]:
    return _datas(_versao_historico())


def curva_em(d: date):
    return historico().curva(d)


def dia_disponivel(d: date) -> date:
    """Último pregão do histórico até `d` — o calendário deixa escolher feriado."""
    return max(x for x in datas_disponiveis() if x <= d)


@st.cache_data(ttl=60, show_spinner="Buscando DI1 na B3…")
def mercado_ao_vivo():
    """Contratos, curva de agora e curva do ajuste anterior — num só pedido à B3."""
    fonte = FonteDI1AoVivo()
    df = fonte.contratos()
    return df, df.attrs["horario"], fonte.curva_ao_vivo(df), fonte.curva_fechamento_anterior(df)


@st.cache_data(ttl=3600)
def _variacoes(versao: float):
    return variacoes_historicas(historico(), VERTICES_PADRAO)


def variacoes():
    return _variacoes(_versao_historico())


@st.cache_data(ttl=3600, show_spinner="Varrendo o histórico…")
def ranking_historico(carteira_df: pd.DataFrame, _curva, horizonte: int, chave: tuple):
    # `chave` identifica a curva (data, fonte, soma das taxas): a curva ao vivo
    # muda a cada minuto e o ranking tem que acompanhar.
    return ranking_por_impacto(montar_carteira(carteira_df), _curva, historico(), janela=horizonte)


def montar_carteira(df: pd.DataFrame) -> Carteira:
    c = Carteira("book de juros")
    for _, r in df.iterrows():
        if pd.isna(r["quantidade"]) or r["quantidade"] == 0:
            continue
        venc = pd.Timestamp(r["vencimento"]).date()
        papel = {"LTN": Titulo.ltn, "NTN-F": Titulo.ntnf, "DI1": Titulo.di1}[r["tipo"]](venc)
        c.adicionar(papel, float(r["quantidade"]))
    return c



_R = lambda rotulo: st.column_config.NumberColumn(rotulo, format="%,.0f")  # noqa: E731
FORMATOS = {
    "PL": _R("P&L (R$)"), "PL_%": st.column_config.NumberColumn("P&L %", format="%.3f"),
    "estimativa_dv01": _R("estimativa DV01"), "erro_dv01": _R("erro do DV01"),
    "carrego": _R("carrego"), "efeito_taxa": _R("efeito de taxa"), "total": _R("total"),
    "VaR": _R("VaR (R$)"), "ES": _R("ES (R$)"), "DV01": _R("DV01"),
    "n": st.column_config.NumberColumn("n", format="%,d"),
    "max_abs_bps": st.column_config.NumberColumn("maior mov. (bps)", format="%.1f"),
    "curta_bps": st.column_config.NumberColumn("6 meses (bps)", format="%+.1f"),
    "longa_bps": st.column_config.NumberColumn("10 anos (bps)", format="%+.1f"),
    "variacao_bps": st.column_config.NumberColumn("variação (bps)", format="%+.2f"),
    "taxa_inicial_%": st.column_config.NumberColumn("taxa inicial %", format="%.3f"),
    "taxa_final_%": st.column_config.NumberColumn("taxa final %", format="%.3f"),
    "taxa_%": st.column_config.NumberColumn("taxa %", format="%.2f"),
    "de": st.column_config.DateColumn("de", format="DD/MM/YYYY"),
    "para": st.column_config.DateColumn("para", format="DD/MM/YYYY"),
    "cenario": st.column_config.TextColumn("cenário"),
}


def tabela(df: pd.DataFrame, **kwargs) -> None:
    """st.dataframe com rótulo e formato padronizados por nome de coluna."""
    config = {c: FORMATOS[c] for c in df.columns if c in FORMATOS}
    config.update(kwargs.pop("column_config", {}))
    st.dataframe(df, use_container_width=True, hide_index=True, column_config=config, **kwargs)

# --- barra lateral --------------------------------------------------------

with st.sidebar:
    st.markdown("### Carteira")
    if "carteira_base" not in st.session_state:
        st.session_state["carteira_base"] = CARTEIRA_PADRAO
    editado = st.data_editor(
        st.session_state["carteira_base"],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "tipo": st.column_config.SelectboxColumn("tipo", options=["LTN", "NTN-F", "DI1"]),
            "vencimento": st.column_config.DateColumn("vencimento", format="DD/MM/YYYY"),
            "quantidade": st.column_config.NumberColumn(
                "quantidade", format="%d",
                help="Negativa = posição vendida. DI1 em contratos: positivo = "
                "comprado em PU (dado em taxa), negativo = vendido em PU (tomado).",
            ),
        },
        key="carteira",
    )
    if st.button("restaurar carteira de exemplo", use_container_width=True):
        st.session_state["carteira_base"] = CARTEIRA_PADRAO
        st.session_state.pop("carteira", None)
        st.rerun()

    st.markdown("### Curva")
    if not historico().caminho.exists():
        st.error(
            "Falta o histórico de ajustes do DI1 (`data/di1_ajustes.csv`). "
            "Gerar com `./.venv/bin/python src/baixar_historico_di1.py`."
        )
        st.stop()
    datas = datas_disponiveis()
    modo = st.radio(
        "marcação", ["Ao vivo (DI1)", "Fechamento histórico"], horizontal=True
    )
    ao_vivo = None
    if modo == "Ao vivo (DI1)":
        try:
            ao_vivo = mercado_ao_vivo()
        except Exception as erro:  # rede, B3 fora do ar, formato mudou
            st.error(f"Cotação da B3 indisponível ({erro}). Usando o último fechamento.")
    if ao_vivo is not None:
        contratos, horario, curva_viva, curva_ajuste = ao_vivo
        st.caption(
            f"Futuros de DI1 da B3, {len(curva_viva.dias_uteis)} vencimentos. "
            f"Cotação de {horario:%d/%m %H:%M}, com o atraso de ~15 min do dado "
            f"público. Atualiza a cada 60 s."
        )
        if contratos.attrs.get("ajuste_de_hoje"):
            st.caption(
                "Pregão encerrado: a B3 já publicou o ajuste de hoje. O P&L do dia "
                f"parte da curva oficial de {curva_ajuste.data:%d/%m}."
            )
        if st.button("atualizar agora", use_container_width=True):
            mercado_ao_vivo.clear()
            st.rerun()
    else:
        data_ref = dia_disponivel(st.date_input(
            "data de referência", value=datas[-1], min_value=datas[0],
            max_value=datas[-1], format="DD/MM/YYYY",
        ))
    st.caption(
        f"Histórico (cenários, decomposição, risco): ajustes diários do DI1, "
        f"{len(datas):,} pregões ({datas[0]:%m/%Y} a {datas[-1]:%d/%m/%Y})."
    )

    st.markdown("### Cenários")
    intensidade = st.slider("intensidade (bps)", 10, 200, 50, step=5)

curva = curva_viva if ao_vivo is not None else curva_em(data_ref)
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
a.metric(
    "Valor da carteira", f"R$ {carteira.valor_aplicado(curva):,.0f}",
    help="Capital aplicado em títulos. Futuro de DI1 não tem desembolso — "
    "entra no risco e no P&L, não no valor.",
)
b.metric("DV01", f"R$ {dv01:,.0f}", help="Variação de valor por 1bp de choque paralelo")
c.metric("Duration média", f"{np.average(resumo['duration'], weights=resumo['valor'].abs()):.2f} anos")
if ao_vivo is not None:
    pl_dia = carteira.decompor_pl(curva_ajuste, curva_viva)
    d.metric(
        "P&L do dia",
        f"R$ {pl_dia['total']:,.0f}",
        help=f"Do ajuste de {curva_ajuste.data:%d/%m} até a cotação de {horario:%H:%M}. "
        "Detalhe na aba Decomposição de P&L.",
    )
else:
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
        if ao_vivo is not None:
            st.plotly_chart(
                grafico_curva(
                    curva_ajuste, curva_viva, rotulo_comparacao="agora",
                    rotulo_base=f"ajuste {curva_ajuste.data:%d/%m}",
                ),
                use_container_width=True,
            )
        else:
            st.plotly_chart(grafico_curva(curva), use_container_width=True)

    if ao_vivo is not None:
        with st.expander(f"Contratos de DI1 usados na curva ({len(contratos)})"):
            tabela_di1 = contratos[
                ["contrato", "vencimento", "compra", "venda", "ultimo",
                 "ajuste_anterior", "taxa", "origem", "contratos_negociados"]
            ].assign(variacao_bps=lambda t: (t["taxa"] - t["ajuste_anterior"]) * 100)
            st.dataframe(
                tabela_di1, use_container_width=True, hide_index=True,
                column_config={
                    "vencimento": st.column_config.DateColumn(format="DD/MM/YYYY"),
                    "variacao_bps": st.column_config.NumberColumn(format="%+.1f"),
                    "contratos_negociados": st.column_config.NumberColumn(format="%,d"),
                },
            )
            st.caption(
                "Taxa do vértice = meio entre compra e venda. Sem livro dos dois "
                "lados, último negócio; sem negócio no dia, ajuste anterior. O "
                "último negócio de um vencimento ilíquido pode ter horas — o meio "
                "acompanha a curva, o último congela o vértice num preço velho."
            )

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

    st.markdown("##### Hedge com futuros de DI1")
    if ao_vivo is None:
        st.info("O hedge escolhe contratos pela liquidez do pregão — use a marcação ao vivo.")
    else:
        ordens, efeito = hedge_com_di1(
            carteira, curva, dict(zip(contratos["vencimento"], contratos["contratos_em_aberto"]))
        )
        if ordens.empty:
            st.success("A carteira já está com o DV01 zerado vértice a vértice.")
        else:
            esq, dir_ = st.columns([3, 2])
            with esq:
                st.dataframe(
                    ordens, use_container_width=True, hide_index=True,
                    column_config={
                        "vencimento": st.column_config.DateColumn(format="MM/YYYY"),
                        "contratos": st.column_config.NumberColumn(format="%+d"),
                        "DV01_hedge": st.column_config.NumberColumn("DV01 do hedge", format="%,.0f"),
                    },
                )
                if st.button("adicionar hedge à carteira"):
                    novas = pd.DataFrame({
                        "tipo": "DI1",
                        "vencimento": ordens["vencimento"],
                        "quantidade": ordens["contratos"],
                    })
                    st.session_state["carteira_base"] = pd.concat(
                        [editado, novas], ignore_index=True
                    )
                    st.session_state.pop("carteira", None)
                    st.rerun()
            with dir_:
                st.dataframe(
                    efeito, use_container_width=True, hide_index=True,
                    column_config={
                        "DV01_antes": st.column_config.NumberColumn("DV01 antes", format="%,.0f"),
                        "DV01_depois": st.column_config.NumberColumn("DV01 depois", format="%,.0f"),
                    },
                )
            st.caption(
                "Zera a key rate duration inteira, não só o DV01 total: um hedge "
                "só paralelo deixaria a carteira exposta a inclinação. Para cada "
                "vértice, o contrato mais negociado (em aberto) com prazo a até "
                "25% do vértice. O que sobra no 'depois' é arredondamento para "
                "contrato inteiro."
            )

# --- cenários -------------------------------------------------------------

with cenarios_tab:
    lista = list(catalogo_nomeado(float(intensidade)).values())

    horizonte = st.radio(
        "horizonte dos cenários históricos", [1, 5], horizontal=True,
        format_func=lambda h: "1 dia" if h == 1 else "5 dias (uma semana de estresse)",
    )
    ranking = ranking_historico(
        editado, curva, horizonte, (curva.data, curva.fonte, float(curva.taxas.sum()), _versao_historico())
    )
    if not ranking.empty:
        pior = ranking.iloc[0]
        lista.append(
            cenario_historico(
                curva_em(pior["de"]),
                curva_em(pior["para"]),
                rotulo=f"Repetir {pior['para']:%d/%m/%Y}",
            )
        )

    tabela_cen = carteira.rodar_cenarios(curva, lista)

    esq, dir_ = st.columns([2, 3])
    with esq:
        tabela(tabela_cen.drop(columns=["origem"]))
    with dir_:
        st.plotly_chart(
            grafico_divergente(list(tabela_cen["cenario"]), list(tabela_cen["PL"]), "P&L por cenário"),
            use_container_width=True,
        )
    st.caption(
        "P&L por reprecificação completa, não por DV01. A coluna erro_dv01 mostra "
        "quanto a aproximação linear teria errado em cada cenário."
    )

    st.markdown("#### Dias históricos, ordenados pelo impacto nesta carteira")
    tabela(ranking.head(8))
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
    if ao_vivo is not None:
        st.markdown(f"##### Hoje: ajuste de {curva_ajuste.data:%d/%m} → cotação das {horario:%H:%M}")
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Carrego", f"R$ {pl_dia['carrego']:,.0f}")
        h2.metric("Efeito de taxa", f"R$ {pl_dia['efeito_taxa']:,.0f}")
        h3.metric("Caixa (cupom / CDI)", f"R$ {pl_dia['caixa_recebido']:,.0f}")
        h4.metric("Total", f"R$ {pl_dia['total']:,.0f}")
        esq, dir_ = st.columns([1, 1])
        with esq:
            tabela(pl_dia["por_vertice"].drop(columns=["taxa_inicial_%", "taxa_final_%"]))
        with dir_:
            tabela(pl_dia["por_papel"])
        st.caption(
            "Mesma decomposição do histórico, aplicada ao pregão em andamento: "
            "o ponto de partida é o ajuste de ontem, que é onde a mesa foi marcada."
        )
        st.divider()
        st.markdown("##### Entre dois fechamentos")

    if len(datas) < 2:
        st.info("Histórico insuficiente para decompor P&L entre duas datas.")
    else:
        col1, col2 = st.columns(2)
        d0 = dia_disponivel(col1.date_input(
            "de", value=datas[-21], min_value=datas[0], max_value=datas[-2],
            format="DD/MM/YYYY",
        ))
        d1 = dia_disponivel(col2.date_input(
            "para", value=datas[-1], min_value=datas[1], max_value=datas[-1],
            format="DD/MM/YYYY",
        ))
        if d1 <= d0:
            st.warning("A data final precisa ser posterior à inicial.")
            st.stop()

        r = carteira.decompor_pl(curva_em(d0), curva_em(d1))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Carrego", f"R$ {r['carrego']:,.0f}")
        m2.metric("Efeito de taxa", f"R$ {r['efeito_taxa']:,.0f}")
        m3.metric("Caixa (cupom / CDI)", f"R$ {r['caixa_recebido']:,.0f}")
        m4.metric("Total", f"R$ {r['total']:,.0f}")

        st.caption(
            "Carrego reprecifica na curva antiga com o prazo novo: é o que a "
            "carteira renderia com a curva parada, e engloba roll-down. As duas "
            "parcelas somam o total exatamente, sem resíduo."
        )

        esq, dir_ = st.columns([1, 1])
        with esq:
            st.markdown("##### Efeito de taxa atribuído por vértice")
            tabela(r["por_vertice"].drop(columns=["taxa_inicial_%", "taxa_final_%"]))
            st.caption(
                "A linha 'não linear' é convexidade e movimento entre vértices. "
                "Aparece explícita em vez de diluída nos vértices."
            )
        with dir_:
            st.markdown("##### Por papel")
            tabela(r["por_papel"])

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
        tabela(tabela_risco)
        st.caption(
            f"{len(var_df):,} variações diárias do DI1 desde {var_df.index[0]:%m/%Y}. "
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

    st.markdown("##### O VaR funciona? Backtest")
    try:
        serie_bt, resumo_bt = backtest_var(carteira, curva, var_df, confianca, janela=500)
    except ValueError as erro:
        st.info(str(erro))
    else:
        tabela(resumo_bt)
        metodo_bt = st.radio(
            "método", ["histórico", "paramétrico"], horizontal=True, key="metodo_bt"
        )
        coluna = "VaR_historico" if metodo_bt == "histórico" else "VaR_parametrico"
        st.plotly_chart(
            grafico_backtest(serie_bt, coluna, f"P&L diário × VaR {confianca:.0%} ({metodo_bt})"),
            use_container_width=True,
        )
        st.caption(
            "Para cada dia, o VaR é estimado só com os 500 pregões anteriores e "
            "comparado com o P&L que esta carteira teria no movimento real do dia. "
            "Kupiec testa se a taxa de exceções é compatível com a prometida; o "
            "semáforo de Basileia olha os últimos 250 dias (verde até 4 exceções)."
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

rodape_curva = (
    f"Curva dos futuros de DI1 da B3, cotação de {horario:%d/%m/%Y %H:%M}, "
    f"{len(curva.dias_uteis)} vencimentos; conferida contra a curva oficial da B3 "
    f"pelo ajuste do pregão anterior."
    if ao_vivo is not None
    else f"Curva DI×pré da B3 em {curva.data:%d/%m/%Y}, {len(curva.dias_uteis)} vértices."
)
st.divider()
st.markdown(
    f"""<div class="rodape">
    {rodape_curva}
    Calendário de dias úteis conferido contra os 269 vértices publicados pela B3:
    divergência zero. Apreçamento conferido contra os PUs oficiais do Tesouro
    Nacional: reconstrói ao centavo em 19 anos de dados.
    </div>""",
    unsafe_allow_html=True,
)
