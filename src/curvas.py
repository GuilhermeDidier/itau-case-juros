"""Fontes de curva de juros, atrás de uma interface única.

O resto do projeto só conhece `Curva` e `FonteCurva`; de onde a curva vem é
detalhe de cada fonte. Todas são públicas e oficiais da B3.

Fontes:
  FonteDI1AoVivo    futuros de DI1, intradiário com ~15 min de atraso — a marcação
  FonteDI1Historico ajustes diários do DI1 desde 2018 (arquivos de pregão) — o risco
  FonteB3           curva referencial PRE oficial, só ~20 dias úteis — a régua que
                    confere as outras duas

Achado que custou caro: a API da B3 tem dois endpoints. `Search/GetList`
ACEITA o parâmetro de data e o IGNORA — devolve sempre a curva mais recente,
byte a byte. Só `Search/GetDownloadFile` honra a data de verdade. Usar o
primeiro produziria "cenários históricos" que são o dia de hoje disfarçado.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import requests

from calendario import BASE_252, dias_uteis, eh_dia_util, proximo_dia_util

RAIZ = Path(__file__).resolve().parent.parent
CACHE = RAIZ / "data" / "raw"

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
)


@dataclass(frozen=True)
class Curva:
    """Curva zero em base 252, indexada por dias úteis.

    `taxas` são taxas anuais efetivas em decimal (0.1365 = 13,65% a.a.).
    """

    data: date
    dias_uteis: np.ndarray  # int, crescente
    taxas: np.ndarray  # float, decimal
    nome: str = "DI x pré"
    fonte: str = ""

    def __post_init__(self):
        if len(self.dias_uteis) != len(self.taxas):
            raise ValueError("dias_uteis e taxas com tamanhos diferentes")
        if len(self.dias_uteis) < 2:
            raise ValueError("curva precisa de ao menos 2 vértices")
        if not np.all(np.diff(self.dias_uteis) > 0):
            raise ValueError("dias_uteis precisa estar estritamente crescente")

    def taxa(self, du) -> np.ndarray | float:
        """Taxa interpolada.

        Interpola linearmente no log do fator de capitalização — equivalente a
        interpolar linearmente a taxa forward contínua, que é a convenção de
        mercado para a curva de DI. Interpolar a taxa spot direto produz
        forwards serrilhados e estraga o cálculo de carrego.

        Fora do intervalo da curva a taxa é achatada na do vértice extremo.
        Sem o clip, o fator de capitalização pararia de crescer enquanto o
        prazo do denominador continuaria — e a taxa extrapolada despencaria
        para perto de zero em silêncio, inflando o PU de qualquer fluxo além
        do último vértice.
        """
        du = np.asarray(du, dtype=float)
        limite = np.clip(du, self.dias_uteis[0], self.dias_uteis[-1])
        log_fator = np.log1p(self.taxas) * (self.dias_uteis / BASE_252)
        interp = np.interp(limite, self.dias_uteis, log_fator)
        taxa = np.expm1(interp / (limite / BASE_252))
        return float(taxa) if taxa.ndim == 0 else taxa

    def desconto(self, du) -> np.ndarray | float:
        """Fator de desconto para um prazo em dias úteis."""
        du = np.asarray(du, dtype=float)
        fator = (1.0 + self.taxa(du)) ** (-du / BASE_252)
        return float(fator) if fator.ndim == 0 else fator

    def deslocar(self, choque_bps: float) -> "Curva":
        """Choque paralelo, em pontos-base."""
        return Curva(
            data=self.data,
            dias_uteis=self.dias_uteis,
            taxas=self.taxas + choque_bps / 10_000.0,
            nome=f"{self.nome} {choque_bps:+.0f}bps",
            fonte=self.fonte,
        )

    def com_choques(self, choques: dict[int, float]) -> "Curva":
        """Cenário definido por âncoras: choque (bps) informado em alguns
        vértices, interpolado linearmente entre eles e mantido constante fora.

        É assim que se monta steepening e flattening — por exemplo
        `{126: -30, 2520: -5}` derruba a ponta curta mais que a longa.

        NÃO serve para key rate duration: com uma âncora só, a interpolação
        devolve o mesmo valor em toda a curva e o "choque localizado" vira um
        choque paralelo silencioso. Para isso existe `com_choque_local`.
        """
        if not choques:
            return self
        if len(choques) < 2:
            raise ValueError(
                "com_choques precisa de ao menos 2 âncoras — com uma só o "
                "choque vira paralelo. Use com_choque_local para bump "
                "localizado em um vértice."
            )
        pontos = np.array(sorted(choques), dtype=float)
        valores = np.array([choques[int(p)] for p in pontos], dtype=float)
        aplicado = np.interp(self.dias_uteis.astype(float), pontos, valores)
        return Curva(
            data=self.data,
            dias_uteis=self.dias_uteis,
            taxas=self.taxas + aplicado / 10_000.0,
            nome=f"{self.nome} (cenário por âncoras)",
            fonte=self.fonte,
        )

    def com_choque_local(
        self, vertice: int, bps: float, vertices_chave: list[int]
    ) -> "Curva":
        """Bump localizado em um vértice, no formato-tenda da convenção de mesa.

        O choque vale `bps` no vértice alvo e decai linearmente até zero nos
        vértices-chave vizinhos. Nas duas pontas o choque fica achatado para
        fora — é essa convenção que faz a soma das key rate durations
        reproduzir o DV01 paralelo, em vez de sobrar ou faltar sensibilidade
        nos extremos da curva.
        """
        chaves = sorted(set(int(v) for v in vertices_chave))
        if vertice not in chaves:
            raise ValueError(f"vértice {vertice} não está em vertices_chave")
        if len(chaves) < 2:
            raise ValueError("key rate duration precisa de ao menos 2 vértices-chave")

        i = chaves.index(vertice)
        x = self.dias_uteis.astype(float)
        peso = np.zeros_like(x)

        if i == 0:
            direita = chaves[1]
            peso = np.where(x <= vertice, 1.0, np.clip((direita - x) / (direita - vertice), 0, 1))
        elif i == len(chaves) - 1:
            esquerda = chaves[i - 1]
            peso = np.where(x >= vertice, 1.0, np.clip((x - esquerda) / (vertice - esquerda), 0, 1))
        else:
            esquerda, direita = chaves[i - 1], chaves[i + 1]
            subida = np.clip((x - esquerda) / (vertice - esquerda), 0, 1)
            descida = np.clip((direita - x) / (direita - vertice), 0, 1)
            peso = np.minimum(subida, descida)

        return Curva(
            data=self.data,
            dias_uteis=self.dias_uteis,
            taxas=self.taxas + peso * bps / 10_000.0,
            nome=f"{self.nome} ({vertice}du {bps:+.0f}bps)",
            fonte=self.fonte,
        )

    def para_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"dias_uteis": self.dias_uteis, "taxa": self.taxas})


class FonteCurva(Protocol):
    def curva(self, data_ref: date) -> Curva: ...
    def datas_disponiveis(self) -> list[date]: ...


class FonteB3:
    """Taxas referenciais da B3 (sistemaswebb3-derivativos).

    Produtos úteis: PRE (DI x pré), DIC (DI x IPCA), SLP (Selic x pré).
    """

    BASE = "https://sistemaswebb3-derivativos.b3.com.br/referenceRatesProxy"

    def __init__(self, produto: str = "PRE", usar_cache: bool = True):
        self.produto = produto
        self.usar_cache = usar_cache

    def _chamar(self, rota: str, payload: dict) -> str:
        token = base64.b64encode(json.dumps(payload).encode()).decode()
        resp = requests.get(
            f"{self.BASE}/{rota}/{token}", headers={"User-Agent": _UA}, timeout=30
        )
        resp.raise_for_status()
        return resp.text

    def datas_disponiveis(self) -> list[date]:
        bruto = self._chamar(
            "Search/GetDate", {"language": "pt-br", "id": self.produto}
        )
        return [date.fromisoformat(d[:10]) for d in json.loads(bruto)]

    def curva(self, data_ref: date) -> Curva:
        destino = CACHE / f"b3_{self.produto.lower()}_{data_ref.isoformat()}.csv"

        if self.usar_cache and destino.exists():
            texto = destino.read_text(encoding="utf-8")
        else:
            # GetDownloadFile: devolve CSV em base64 e HONRA a data.
            bruto = self._chamar(
                "Search/GetDownloadFile",
                {"language": "pt-br", "id": self.produto, "date": data_ref.isoformat()},
            ).strip().strip('"')
            if not bruto:
                raise ValueError(
                    f"B3 não tem {self.produto} em {data_ref}. "
                    f"A janela é curta — use datas_disponiveis()."
                )
            texto = base64.b64decode(bruto).decode("latin-1")
            if self.usar_cache:
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_text(texto, encoding="utf-8")

        df = pd.read_csv(io.StringIO(texto), sep=";", decimal=",")
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(
            columns={
                "Dias Úteis": "du",
                "Dias Corridos": "dc",
                "Preço/Taxa": "taxa",
                "Descrição da Taxa": "nome",
            }
        )
        df = df.dropna(subset=["du", "taxa"]).sort_values("du")

        return Curva(
            data=data_ref,
            dias_uteis=df["du"].to_numpy(dtype=int),
            taxas=df["taxa"].to_numpy(dtype=float) / 100.0,
            nome=str(df["nome"].iloc[0]),
            fonte=f"B3/{self.produto}",
        )

    def dias_corridos(self, data_ref: date) -> pd.DataFrame:
        """Pares (dias úteis, dias corridos) publicados pela B3 — usados para
        conferir o calendário de feriados contra a fonte oficial."""
        self.curva(data_ref)
        destino = CACHE / f"b3_{self.produto.lower()}_{data_ref.isoformat()}.csv"
        df = pd.read_csv(destino, sep=";", decimal=",")
        df.columns = [c.strip() for c in df.columns]
        return df.rename(columns={"Dias Úteis": "du", "Dias Corridos": "dc"})[["du", "dc"]]


class FonteDI1AoVivo:
    """Curva intradiária montada a partir dos futuros de DI1 da B3.

    Cotação pública, sem cadastro, com o atraso de ~15 min que a B3 aplica
    ao dado gratuito. Cada vencimento negociado vira um vértice: a taxa do
    contrato já é anual, base 252 — a mesma convenção de `Curva`.

    Taxa de cada vértice, em ordem de preferência:
      meio     média de compra e venda — o que o mercado aceita AGORA
      último   último negócio, quando falta um dos lados do livro
      ajuste   ajuste do pregão anterior, quando o contrato não negociou

    Não uso o último negócio como primeira opção porque, nos vencimentos
    ilíquidos, ele pode ter horas: em 28/09 o DI1Z28 tinha 5 negócios no dia
    e último a 13,915% com o livro em 13,870/13,880. O meio segue a curva;
    o último congela o vértice num preço velho.
    """

    URL = "https://cotacao.b3.com.br/mds/api/v1/DerivativeQuotation/DI1"

    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    def contratos(self) -> pd.DataFrame:
        """Um contrato por linha, com a taxa escolhida e de onde ela veio."""
        resp = requests.get(self.URL, headers={"User-Agent": _UA}, timeout=self.timeout)
        resp.raise_for_status()
        corpo = resp.json()
        if corpo.get("BizSts", {}).get("cd") != "OK":
            raise ValueError(f"B3 respondeu {corpo.get('BizSts')}")

        horario = pd.Timestamp(corpo["Msg"]["dtTm"])
        linhas = []
        for s in corpo.get("Scty", []):
            resumo = s.get("asset", {}).get("AsstSummry", {})
            venc = resumo.get("mtrtyCode", "")
            if not venc or venc.startswith("9999"):  # DI1D: o próprio DI, não um vencimento
                continue
            q = s.get("SctyQtn", {})
            compra = s.get("buyOffer", {}).get("price")
            venda = s.get("sellOffer", {}).get("price")
            linhas.append(
                {
                    "contrato": s["symb"],
                    "vencimento": date.fromisoformat(venc),
                    "compra": compra,
                    "venda": venda,
                    "ultimo": q.get("curPrc"),
                    "ajuste_anterior": q.get("prvsDayAdjstmntPric"),
                    "contratos_negociados": resumo.get("traddCtrctsQty") or 0,
                    "contratos_em_aberto": resumo.get("opnCtrcts") or 0,
                }
            )

        df = pd.DataFrame(linhas).sort_values("vencimento").reset_index(drop=True)
        tem_livro = df["compra"].notna() & df["venda"].notna()
        df["taxa"] = np.where(
            tem_livro,
            (df["compra"] + df["venda"]) / 2,
            df["ultimo"].fillna(df["ajuste_anterior"]),
        )
        df["origem"] = np.select(
            [tem_livro, df["ultimo"].notna()], ["meio", "último"], default="ajuste"
        )
        df.attrs["horario"] = horario
        return df.dropna(subset=["taxa"])

    @staticmethod
    def _montar(df: pd.DataFrame, data_ref: date, coluna: str, nome: str) -> Curva:
        du = np.array([dias_uteis(data_ref, v) for v in df["vencimento"]])
        taxas = df[coluna].to_numpy(dtype=float) / 100.0
        validos = (du > 0) & ~np.isnan(taxas)
        du, taxas = du[validos], taxas[validos]
        # Dois vencimentos no mesmo dia útil (feriado no dia 1º) viram um vértice.
        du, idx = np.unique(du, return_index=True)
        return Curva(data=data_ref, dias_uteis=du, taxas=taxas[idx], nome=nome, fonte="B3/DI1")

    @staticmethod
    def pregao(df: pd.DataFrame) -> date:
        """Dia do pregão a que a cotação se refere.

        Em fim de semana ou feriado a B3 continua servindo o último pregão;
        rolar para frente dataria a curva de sexta como se fosse segunda, e o
        "P&L do dia" ganharia um fim de semana de carrego que não aconteceu.
        """
        dia = df.attrs["horario"].date()
        return dia if eh_dia_util(dia) else proximo_dia_util(dia, -1)

    def curva_ao_vivo(self, df: pd.DataFrame | None = None) -> Curva:
        """Curva de agora (com o atraso da B3), liquidando hoje."""
        df = self.contratos() if df is None else df
        hoje = self.pregao(df)
        return self._montar(df, hoje, "taxa", "DI1 ao vivo")

    def curva_fechamento_anterior(self, df: pd.DataFrame | None = None) -> Curva:
        """Curva do pregão anterior — o ponto de partida do P&L do dia.

        Durante o pregão, é a dos ajustes anteriores que a própria cotação
        traz. Depois do fechamento a B3 troca esse campo pelo ajuste DE HOJE
        (medido em 28/09/2026: 13,548 às 15h48, 13,560 às 17h48 no DI1F27), e
        usá-lo daria um "P&L do dia" perto de zero e datado errado. Por isso o
        campo é conferido contra a curva oficial de ontem; se não bater, a
        curva de ontem vem da própria fonte oficial.
        """
        df = self.contratos() if df is None else df
        hoje = self.pregao(df)
        ontem = proximo_dia_util(hoje, -1)
        pelo_campo = self._montar(df, ontem, "ajuste_anterior", "DI1 ajuste anterior")
        try:
            oficial = FonteB3("PRE").curva(ontem)
        except Exception:  # noqa: BLE001 — sem a oficial, fica o campo
            df.attrs["ajuste_de_hoje"] = None
            return pelo_campo
        dif = np.abs(pelo_campo.taxas - oficial.taxa(pelo_campo.dias_uteis)) * 10_000
        if dif.max() < 2.0:
            df.attrs["ajuste_de_hoje"] = False
            return pelo_campo
        df.attrs["ajuste_de_hoje"] = True
        return Curva(data=ontem, dias_uteis=oficial.dias_uteis, taxas=oficial.taxas,
                     nome="PRE oficial", fonte="B3/PRE")

    # Protocolo FonteCurva: a fonte ao vivo só conhece o dia de hoje.
    def datas_disponiveis(self) -> list[date]:
        c = self.curva_ao_vivo()
        return [c.data]

    def curva(self, data_ref: date) -> Curva:
        c = self.curva_ao_vivo()
        if data_ref != c.data:
            raise ValueError("FonteDI1AoVivo só fornece a curva de hoje.")
        return c


class FonteDI1Historico:
    """Curva diária desde 2018, dos ajustes do DI1 nos arquivos de pregão da B3.

    É a mesma régua da curva ao vivo (`FonteDI1AoVivo`): cada vencimento de
    DI1 vira um vértice, taxa de ajuste do dia. Dá ao risco o histórico que a
    curva referencial da B3 não guarda.

    Os dados vêm de `src/baixar_historico_di1.py`.

    Descartado antes deste: montar a curva com os títulos do Tesouro Direto
    (2004 em diante). As taxas são da manhã, de varejo, com spread fixo, e a
    variação diária delas correlacionou só ~0,25 com a da B3 nos dias em
    comum — não serve para medir risco diário.
    """

    MESES = {c: i for i, c in enumerate("FGHJKMNQUVXZ", start=1)}

    def __init__(self, caminho: Path | None = None):
        self.caminho = caminho or RAIZ / "data" / "di1_ajustes.csv"
        self._por_dia: dict[date, pd.DataFrame] | None = None
        self._curvas: dict[date, Curva] = {}

    @classmethod
    def vencimento(cls, contrato: str) -> date:
        """DI1F27 -> primeiro dia útil de janeiro de 2027."""
        return proximo_dia_util(date(2000 + int(contrato[-2:]), cls.MESES[contrato[3]], 1))

    def _carregar(self) -> dict[date, pd.DataFrame]:
        if self._por_dia is None:
            df = pd.read_csv(self.caminho, dtype={"contrato": str})
            df["data"] = pd.to_datetime(df["data"]).dt.date
            vencs = {c: self.vencimento(c) for c in df["contrato"].unique()}
            df["vencimento"] = df["contrato"].map(vencs)
            df = df.drop_duplicates(["data", "contrato"], keep="last")
            self._por_dia = {d: g for d, g in df.groupby("data")}
        return self._por_dia

    def datas_disponiveis(self) -> list[date]:
        return sorted(self._carregar())

    def curva(self, data_ref: date) -> Curva:
        if data_ref not in self._curvas:
            dia = self._carregar().get(data_ref)
            if dia is None:
                raise ValueError(f"sem ajuste de DI1 em {data_ref}")
            df = dia.assign(ajuste=dia["ajuste"].astype(float))
            self._curvas[data_ref] = FonteDI1AoVivo._montar(
                df, data_ref, "ajuste", "DI1 ajuste"
            )
        return self._curvas[data_ref]


def conferir_calendario(data_ref: date, produto: str = "PRE") -> pd.DataFrame:
    """Confronta o calendário de feriados local contra os pares
    (dias úteis, dias corridos) que a B3 publica.

    Se bater em todos os vértices, a contagem de 252 está provada contra a
    fonte oficial — não contra um teste que eu mesmo inventei.
    """
    from datetime import timedelta

    fonte = FonteB3(produto)
    pares = fonte.dias_corridos(data_ref).dropna()

    linhas = []
    for du_b3, dc in zip(pares["du"].astype(int), pares["dc"].astype(int)):
        vencimento = data_ref + timedelta(days=int(dc))
        linhas.append(
            {
                "dias_corridos": int(dc),
                "du_b3": int(du_b3),
                "du_calculado": dias_uteis(data_ref, vencimento),
            }
        )

    df = pd.DataFrame(linhas)
    df["diferenca"] = df["du_calculado"] - df["du_b3"]
    return df
