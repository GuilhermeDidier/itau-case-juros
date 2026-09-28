"""Extrai da Bloomberg o histórico da curva DI pré e a ficha de dois títulos.

Roda no computador do laboratório, com o Terminal aberto e logado — a API
Desktop (blpapi) conversa com o Terminal em localhost:8194.

Instalar uma vez:
    pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi

Rodar:
    python extrair_bloomberg.py

Saídas, na pasta atual:
    curva_di_bloomberg.csv   uma linha por data, uma coluna por vértice
                             (é o formato que src/curvas.py:FonteCSV lê)
    titulos_bloomberg.csv    taxa, PU, duration, DV01 e convexidade de hoje

Só a biblioteca padrão além do blpapi — o computador do laboratório pode
não ter pandas.
"""

from __future__ import annotations

import csv
import sys
from datetime import date

import blpapi

# --- PREENCHER ------------------------------------------------------------
# Tickers de cada vértice da curva Pre x DI. Achar no Terminal com
# CRVF <GO> (Brazil, BRL, Pre x DI) ou perguntando no F1 F1.
# O rótulo da esquerda vira o nome da coluna — não mudar.
VERTICES = {
    "1M": "PREENCHER Index",
    "3M": "PREENCHER Index",
    "6M": "PREENCHER Index",
    "1A": "PREENCHER Index",
    "2A": "PREENCHER Index",
    "3A": "PREENCHER Index",
    "5A": "PREENCHER Index",
    "10A": "PREENCHER Index",
}

# Os dois títulos para conferir contra o simulador. Confirmar o ticker
# digitando-o no Terminal antes (deve abrir a tela do papel).
TITULOS = [
    "BLTN 0 01/01/28 Govt",
    "BNTNF 10 01/01/35 Govt",
]

INICIO = "20060101"  # o máximo que a Bloomberg devolver
# --------------------------------------------------------------------------

CAMPOS_TITULO = [
    "PX_MID",       # PU
    "YLD_YTM_MID",  # taxa (meio)
    "DUR_MID",      # duration de Macaulay
    "DUR_ADJ_MID",  # duration modificada
    "RISK_MID",     # DV01 por 100 de face
    "CNVX_MID",     # convexidade
    "LAST_UPDATE_DT",
    "TIME",
]


def abrir_sessao() -> tuple[blpapi.Session, blpapi.Service]:
    opcoes = blpapi.SessionOptions()
    opcoes.setServerHost("localhost")
    opcoes.setServerPort(8194)
    sessao = blpapi.Session(opcoes)
    if not sessao.start():
        sys.exit("Não conectou. O Terminal da Bloomberg está aberto e logado?")
    if not sessao.openService("//blp/refdata"):
        sys.exit("Não abriu o serviço //blp/refdata.")
    return sessao, sessao.getService("//blp/refdata")


def respostas(sessao: blpapi.Session):
    """Itera as mensagens até a resposta final da requisição."""
    while True:
        evento = sessao.nextEvent(5000)
        for msg in evento:
            yield msg
        if evento.eventType() == blpapi.Event.RESPONSE:
            return


def extrair_curva(sessao, servico) -> None:
    pendentes = [t for t in VERTICES.values() if t.startswith("PREENCHER")]
    if pendentes:
        sys.exit("Preencha os tickers da curva em VERTICES antes de rodar.")

    req = servico.createRequest("HistoricalDataRequest")
    for ticker in VERTICES.values():
        req.getElement("securities").appendValue(ticker)
    req.getElement("fields").appendValue("PX_LAST")
    req.set("startDate", INICIO)
    req.set("endDate", date.today().strftime("%Y%m%d"))
    req.set("periodicitySelection", "DAILY")
    sessao.sendRequest(req)

    por_ticker: dict[str, dict[str, float]] = {}
    for msg in respostas(sessao):
        if not msg.hasElement("securityData"):
            continue
        dados = msg.getElement("securityData")
        ticker = dados.getElementAsString("security")
        if dados.hasElement("securityError"):
            print(f"  ERRO  {ticker}: ticker não reconhecido")
            continue
        serie = por_ticker.setdefault(ticker, {})
        campos = dados.getElement("fieldData")
        for i in range(campos.numValues()):
            linha = campos.getValueAsElement(i)
            if linha.hasElement("PX_LAST"):
                dia = linha.getElementAsDatetime("date").strftime("%Y-%m-%d")
                serie[dia] = linha.getElementAsFloat("PX_LAST")

    datas = sorted({d for s in por_ticker.values() for d in s})
    with open("curva_di_bloomberg.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["data", *VERTICES])
        for dia in datas:
            w.writerow([dia, *(por_ticker.get(t, {}).get(dia, "") for t in VERTICES.values())])

    print(f"curva_di_bloomberg.csv: {len(datas)} datas")
    for rotulo, ticker in VERTICES.items():
        s = por_ticker.get(ticker, {})
        if s:
            print(f"  {rotulo:>4}  {min(s)} a {max(s)}  ({len(s)} obs)  último: {s[max(s)]}")
        else:
            print(f"  {rotulo:>4}  SEM DADOS ({ticker})")


def extrair_titulos(sessao, servico) -> None:
    req = servico.createRequest("ReferenceDataRequest")
    for ticker in TITULOS:
        req.getElement("securities").appendValue(ticker)
    for campo in CAMPOS_TITULO:
        req.getElement("fields").appendValue(campo)
    sessao.sendRequest(req)

    linhas = []
    for msg in respostas(sessao):
        if not msg.hasElement("securityData"):
            continue
        lista = msg.getElement("securityData")
        for i in range(lista.numValues()):
            dados = lista.getValueAsElement(i)
            ticker = dados.getElementAsString("security")
            if dados.hasElement("securityError"):
                print(f"  ERRO  {ticker}: ticker não reconhecido")
                continue
            campos = dados.getElement("fieldData")
            linha = {"titulo": ticker}
            for campo in CAMPOS_TITULO:
                linha[campo] = (
                    campos.getElement(campo).getValueAsString()
                    if campos.hasElement(campo) else ""
                )
            linhas.append(linha)

    with open("titulos_bloomberg.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["titulo", *CAMPOS_TITULO])
        w.writeheader()
        w.writerows(linhas)

    print("titulos_bloomberg.csv:")
    for linha in linhas:
        print("  " + "  ".join(f"{k}={v}" for k, v in linha.items()))


def main() -> None:
    sessao, servico = abrir_sessao()
    try:
        extrair_curva(sessao, servico)
        extrair_titulos(sessao, servico)
    finally:
        sessao.stop()
    print("\nMande os dois CSVs para o Guilherme.")


if __name__ == "__main__":
    main()
