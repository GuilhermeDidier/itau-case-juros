"""Baixa o histórico de ajustes do DI1 dos arquivos de pregão da B3.

A curva referencial da B3 só guarda ~20 dias úteis. Os arquivos de pregão
(BVBG.086, "PR" na Pesquisa por Pregão) existem desde 2018 e trazem, para
cada vencimento de DI1, a taxa de ajuste do dia e a do dia anterior. É o
mesmo instrumento da curva ao vivo — histórico e marcação na mesma régua.

Cada arquivo tem ~46 MB de XML; aqui fica só o DI1, em
data/di1_ajustes.csv (versionado: ~4 MB contra ~1 h de download). Retoma de onde parou.

Rodar:  ./.venv/bin/python src/baixar_historico_di1.py
"""

from __future__ import annotations

import csv
import io
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calendario import grade_dias_uteis  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "data" / "di1_ajustes.csv"
INICIO = date(2018, 1, 1)  # arquivos anteriores voltam vazios (medido em 28/09/2026)
URL = "https://www.b3.com.br/pesquisapregao/download?filelist=PR{:%y%m%d}.zip"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
)
CAMPOS = ["data", "contrato", "ajuste", "ajuste_anterior"]

_BLOCO = re.compile(
    r"<PricRpt>(?:(?!</PricRpt>).)*?<TckrSymb>(DI1[A-Z]\d\d)</TckrSymb>(.*?)</PricRpt>", re.S
)
_AJUSTE = re.compile(r"<AdjstdQtTax[^>]*>([\d.]+)<")
_ANTERIOR = re.compile(r"<PrvsAdjstdQtTax[^>]*>([\d.]+)<")


def extrair(dia: date) -> list[dict] | None:
    """Linhas de DI1 do pregão; None se a B3 não tem arquivo para o dia."""
    # Um pedido por vez, com pausa: em paralelo a B3 passa a devolver 500 e
    # depois bloqueia o IP por alguns minutos (medido em 28/09/2026).
    time.sleep(1.0)
    resp = requests.get(URL.format(dia), headers={"User-Agent": _UA}, timeout=30)
    resp.raise_for_status()
    if len(resp.content) < 100:
        return None
    externo = zipfile.ZipFile(io.BytesIO(resp.content))
    interno = zipfile.ZipFile(io.BytesIO(externo.read(sorted(externo.namelist())[-1])))
    # Há várias versões do mesmo arquivo no dia; a última é a definitiva.
    xml = interno.read(sorted(interno.namelist())[-1]).decode("utf-8")

    linhas = []
    for simbolo, corpo in _BLOCO.findall(xml):
        ajuste = _AJUSTE.search(corpo)
        if not ajuste:
            continue
        anterior = _ANTERIOR.search(corpo)
        linhas.append(
            {
                "data": dia.isoformat(),
                "contrato": simbolo,
                "ajuste": ajuste.group(1),
                "ajuste_anterior": anterior.group(1) if anterior else "",
            }
        )
    return linhas


def main() -> int:
    feitos: set[str] = set()
    if SAIDA.exists():
        with SAIDA.open() as f:
            feitos = {r["data"] for r in csv.DictReader(f)}
    vazios_arq = SAIDA.with_suffix(".vazios")
    vazios = set(vazios_arq.read_text().split()) if vazios_arq.exists() else set()

    ontem = date.today() - timedelta(days=1)
    pendentes = [
        d for d in grade_dias_uteis(INICIO, ontem)
        if d.isoformat() not in feitos and d.isoformat() not in vazios
    ]
    # Do mais recente para trás: se a B3 limitar o ritmo, o que chegar primeiro
    # é o que mais importa para o risco de hoje.
    pendentes.reverse()
    print(f"{len(feitos)} dias já baixados, {len(pendentes)} pendentes", flush=True)

    novo = not SAIDA.exists()
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    falhas = 0
    with SAIDA.open("a", newline="") as f, ThreadPoolExecutor(max_workers=1) as pool:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS)
        if novo:
            escritor.writeheader()
        tarefas = {pool.submit(extrair, d): d for d in pendentes}
        for i, tarefa in enumerate(as_completed(tarefas), 1):
            dia = tarefas[tarefa]
            try:
                linhas = tarefa.result()
            except Exception as erro:  # noqa: BLE001 — refaz na próxima execução
                falhas += 1
                print(f"  falhou {dia}: {erro}", flush=True)
                continue
            if linhas is None:
                with vazios_arq.open("a") as v:
                    v.write(dia.isoformat() + "\n")
            else:
                escritor.writerows(linhas)
                f.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(pendentes)}", flush=True)

    print(f"fim. falhas: {falhas} (rodar de novo refaz só elas)")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
