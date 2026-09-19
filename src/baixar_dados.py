"""Baixa as fontes públicas que o projeto usa.

Os arquivos ficam fora do versionamento: o CSV do Tesouro tem ~14 MB e todos
são reprodutíveis a partir daqui.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "data" / "raw"

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
)

FONTES = {
    "feriados_nacionais.xls": (
        "https://www.anbima.com.br/feriados/arqs/feriados_nacionais.xls"
    ),
    "PrecoTaxaTesouroDireto.csv": (
        "https://www.tesourotransparente.gov.br/ckan/dataset/"
        "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
        "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv"
    ),
}


def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)
    falhas = 0
    for nome, url in FONTES.items():
        alvo = DESTINO / nome
        try:
            resp = requests.get(url, headers={"User-Agent": _UA}, timeout=180)
            resp.raise_for_status()
            alvo.write_bytes(resp.content)
            print(f"  ok      {nome}  ({len(resp.content)/1e6:.1f} MB)")
        except Exception as exc:  # noqa: BLE001
            print(f"  FALHOU  {nome}  {type(exc).__name__}: {exc}")
            falhas += 1
    print("\nA curva da B3 é baixada sob demanda por src/curvas.py (FonteB3).")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
