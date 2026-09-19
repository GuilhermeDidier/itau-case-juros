"""Calendário de dias úteis da praça brasileira (ANBIMA/B3).

A convenção de juros do mercado local é exponencial em base 252 dias úteis.
Errar a contagem de dias úteis contamina PU, DV01 e todo o resto, então o
calendário vem do arquivo oficial de feriados nacionais da ANBIMA, não de uma
lista escrita à mão.
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parent.parent
ARQ_FERIADOS = RAIZ / "data" / "raw" / "feriados_nacionais.xls"
URL_FERIADOS = "https://www.anbima.com.br/feriados/arqs/feriados_nacionais.xls"

BASE_252 = 252.0

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
)


def baixar_feriados(destino: Path = ARQ_FERIADOS) -> Path:
    """Baixa a planilha oficial de feriados nacionais da ANBIMA."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL_FERIADOS, headers={"User-Agent": _UA}, timeout=60)
    resp.raise_for_status()
    destino.write_bytes(resp.content)
    return destino


@lru_cache(maxsize=1)
def feriados() -> tuple[date, ...]:
    """Feriados nacionais, já sem o rodapé de notas da planilha."""
    if not ARQ_FERIADOS.exists():
        baixar_feriados()

    df = pd.read_excel(ARQ_FERIADOS)
    datas = pd.to_datetime(df["Data"], errors="coerce").dropna()
    return tuple(sorted({d.date() for d in datas}))


@lru_cache(maxsize=1)
def _calendario_np() -> np.busdaycalendar:
    return np.busdaycalendar(
        weekmask="1111100",
        holidays=np.array([np.datetime64(d) for d in feriados()], dtype="datetime64[D]"),
    )


def cobertura() -> tuple[date, date]:
    """Primeiro e último feriado conhecidos — limite de confiança do calendário."""
    f = feriados()
    return f[0], f[-1]


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return pd.Timestamp(d).date()


def eh_dia_util(d) -> bool:
    return bool(np.is_busday(np.datetime64(_as_date(d)), busdaycal=_calendario_np()))


def dias_uteis(inicio, fim) -> int:
    """Dias úteis no intervalo [inicio, fim) — a contagem do mercado.

    Liquidação em `inicio`, vencimento em `fim`: o dia de liquidação conta,
    o de vencimento não.
    """
    return int(
        np.busday_count(
            np.datetime64(_as_date(inicio)),
            np.datetime64(_as_date(fim)),
            busdaycal=_calendario_np(),
        )
    )


def prazo_252(inicio, fim) -> float:
    """Prazo em anos na base 252."""
    return dias_uteis(inicio, fim) / BASE_252


def proximo_dia_util(d, deslocamento: int = 0):
    """Rola para dia útil; `deslocamento` move em dias úteis a partir dali."""
    resultado = np.busday_offset(
        np.datetime64(_as_date(d)),
        deslocamento,
        roll="forward",
        busdaycal=_calendario_np(),
    )
    return resultado.astype("datetime64[D]").astype(date)


def grade_dias_uteis(inicio, fim) -> list[date]:
    """Todos os dias úteis em [inicio, fim]."""
    cal = _calendario_np()
    dias = pd.date_range(_as_date(inicio), _as_date(fim), freq="D")
    marcados = np.is_busday(dias.values.astype("datetime64[D]"), busdaycal=cal)
    return [d.date() for d, ok in zip(dias, marcados) if ok]
