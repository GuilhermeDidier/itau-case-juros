"""Exercita FonteCSV — o caminho que o arquivo da Bloomberg vai usar.

O CSV histórico chega só na semana da entrega. Se esse caminho estiver
quebrado, a descoberta aconteceria sob pressão e sem tempo de conserto.
Então ele é testado agora, contra arquivos sintéticos nos formatos que a
extração pode plausivelmente produzir.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from curvas import FonteCSV  # noqa: E402

FORMATOS = {
    "rótulos de mercado, data ISO": (
        "data,1M,6M,1A,2A,5A,10A\n"
        "2026-09-16,13.90,13.80,13.85,14.00,14.30,14.40\n"
        "2026-09-17,13.70,13.60,13.65,13.80,14.10,14.20\n"
        "2026-09-18,13.65,13.55,13.61,13.79,14.08,14.17\n"
    ),
    "dias úteis, data BR": (
        "data;21;126;252;504;1260;2520\n"
        "16/09/2026;13,90;13,80;13,85;14,00;14,30;14,40\n"
        "17/09/2026;13,70;13,60;13,65;13,80;14,10;14,20\n"
        "18/09/2026;13,65;13,55;13,61;13,79;14,08;14,17\n"
    ),
    "coluna extra que não é vértice": (
        "data,fonte,1A,2A,5A\n"
        "2026-09-17,BBG,13.65,13.80,14.10\n"
        "2026-09-18,BBG,13.61,13.79,14.08\n"
    ),
}


def checar(nome: str, conteudo: str) -> bool:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(conteudo)
        caminho = fh.name

    try:
        fonte = FonteCSV(caminho)
        datas = fonte.datas_disponiveis()
        c = fonte.curva(date(2026, 9, 18))

        # taxa no vértice informado tem que voltar igual ao arquivo
        vertice_longo = int(c.dias_uteis[-1])
        esperado = c.taxas[-1]
        obtido = c.taxa(vertice_longo)
        bate_vertice = abs(obtido - esperado) < 1e-9

        # data anterior à cobertura deve cair para a última disponível
        anterior = fonte.curva(date(2026, 9, 17))

        # data fora do início deve falhar, não devolver lixo
        try:
            fonte.curva(date(2020, 1, 1))
            falha_esperada = False
        except ValueError:
            falha_esperada = True

        ok = (
            len(datas) >= 2
            and bate_vertice
            and anterior.data == date(2026, 9, 17)
            and falha_esperada
            and np.all(np.diff(c.dias_uteis) > 0)
            and 0.10 < c.taxas.min() < 0.20
        )
        print(
            f"  {'OK  ' if ok else 'FALHOU'} {nome}\n"
            f"         {len(datas)} datas, {len(c.dias_uteis)} vértices "
            f"({c.dias_uteis[0]}..{c.dias_uteis[-1]} du), "
            f"taxa 1 ano {c.taxa(252)*100:.2f}%"
        )
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"  FALHOU {nome}\n         {type(exc).__name__}: {exc}")
        return False
    finally:
        Path(caminho).unlink(missing_ok=True)


def checar_recusa_arquivo_sem_vertice() -> bool:
    """Um CSV sem nenhuma coluna de vértice tem que falhar com mensagem
    clara, não produzir uma curva vazia silenciosamente."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, encoding="utf-8"
    ) as fh:
        fh.write("data,observacao\n2026-09-18,nada aqui\n")
        caminho = fh.name
    try:
        FonteCSV(caminho).curva(date(2026, 9, 18))
        print("  FALHOU recusa de arquivo sem vértice: aceitou sem reclamar")
        return False
    except ValueError as exc:
        print(f"  OK   recusa de arquivo sem vértice\n         {exc}")
        return True
    finally:
        Path(caminho).unlink(missing_ok=True)


def main() -> int:
    print("FonteCSV — formatos que a extração da Bloomberg pode produzir\n")
    resultados = [checar(n, c) for n, c in FORMATOS.items()]
    resultados.append(checar_recusa_arquivo_sem_vertice())

    print()
    if all(resultados):
        print("APROVADO — o caminho do CSV da Bloomberg está pronto")
        return 0
    print("REPROVADO")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
