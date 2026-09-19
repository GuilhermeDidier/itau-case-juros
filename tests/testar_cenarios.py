"""Exercita os cenários e prova que fazem o que o nome diz.

Um cenário chamado "bull steepening" que na verdade achata a curva é pior
que nenhum cenário: o número sai plausível e a leitura sai errada. Então
cada cenário nomeado é verificado contra sua própria definição — direção do
nível e sinal da inclinação — em vez de ser aceito pelo rótulo.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from cenarios import (  # noqa: E402
    CURTA,
    DEZ_ANOS,
    catalogo_nomeado,
    cenario_historico,
)
from curvas import FonteB3  # noqa: E402

# nome -> (nível esperado, inclinação esperada); None = não restrito
ESPERADO = {
    "Paralelo +50bps": ("sobe", "igual"),
    "Paralelo -50bps": ("cai", "igual"),
    "Bull steepening": ("cai", "inclina"),
    "Bull flattening": ("cai", "achata"),
    "Bear steepening": ("sobe", "inclina"),
    "Bear flattening": ("sobe", "achata"),
    "Surpresa de Copom": ("sobe", "achata"),
}


def checar(nome: str, cenario, curva) -> bool:
    nova = cenario(curva)
    d_curta = (nova.taxa(CURTA) - curva.taxa(CURTA)) * 10_000
    d_longa = (nova.taxa(DEZ_ANOS) - curva.taxa(DEZ_ANOS)) * 10_000
    d_inclinacao = d_longa - d_curta

    nivel_esperado, incl_esperada = ESPERADO[nome]
    media = (d_curta + d_longa) / 2

    nivel_ok = (media > 1) if nivel_esperado == "sobe" else (media < -1)
    if incl_esperada == "inclina":
        incl_ok = d_inclinacao > 1
    elif incl_esperada == "achata":
        incl_ok = d_inclinacao < -1
    else:
        incl_ok = abs(d_inclinacao) < 1

    ok = nivel_ok and incl_ok
    print(
        f"  {'OK  ' if ok else 'FALHOU'} {nome:<20} "
        f"curta {d_curta:+7.1f}  longa {d_longa:+7.1f}  "
        f"inclinação {d_inclinacao:+7.1f}bps"
    )
    if not ok:
        print(f"         esperado: nível {nivel_esperado}, {incl_esperada}")
    return ok


def main() -> int:
    fonte = FonteB3("PRE")
    datas = sorted(fonte.datas_disponiveis())
    curva = fonte.curva(datas[-1])

    print(f"Curva base: {curva.data:%d/%m/%Y}\n")
    print("Cenários nomeados — cada um contra a própria definição:")
    resultados = [checar(n, c, curva) for n, c in catalogo_nomeado(50.0).items()]

    print("\nCenário histórico — dia de decisão do Copom:")
    c0, c1 = fonte.curva(date(2026, 9, 16)), fonte.curva(date(2026, 9, 17))
    copom = cenario_historico(c0, c1, rotulo="Copom 17/09/2026")
    nova = copom(curva)

    d1 = (nova.taxa(1) - curva.taxa(1)) * 10_000
    d252 = (nova.taxa(252) - curva.taxa(252)) * 10_000
    print(f"  {copom.nome}: {copom.descricao}")
    print(f"  aplicado sobre a curva de hoje -> 1du {d1:+.1f}bps, 252du {d252:+.1f}bps")

    # O movimento reaplicado tem que reproduzir o observado em toda a malha,
    # inclusive no overnight — é ele que identifica o dia como Copom.
    fiel = True
    for du in (1, 21, 252, 2520):
        observado = (c1.taxa(du) - c0.taxa(du)) * 10_000
        aplicado = (nova.taxa(du) - curva.taxa(du)) * 10_000
        bate = abs(aplicado - observado) < 0.5
        fiel = fiel and bate
        print(
            f"  {'OK  ' if bate else 'FALHOU'} {du:>4} du: observado "
            f"{observado:+7.2f}bps, reaplicado {aplicado:+7.2f}bps"
        )
    resultados.append(fiel)

    print()
    if all(resultados):
        print("APROVADO — cada cenário faz o que o nome diz")
        return 0
    print("REPROVADO")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
