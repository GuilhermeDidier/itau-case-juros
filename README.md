# Simulador de P&L e Risco — Carteira de Juros

Case de seleção, Mesa de Trading. Simula o impacto de cenários de mercado
sobre uma carteira de renda fixa brasileira.

## Estado

| Etapa | Situação |
|---|---|
| Calendário 252 dias úteis | pronto, conferido contra a B3 |
| Curva de juros (fonte plugável) | pronto |
| Apreçamento LTN / NTN-F | pronto, conferido contra o Tesouro Nacional |
| DV01, convexidade, key rate duration | pronto |
| Cenários nomeados e históricos | pendente |
| Decomposição de P&L | pendente |
| VaR / Expected Shortfall | pendente |
| Interface | pendente |

## Como rodar

```bash
python3 -m venv .venv
./.venv/bin/pip install pandas numpy requests scipy xlrd openpyxl
./.venv/bin/python src/baixar_dados.py
./.venv/bin/python tests/validar_pu_tesouro.py
./.venv/bin/python tests/testar_fonte_csv.py
```

## Validações

Nenhum teste aqui compara o código com expectativa própria. Os dois
confrontam o cálculo com números publicados por terceiros.

**Calendário contra a B3.** A B3 publica, para cada vértice da curva, dias
úteis e dias corridos lado a lado. A contagem local reproduz os 269 vértices,
até 8.492 dias úteis. Divergência: zero.

**Apreçamento contra o Tesouro Nacional.** O Tesouro Transparente publica
taxa e PU de cada título, todo dia, desde 2004. O apreçador reconstrói o PU a
partir da taxa publicada e bate ao centavo em 55 de 83 observações, LTN e
NTN-F, ao longo de 19 anos.

As 28 restantes caem na janela 2015–2021, descrita abaixo.

## O que foi medido, e não suposto

**A API da B3 tem uma armadilha.** `Search/GetList` aceita o parâmetro de data
e o ignora — devolve sempre a curva mais recente, byte a byte idêntica.
Apenas `Search/GetDownloadFile` honra a data. Confiar no primeiro produziria
"cenários históricos" que são o dia de hoje disfarçado.

**A curva da B3 tem só ~20 dias úteis de histórico.** Janela rolante. Por isso
existe `FonteCSV`: o histórico longo vem de fora, e trocar a fonte é
configuração.

**Três regimes de convenção no Tesouro Direto.** Até 2013, liquidação em D+1 e
par taxa/PU consistente. De 2022 em diante, D+0 e par consistente. Entre 2015
e 2021 não reconcilia sob nenhuma das duas, com resíduo de 0,02% a 0,12% do
PU. Arredondamento de meia casa na taxa explicaria até ~0,03%; o resto está em
aberto e assim foi registrado, em vez de alargar a tolerância do teste.

**O lado "Compra" do Tesouro carrega spread de recompra**, não erro de cálculo:
0,045% a 0,051% do preço, uniforme entre todos os títulos (desvio 0,001%). É
verificado como constante, que é teste mais forte do que ignorá-lo.

## Decisões de arquitetura

**Fonte de dados plugável.** O resto do projeto só conhece `Curva` e
`FonteCurva`. Existem `FonteB3` (curva completa do dia) e `FonteCSV`
(histórico longo). O caminho do CSV já é testado contra os formatos que uma
exportação de terminal pode produzir — separador e decimal são detectados, não
assumidos.

**Interpolação no log do fator de capitalização**, equivalente a interpolar a
forward contínua. Interpolar a taxa spot direto produz forward serrilhada e
estraga o cálculo de carrego.

**Key rate duration com bump em tenda.** O choque vale no vértice alvo e decai
até zero nos vizinhos, achatando nas pontas. É essa convenção que faz a soma
das key rate durations reproduzir o DV01 paralelo — verificado, erro 0,000%.

**`convexidade` é normalizada** ((d²PU/dy²)/PU, em anos²), que é como a mesa
cota. A segunda diferença crua em reais está em `gamma_pu`, separada, para não
confundir as duas.

## Fontes

- Curva DI×pré e DI×IPCA — B3, taxas referenciais
- Taxa e PU de títulos públicos — Tesouro Transparente
- Selic diária — Banco Central, série SGS 11
- Feriados nacionais — ANBIMA
