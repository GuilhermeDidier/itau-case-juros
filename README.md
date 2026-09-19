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
| Cenários nomeados e históricos | pronto, cada um testado contra a própria definição |
| Decomposição de P&L | pronto, fecha sem resíduo |
| VaR / Expected Shortfall | máquina pronta; números aguardam histórico longo |
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

**Cenário histórico reaplica a variação, não substitui a curva.** A pergunta
não é quanto a carteira valia naquele dia, é quanto ela perderia se aquele dia
se repetisse hoje.

**Dias históricos são ranqueados por impacto na carteira, não por bps.** No
histórico disponível, o dia de maior movimento de curva (Copom, −25bps no
overnight) foi o quarto menos relevante em P&L — e chegou a dar lucro. O pior
dia real moveu 20bps e custou dez vezes mais. Ranquear por bps aponta o dia
errado.

**Três métodos de VaR, de propósito.** Histórico (sem hipótese de
distribuição), paramétrico delta-normal (rápido, gaussiano) e Monte Carlo
(gaussiano, mas com reprecificação completa). A distância entre eles isola o
custo de cada hipótese: histórico contra Monte Carlo mede a hipótese
distribucional; Monte Carlo contra paramétrico mede a não linearidade.

**Todo número de risco carrega o tamanho da amostra.** Com os 19 movimentos
que a B3 disponibiliza, o quantil de 99% se apoia em 0,2 observação — o VaR
histórico é o pior dia da amostra e não existe cauda para estimar. Medido
hoje, o paramétrico sai MAIOR que o histórico, o oposto do esperado na
literatura. Isso não refuta cauda gorda; mostra que 19 pontos não estimam
cauda. Está escrito no módulo em vez de virar um número bonito sem ressalva.

**PCA como sanidade antes de confiar na covariância.** Os três primeiros
componentes saem nível (91,1%), inclinação (7,6%) e curvatura (1,0%), 99,7%
acumulado — e o nome de cada um é deduzido do padrão de trocas de sinal das
cargas, não escrito à mão. É o resultado canônico de curva de juros.

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
