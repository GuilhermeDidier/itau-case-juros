# Simulador de P&L e Risco — Carteira de Juros

Case de seleção, Mesa de Trading. Marca uma carteira de renda fixa pré na curva
de DI1 de agora e mostra quanto ela ganha ou perde quando a curva mexe, de onde
vem cada real e qual o risco de amanhã.

No ar: **https://simulador-juros-lucas.streamlit.app**

Todo dado é público e oficial da B3. Nada depende de terminal pago.

## O que a tela mostra

| Parte | O que responde |
|---|---|
| Curva e fita | O que a curva fez desde o ajuste de ontem e quanto cada vértice (6m a 10a) rendeu ou custou à carteira |
| Indicadores | Valor aplicado, DV01, VaR 99% de 1 dia e P&L do dia |
| Marcação | PU, duration, DV01 e convexidade por papel; DV01 por vértice; hedge com futuros de DI1 |
| Cenários | Choques nomeados e os piores dias desde 2018, ranqueados pelo impacto nesta carteira (1 ou 5 dias) |
| Decomposição de P&L | Carrego contra efeito de taxa, por vértice e por papel, fechando sem resíduo |
| Risco | VaR e ES por três métodos, backtest do VaR (Kupiec e semáforo de Basileia) e PCA da curva |

A carteira aceita LTN, NTN-F e futuros de DI1 e é editável na barra lateral.

## Como rodar

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python src/baixar_dados.py        # feriados e PUs do Tesouro (validação)

for t in tests/*.py; do ./.venv/bin/python "$t"; done
./.venv/bin/streamlit run app.py
```

O histórico do DI1 (`data/di1_ajustes.csv`) já vem no repositório. Para
atualizá-lo: `./.venv/bin/python src/baixar_historico_di1.py` — retoma de onde
parou, um arquivo por vez (em paralelo a B3 bloqueia o IP por alguns minutos).

## De onde vêm os dados

| Fonte | Uso | Módulo |
|---|---|---|
| Cotação dos futuros de DI1 (B3, ~15 min de atraso) | Curva de agora: marcação, P&L do dia, hedge | `FonteDI1AoVivo` |
| Arquivos diários de pregão da B3 (BVBG.086), desde 2018 | Histórico: cenários, VaR, backtest, PCA | `FonteDI1Historico` |
| Curva referencial PRE da B3 (~20 dias úteis) | Régua que confere as duas acima | `FonteB3` |
| Tesouro Transparente | Régua do apreçador | `tests/validar_pu_tesouro.py` |
| Feriados nacionais (ANBIMA) | Calendário de 252 dias úteis | `calendario.py` |

Taxa de cada vértice ao vivo = meio entre compra e venda; sem livro dos dois
lados, último negócio; sem negócio, ajuste anterior. O último negócio de um
vencimento ilíquido pode ter horas: o DI1Z28 chegou a ter último em 13,915%
com o livro em 13,870/13,880.

## Validações

Nenhuma confronta o código com expectativa própria: todas usam números
publicados por terceiros.

| O quê | Contra | Resultado |
|---|---|---|
| Curva ao vivo (ajuste do pregão anterior) | Curva PRE oficial da B3 | 45 vencimentos, diferença máxima 0,6 bp |
| Histórico do DI1 | Curva PRE oficial, nos dias em comum | Diferença máxima 0,5 bp |
| Histórico do DI1, encadeamento | Ajuste anterior de cada arquivo × ajuste do dia anterior | p99 de 0,7 bp em contratos acima de 1 ano |
| Preço do DI1 | Ajuste oficial (DI1F21, 25/09/2020) | R$ 99.486,56, ao centavo |
| PU de LTN e NTN-F | PUs oficiais do Tesouro | Ao centavo em 55 de 83 observações, 19 anos (ver abaixo) |
| Calendário de dias úteis | Pares dias úteis/corridos da B3 | 269 vértices, divergência zero |

## O que foi medido, e não suposto

**Depois do fechamento, a cotação da B3 troca o ajuste anterior pelo de hoje.**
Às 15h48 de 28/09/2026 o campo do DI1F27 trazia 13,548 (sexta); às 17h48,
13,560 (segunda). Usá-lo daria um P&L do dia perto de zero e datado errado. A
fonte confere o campo contra a PRE oficial de ontem e, se não bater, parte da
oficial.

**O Tesouro Direto não serve como histórico de risco.** Foi testado: a
variação diária da curva montada com os títulos acompanhou a da B3 só em
~0,25 de correlação nos dias em comum. É taxa de varejo, da manhã, com spread
fixo. A hipótese de ser só diferença de horário foi testada e refutada.

**Buraco no histórico não pode virar "variação diária".** Parear datas pela
posição transformou um intervalo de seis anos num movimento de 766 bp e
contaminou VaR e ranking. Só entram pares de dias úteis de fato consecutivos
(`pares_consecutivos`); o mesmo vale para a base do P&L do dia no histórico.

**A API da B3 tem uma armadilha.** `Search/GetList` aceita a data e a ignora —
devolve sempre a curva mais recente. Só `Search/GetDownloadFile` honra a data.

**Três regimes de convenção no Tesouro Direto.** Até 2013, liquidação em D+1 e
par taxa/PU consistente. De 2022 em diante, D+0 e consistente. Entre 2015 e
2021 não reconcilia sob nenhuma das duas (resíduo de 0,02% a 0,12% do PU);
registrado como aberto, em vez de alargar a tolerância do teste.

## Decisões

**Futuro de DI1 não tem desembolso.** O ajuste diário corrige o PU de ontem
pelo CDI, então o carrego do DI1 é a taxa do contrato menos o CDI, e o futuro
entra no risco e no P&L mas não no valor aplicado.

**Hedge vértice a vértice.** Zerar só o DV01 total deixa a carteira exposta à
inclinação. O hedge resolve um sistema linear pela key rate duration inteira,
com o contrato mais negociado a até 25% de cada vértice. Na carteira padrão, o
DV01 vai de R$ −6.257 a R$ −10.

**Key rate duration com bump em tenda.** O choque vale no vértice e zera nos
vizinhos, achatando nas pontas; a soma reproduz o DV01 paralelo.

**Interpolação no log do fator de capitalização**, equivalente a interpolar a
forward contínua. Interpolar a taxa spot produz forward serrilhada e estraga o
carrego.

**Cenário histórico reaplica a variação, não substitui a curva**, e o ranking
é por P&L nesta carteira, não por bps: o dia que mais move a curva não é o que
mais dói.

**Três métodos de VaR, de propósito, e um backtest.** Histórico contra Monte
Carlo mede o custo da hipótese normal; Monte Carlo contra paramétrico mede o da
linearização. O backtest estima o VaR de cada dia só com os 500 pregões
anteriores e testa pela razão de Kupiec se as exceções saem na frequência
prometida. Todo número de risco carrega o tamanho da amostra.

**`convexidade` é normalizada** ((d²PU/dy²)/PU, em anos²), como a mesa cota. A
segunda diferença em reais está em `gamma_pu`.

**Visual.** Papel e tinta: fundo cinza frio, barra lateral azul-tinta, azul e
vermelho só para ganho e perda (par validado para daltonismo), laranja só no
indicador ao vivo. Bricolage Grotesque, Instrument Sans e Martian Mono. Tema
fixo em claro: o vídeo é gravado uma vez.
