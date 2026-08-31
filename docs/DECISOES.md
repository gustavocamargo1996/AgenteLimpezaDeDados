# Decisões medidas

Este documento existe para que o raciocínio por trás de cada parâmetro e cada
gate sobreviva à poda de comentários (Task 9 do refactor). Cada entrada é
curta de propósito: o raciocínio longo mora aqui, o código só precisa de um
comentário de mecânica e um link para a entrada correspondente.

Os caminhos em **Onde** são os do pacote reestruturado (`limpeza/...`), não os
do `poc/...` atual — este documento é escrito para sobreviver ao rename da
Task 6.

<a id="carga-literal"></a>
## Carga literal

**Decisão:** `pd.read_csv(dtype=str, keep_default_na=False, na_values=[])` na
leitura dos dois CSVs (dirty e clean), em vez do `read_csv(dtype=str).fillna('null')`
do ZeroDC original.

**Por quê:** o pandas converte "N/A" em `NaN` por default. Na coluna `ibu` do
dataset beers isso funde a sentinela do dirty com o vazio do clean — os dois
viram o mesmo token — e apaga 1.005 erros reais (42% da coluna) antes de
qualquer algoritmo rodar.

**Onde:** `limpeza/dados.py::carregar` (hoje `poc/dados.py:37-57`, comentário
"DESVIO DELIBERADO DO ZERODC" nas linhas 40-48).

<a id="kmeans-sobre-distintos"></a>
## KMeans sobre distintos

**Decisão:** `amostragem.selecionar` roda o KMeans sobre `valores_distintos`
da coluna (lista deduplicada), não sobre as células.

**Por quê:** 2.410 células de `ounces` colapsam em apenas 25 valores
distintos. Embedar e clusterizar os 25 é o que revela o padrão de corrupção;
embedar célula a célula (como o ZeroDC faz) multiplicaria o custo de
embedding por ~96x sem acrescentar sinal novo, porque a maioria das células é
repetição do mesmo valor sujo.

**Onde:** `limpeza/config.py` (`N_CLUSTERS=6`, `MAX_REPRESENTANTES=12`; hoje
`poc/config.py:29-34`) e `limpeza/amostragem.py::selecionar` (hoje
`poc/amostragem.py:17-46`).

<a id="representantes-tipico-e-atipico"></a>
## Representantes típico e atípico

**Decisão:** por cluster, `selecionar` escolhe dois pontos — o mais distante
do centróide (mais atípico) e o mais próximo (mais típico) — em vez de uma
amostra aleatória do cluster.

**Por quê:** o mecanismo é uma medida, não uma escolha estética: por cluster,
`selecionar` calcula `np.linalg.norm(matriz[membros] - centro)` sobre os
membros do grupo e usa `argmax` (mais distante do centróide → mais atípico) e
`argmin` (mais próximo → mais típico) — cada representante entra por
distância no espaço de embeddings, não por amostragem aleatória. A dupla só
cumpre a função quando o cluster mistura sujo e limpo; o caso `ounces` mostra
onde essa premissa quebra: as mesmas 2.410 células que colapsam em 25 valores
distintos (ver "KMeans sobre distintos") estão 100% corrompidas, então a
forma corrompida vira a própria norma do cluster — não sobra um "típico
limpo" para o atípico contrastar, e a detecção correspondente dá F1=0
(`poc/deteccao.py:163-166`, ver "Ponto cego da corrupção universal"). É essa
mesma falha que `N_CLUSTERS=6`/`MAX_REPRESENTANTES=12` sozinhos não resolvem,
e que motiva o loop de refinamento com oráculo.

**Onde:** `limpeza/amostragem.py::selecionar` (hoje `poc/amostragem.py:31-39`,
comentários `# mais atipico` / `# mais tipico`, e `poc/amostragem.py:1-10`
para o limite estrutural declarado no docstring de módulo).

<a id="portao-ast"></a>
## Portão AST

**Decisão:** todo código Python gerado por LLM passa por um portão AST
(`sandbox.validar`) antes de ser materializado e executado: namespace
descartável (nunca `globals()`), allowlist explícita de builtins, proibição de
imports, de atributos dunder e de uma lista de chamadas perigosas
(`eval`, `exec`, `open`, `__import__`, `getattr`, `globals`, ...).

**Por quê:** o ZeroDC original (`correction.py:790`) faz
`exec(code, globals())`, o que permite ao código gerado redefinir qualquer
nome do módulo hospedeiro. Aqui o namespace do exec é sempre
`{re, __builtins__}` descartável, criado por chamada — o código gerado não
tem como alcançar nada fora dele.

**Onde:** `limpeza/sandbox.py::validar`/`materializar` (hoje
`poc/sandbox.py:1-27` para a justificativa, `65-155` para a implementação).

<a id="modo-serie"></a>
## Modo série

**Decisão:** `series_mode=True` acrescenta ao portão uma allowlist FECHADA de
17 atributos (`_ATRIBUTOS_SERIE`) — todo `ast.Attribute` do corpo da função
`detectar(col)` precisa ter `.attr` nesse conjunto, ou o código é rejeitado.

**Por quê:** sem essa allowlist, `detectar(col)` — que recebe a coluna inteira
como `pd.Series`, não uma célula — poderia chamar `to_csv`/`to_pickle` (I/O) ou
`duplicated`/`mode`/`groupby`/`shift`/`apply` (operações cross-row, fora do
contrato intra-coluna). `pd` também fica fora do namespace do exec nos dois
modos, então `pd.read_csv`/`pd.eval` dão `NameError` mesmo antes da allowlist
agir — duas barreiras independentes para o mesmo risco.

**Onde:** `limpeza/sandbox.py` (hoje `poc/sandbox.py:10-26` para a
justificativa, `51-62` para a allowlist, `117-124` para o gate).

<a id="orcamento-vs-gabarito"></a>
## Orçamento vs. gabarito

**Decisão:** o CSV `--limpo` alimenta 3 papéis distintos e não intercambiáveis
— orçamento de rotulagem (`dados.montar_rotulados`, ~10 valores por coluna),
gates da cascata (100% nos rótulos mostrados) e gabarito da métrica final —
e só o terceiro exige a tabela `clean` completa.

**Por quê:** quando `--limpo` traz só uma amostra parcial, orçamento e gates
continuam funcionando (operam sobre as linhas efetivamente mostradas ao
agente) e a métrica reporta apenas onde há verdade disponível, sem quebrar.
Colapsar os três papéis custa caro: em `state`, 127 células vazias
correspondem a 38 estados-limpo distintos; reduzir esse mapeamento a uma moda
única ("vazio → CO") ensinaria ao agente uma regra falsa, e ele produziria,
com toda a lógica do mundo, um corretor que escreve "CO" em toda célula vazia.

**Onde:** `limpeza/dados.py::montar_rotulados` (hoje `poc/dados.py:66-88`);
exemplo da ambiguidade em `main.py:52-60` (hoje `_preparar_itens`); os 3
papéis também estão descritos em
`docs/superpowers/specs/2026-08-30-simplificacao-limpador-design.md`, seção 3,
Decisão 4.

<a id="gates-de-100"></a>
## Gates de 100%

**Decisão:** nenhuma regra de código (camada 1) e nenhuma FD (camada 2) é
aplicada às células marcadas a menos que acerte 100% do conjunto rotulado
(sujos e limpos) — um único erro no rotulado já visto reprova a regra
inteira.

**Por quê:** é o mecanismo que mantém o dano da correção perto de ~0% nas
avaliações da POC — em vez de aplicar uma regra parcialmente certa e arriscar
estragar células que já estavam corretas, a cascata prefere não aplicar nada
e deixar a célula escalar para a próxima camada (ver "Escalonamento célula a
célula").

**Onde:** `limpeza/correcao/cascata.py::_gate_codigo` (hoje
`poc/cascata.py:24-36`) e `limpeza/correcao/fd.py::validar_fd` (hoje
`poc/fd.py:8,113-...`, comentário de módulo "exige 100% no conjunto
rotulado").

<a id="flag-em-vez-de-chute"></a>
## Flag em vez de chute

**Decisão:** célula que nenhuma camada resolve permanece com o valor sujo e
recebe `trilha_celula[idx] = "nao_resolvida"`; a cascata nunca escreve um
valor inventado numa célula que não conseguiu tratar.

**Por quê:** com `USAR_FALLBACK=False` (default), a contagem `fallback` fica
sempre 0 e a invariante contábil da cascata —
`contagem[codigo] + contagem[fd] + contagem[fallback] + contagem[nao_resolvida]
== células marcadas` — só fecha porque a flag absorve exatamente o resto.
Sinalizar substitui chutar.

**Onde:** `limpeza/correcao/cascata.py::rodar_cascata` (hoje
`poc/cascata.py:75-165`, invariante contábil documentada nas linhas 17-19).

<a id="veredito-contra-o-fallback"></a>
## Veredito contra o fallback

**Decisão:** a camada 3 da cascata (fallback por célula via LLM) fica
desligada por default (`USAR_FALLBACK=False`) e sai inteiramente do pacote
reestruturado — módulo, camada e flag `--limite-fallback`.

**Por quê:** medição de 7 execuções mostrou que o fallback resolve ~100% das
correções pendentes de `ounces`/`abv` mas com acerto de apenas ~0–0,5%, ao
custo de milhares de chamadas de LLM — e é a única camada da cascata sem gate
de 100%. O fallback não participou de nenhum número publicado da POC.

**Onde:** hoje `poc/config.py:39-50` (comentário do `USAR_FALLBACK`) e
`poc/fallback_celula.py` — ambos removidos no refactor (ver
`docs/superpowers/specs/2026-08-30-simplificacao-limpador-design.md`, seção 3,
Decisão 3, e seção 8 "Fora de escopo").

<a id="loop-de-refinamento-com-oraculo"></a>
## Loop de refinamento com oráculo

**Decisão:** o loop de active learning com oráculo (hoje `--e2e
--iteracoes-deteccao N>1`) fica e vira comportamento default, com orçamento de
`AMOSTRAS_POR_ITERACAO=2` valores distintos rotulados por iteração (metade
previsto-sujo, metade previsto-limpo).

**Por quê:** é este loop — e não a detecção de 1 passe — que produz os
números publicados da POC (Decisão 2 do design doc). Sem ele, a detecção fica
presa ao 1-passe, que é cego à corrupção universal (ver próxima entrada).

**Onde:** `limpeza/config.py` (`ITERACOES_DETECCAO`, `AMOSTRAS_POR_ITERACAO`;
hoje `poc/config.py:52-59`) e `limpeza/deteccao/refino.py` (hoje
`poc/deteccao.py::refinar_regra_deteccao`, linhas 570-722 — é o maior bloco
do módulo: 562 das 776 linhas de `poc/deteccao.py` pertencem ao loop de
refino).

<a id="ponto-cego-da-corrupcao-universal"></a>
## Ponto cego da corrupção universal

**Decisão:** aceito como limitação conhecida e fora de escopo (não é
"bug a corrigir" nesta POC): quando 100% das células de uma coluna estão
corrompidas, a forma corrompida vira a norma estatística da coluna, e nem o
KMeans nem a detecção de 1 passe encontram um "atípico" para apontar o erro.

**Por quê:** `poc/deteccao.py:163-166` documenta que `ounces`/`abv`/`city` dão
F1=0 no 1-passe exatamente por esse motivo. A causa mecânica está em
`poc/config.py:30-32` e no docstring de `poc/amostragem.py:1-10`: as 2.410
células de `ounces` colapsam em 25 valores, todos na mesma forma corrompida —
não existe cluster "limpo" para o atípico contrastar contra.

**Onde:** `limpeza/amostragem.py` (docstring de módulo, hoje
`poc/amostragem.py:1-10`) e `limpeza/deteccao/refino.py` (hoje
`poc/deteccao.py:163-166`).

<a id="deteccao-intra-coluna"></a>
## Detecção intra-coluna

**Decisão:** o contrato de detecção é `detectar(col) -> pd.Series[bool]` —
recebe só a própria coluna (uma `pd.Series` de strings) e devolve uma
`pd.Series` booleana do mesmo tamanho, sem olhar outras colunas da linha.
Detecção cross-column fica fora de escopo desta POC — é o próximo projeto, não
um descarte.

**Por quê:** essa restrição está hoje concentrada em exatamente **7 pontos, 3
arquivos**. Se esta simplificação espalhar esse conhecimento por mais
lugares, o próximo projeto (trocar `detectar(col)` por uma forma cross-column)
tem que redescobrir o contrato em vez de trocá-lo num lugar só. A tabela
abaixo é a mesma da seção 9 do design doc:

| # | Onde (hoje → futuro) | O quê | Já preparado para a troca? |
|---|---|---|---|
| 1 | `poc/sandbox.py::validar` → `limpeza/sandbox.py::validar` | `nome_funcao`/`nome_argumento` parametrizados | **sim** |
| 2 | `poc/sandbox.py::_ATRIBUTOS_SERIE` → `limpeza/sandbox.py` | allowlist que bloqueia cross-row e travessia de `pd` | não |
| 3 | `poc/sandbox.py::materializar` → `limpeza/sandbox.py` | `pd` fora do namespace do exec | não |
| 4 | `poc/deteccao.py` (prompt, 4 ocorrências) → `limpeza/deteccao/regra.py` | o prompt declara `detectar(col)` | não |
| 5 | `poc/deteccao.py::DETECTA_NADA` → `limpeza/deteccao/regra.py` | sentinela `"def detectar(col): ..."` | não |
| 6 | `poc/deteccao.py::construir_mascara` → `limpeza/deteccao/mascara.py` | aplica `detectar(col)` uma vez por coluna, posicional | não |
| 7 | `poc/empacotar.py` → `limpeza/empacotar.py` | cópia congelada da mesma aplicação, no limpador gerado | não |

**Onde:** ver tabela acima; lista original em
`docs/superpowers/specs/2026-08-30-simplificacao-limpador-design.md`, seção 9.

<a id="escalonamento-celula-a-celula"></a>
## Escalonamento célula a célula

**Decisão:** em cada camada da cascata, uma célula só conta como "resolvida"
por aquela camada se a camada ALTEROU seu valor. Célula que a camada deixou
intacta escala para a próxima camada (código → FD → flag). Se o gate de uma
camada reprova, TODAS as células marcadas que chegaram até ali escalam.

**Por quê:** replica a regra de escalonamento de `correction.py:1000-1002` do
ZeroDC, onde `mask = dirty != corrections` zera a detecção das células que
mudaram de valor. Sem essa regra, uma camada que roda sem alterar nada
esconderia células que continuam erradas, como se tivessem sido tratadas.

**Onde:** `limpeza/correcao/cascata.py` (docstring de módulo, hoje
`poc/cascata.py:6-19`; aplicação nos comentários `# intacta escala` das
linhas 104 e 123).

<a id="random-state-0"></a>
## random_state=0

**Decisão:** o KMeans de `amostragem.selecionar` roda sempre com
`random_state=0` fixo — nunca amostrado, nunca configurável por execução.

**Por quê:** a seleção de representantes fica determinística: uma reavaliação
reproduz exatamente os mesmos representantes de uma execução anterior. É o
que permite recalcular métricas (holdout, F1) sobre uma run já feita sem
chamar o LLM de novo — reavaliar custa zero de API.

**Onde:** `limpeza/amostragem.py::selecionar` (hoje `poc/amostragem.py:29`);
motivo explicado no docstring de `main.py::_selecionar_representantes` (hoje
`main.py:35-41`), que hoje chama `amostragem.selecionar` a partir do `main.py`
e migra para dentro de `limpeza/amostragem.py` na etapa (2) do pipeline (ver
design doc, seção 5, "A espinha").
