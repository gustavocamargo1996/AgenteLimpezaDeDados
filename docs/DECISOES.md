# Decisões medidas

Este documento existe para que o raciocínio por trás de cada parâmetro e cada
gate sobreviva à poda de comentários (Task 9 do refactor). Cada entrada é
curta de propósito: o raciocínio longo mora aqui, o código só precisa de um
comentário de mecânica e um link para a entrada correspondente.

O refactor terminou: o pacote é `limpeza/`, e todo caminho citado em **Onde**
aponta para um símbolo que existe hoje. Se um deles deixar de existir, corrigir
esta referência faz parte da tarefa que o removeu — não é dívida para depois.
Para a arquitetura e as políticas em vigor, ver `CLAUDE.md`.

<a id="carga-literal"></a>
## Carga literal

**Decisão:** `pd.read_csv(dtype=str, keep_default_na=False, na_values=[])` na
leitura dos dois CSVs (dirty e clean), em vez do `read_csv(dtype=str).fillna('null')`
do ZeroDC original.

**Por quê:** o pandas converte "N/A" em `NaN` por default. Na coluna `ibu` do
dataset beers isso funde a sentinela do dirty com o vazio do clean — os dois
viram o mesmo token — e apaga 1.005 erros reais (42% da coluna) antes de
qualquer algoritmo rodar.

**Onde:** `limpeza/dados.py::carregar` — o dicionário `ler` e o comentário
`keep_default_na=False` logo acima dele.

<a id="kmeans-sobre-distintos"></a>
## KMeans sobre distintos

**Decisão:** `amostragem.selecionar` roda o KMeans sobre `valores_distintos`
da coluna (lista deduplicada), não sobre as células.

**Por quê:** 2.410 células de `ounces` colapsam em apenas 25 valores
distintos. Embedar e clusterizar os 25 é o que revela o padrão de corrupção;
embedar célula a célula (como o ZeroDC faz) multiplicaria o custo de
embedding por ~96x sem acrescentar sinal novo, porque a maioria das células é
repetição do mesmo valor sujo.

**Onde:** `limpeza/config.py` (`N_CLUSTERS=6`, `MAX_REPRESENTANTES=12`) e
`limpeza/amostragem.py::selecionar`, chamado por
`limpeza/amostragem.py::representantes` sobre `coluna.valores_distintos`.

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
limpo" para o atípico contrastar, e a detecção de 1 passe dá F1=0 (ver
"Ponto cego da corrupção universal"). É essa mesma falha que
`N_CLUSTERS=6`/`MAX_REPRESENTANTES=12` sozinhos não resolvem, e que motiva o
loop de refinamento com oráculo.

**Onde:** `limpeza/amostragem.py::selecionar`, comentários `# mais atipico` /
`# mais tipico`; o limite estrutural está declarado no comentário de módulo de
`limpeza/amostragem.py`, logo abaixo do docstring.

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

**Onde:** `limpeza/sandbox.py` — `_CHAMADAS_PROIBIDAS` e
`_BUILTINS_PERMITIDOS` para as listas, `validar` para o portão e
`materializar` para o namespace descartável do `exec`.

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

**Onde:** `limpeza/sandbox.py` — `_ATRIBUTOS_SERIE` para a allowlist (17
nomes) e o ramo `series_mode` de `validar` para o gate.

<a id="orcamento-vs-gabarito"></a>
## Orçamento vs. gabarito

**Decisão:** o CSV `--limpo` alimenta 3 papéis distintos e não intercambiáveis
— orçamento de rotulagem (`amostragem._rotular`, ~10 valores por coluna),
gates da cascata (100% nos rótulos mostrados) e gabarito da métrica final —
e só o terceiro exige a tabela `clean` completa.

**Por quê:** quando `--limpo` traz só uma amostra parcial, orçamento e gates
continuam funcionando (operam sobre as linhas efetivamente mostradas ao
agente) e a métrica reporta apenas onde há verdade disponível, sem quebrar.
Colapsar os três papéis custa caro: em `state`, 127 células vazias
correspondem a 38 estados-limpo distintos; reduzir esse mapeamento a uma moda
única ("vazio → CO") ensinaria ao agente uma regra falsa, e ele produziria,
com toda a lógica do mundo, um corretor que escreve "CO" em toda célula vazia.

**Onde:** `limpeza/amostragem.py::_rotular` (chamado por `representantes`,
monta os pares `sujo`/`limpo`/`eh_erro` das linhas exibidas); o tratamento da
ambiguidade em `limpeza/correcao/cascata.py::_itens` (`ambiguo`,
`limpos_distintos`, `exemplos_limpos`); os gates em
`limpeza/correcao/cascata.py::_gate_codigo` e
`limpeza/correcao/fd.py::validar_fd`; o gabarito da métrica em
`limpeza/metricas.py::avaliar`. Os 3 papéis também estão descritos em
`CLAUDE.md`, seção 1, e em
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

**Onde:** `limpeza/correcao/cascata.py::_gate_codigo` (camada 1) e
`limpeza/correcao/fd.py::validar_fd` (camada 2). O resultado de cada um vira
`gate_codigo`/`gate_fd` na trilha, e só um gate aprovado deixa a camada
escrever em `correcoes`.

<a id="flag-em-vez-de-chute"></a>
## Flag em vez de chute

**Decisão:** célula que nenhuma camada resolve permanece com o valor sujo e
recebe `trilha_celula[idx] = "nao_resolvida"`; a cascata nunca escreve um
valor inventado numa célula que não conseguiu tratar.

**Por quê:** a invariante contábil da cascata —
`contagem["codigo"] + contagem["fd"] + contagem["nao_resolvida"]
== células marcadas` — só fecha porque a flag absorve exatamente o resto.
Sinalizar substitui chutar. (A chave `fallback` existia enquanto havia uma
camada 3; ela saiu junto com a camada — ver "Veredito contra o fallback".)

**Onde:** `limpeza/correcao/cascata.py::rodar_coluna` — a inicialização de
`trilha_celula` com `"nao_resolvida"`, o dicionário `contagem` com as três
chaves, e `contagem["nao_resolvida"] = len(pendentes)` no fim.
`_trilha_de_falha` mantém a mesma contabilidade quando a coluna inteira falha.

<a id="veredito-contra-o-fallback"></a>
## Veredito contra o fallback

**Decisão:** a camada 3 da cascata (fallback por célula via LLM) saiu
inteiramente do pacote — módulo, camada, flag de configuração e argumento de
CLI. Já vinha desligada por default; agora não existe.

**Por quê:** medição de 7 execuções mostrou que o fallback resolve ~100% das
correções pendentes de `ounces`/`abv` mas com acerto de apenas ~0–0,5%, ao
custo de milhares de chamadas de LLM — e era a única camada da cascata sem
gate de 100%. O fallback não participou de nenhum número publicado da POC.
Não reintroduza sem uma medição nova que contrarie essa.

**Onde:** em lugar nenhum do código — é o ponto da decisão. O módulo
`fallback_celula.py`, a flag `USAR_FALLBACK`, o argumento `--limite-fallback`
e a chave `contagem["fallback"]` foram removidos;
`limpeza/correcao/cascata.py::rodar_coluna` tem exatamente duas camadas mais a
flag. Registro da remoção no `CHANGELOG.md`
(`[Não publicado]`, Removido) e o racional em
`docs/superpowers/specs/2026-08-30-simplificacao-limpador-design.md`, seção 3,
Decisão 3, e seção 8 "Fora de escopo".

<a id="loop-de-refinamento-com-oraculo"></a>
## Loop de refinamento com oráculo

**Decisão:** o loop de active learning com oráculo fica e vira comportamento
default (`ITERACOES_DETECCAO=5`; `--iteracoes-deteccao 1` ainda desliga o
loop e volta ao 1-passe), com orçamento de `AMOSTRAS_POR_ITERACAO=2` valores
distintos rotulados por iteração (metade previsto-sujo, metade
previsto-limpo).

**Por quê:** é este loop — e não a detecção de 1 passe — que produz os
números publicados da POC (Decisão 2 do design doc). Sem ele, a detecção fica
presa ao 1-passe, que é cego à corrupção universal (ver próxima entrada).

**Onde:** `limpeza/config.py` (`ITERACOES_DETECCAO`, `AMOSTRAS_POR_ITERACAO`,
`LIMITE_PRECISAO_DETECCAO`), `limpeza/deteccao/refino.py::refinar_regra_deteccao`
(o loop) e `limpeza/deteccao/oraculo.py` (a escolha do que rotular a cada
iteração). O acionamento está em `limpeza/pipeline.py::_processar_coluna`.

<a id="ponto-cego-da-corrupcao-universal"></a>
## Ponto cego da corrupção universal

**Decisão:** aceito como limitação conhecida e fora de escopo (não é
"bug a corrigir" nesta POC): quando 100% das células de uma coluna estão
corrompidas, a forma corrompida vira a norma estatística da coluna, e nem o
KMeans nem a detecção de 1 passe encontram um "atípico" para apontar o erro.

**Por quê:** `ounces`/`abv`/`city` dão F1=0 no 1-passe exatamente por esse
motivo. A causa mecânica: as 2.410 células de `ounces` colapsam em 25 valores,
todos na mesma forma corrompida — não existe cluster "limpo" para o atípico
contrastar contra. É o que o loop de refino com oráculo compensa, comprando
rótulos em vez de confiar na estatística da coluna.

**Onde:** o limite está anotado no comentário de módulo de
`limpeza/amostragem.py`; a compensação, no comentário do prompt
`SISTEMA_UPDATE` de `limpeza/deteccao/refino.py` ("aqui e' onde o 1-passe
falha quando a corrupcao e' a norma"). O caso `ounces` está exercitado em
`tests/test_refino.py::test_ounces_like_regra_ampla_nao_punida_quando_todos_os_rotulos_sao_erro`:
com todos os rótulos sendo erro, uma regra que marca tudo tem precisão 1,0 e a
guarda não tem como puni-la — a cegueira é declarada, não acidental.

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

| # | Onde | O quê | Já preparado para a troca? |
|---|---|---|---|
| 1 | `limpeza/sandbox.py::validar` | `nome_funcao`/`nome_argumento` parametrizados | **sim** |
| 2 | `limpeza/sandbox.py::_ATRIBUTOS_SERIE` | allowlist que bloqueia cross-row e travessia de `pd` | não |
| 3 | `limpeza/sandbox.py::materializar` | `pd` fora do namespace do exec | não |
| 4 | `limpeza/deteccao/regra.py` (prompt, 4 ocorrências) | o prompt declara `detectar(col)` | não |
| 5 | `limpeza/deteccao/regra.py::DETECTA_NADA` | sentinela `"def detectar(col): ..."` | não |
| 6 | `limpeza/deteccao/mascara.py::aplicar_detectores` | aplica `detectar(col)` uma vez por coluna, posicional | não |
| 7 | `limpeza/empacotar.py::_ESTATICO` | cópia congelada da mesma aplicação, no limpador gerado | não |

O tipo `Detector` (`limpeza/tipos.py`) guarda `codigo` e `funcao` **sem
inspecionar a assinatura**, de propósito: é o que impede um oitavo ponto de
aparecer. A máscara é consumida como `DataFrame[bool]` pelas etapas de
correção, métrica e empacotamento, que não sabem como ela foi construída.

**Onde:** ver tabela acima; a mesma lista está reproduzida em `CLAUDE.md`,
seção 8, e a original em
`docs/superpowers/specs/2026-08-30-simplificacao-limpador-design.md`, seção 9.

<a id="deteccao-por-fd"></a>
## Detecção por dependência funcional

**Decisão:** a FD deixa de ser só camada de correção e passa a **detectar**
também: `deteccao/dependencia.py::detectar_dependencia` roda como etapa 3b,
depois da detecção intra-coluna, recebe `mascara_intra` (nunca a combinada) e
marca quem se desvia da moda condicionada do determinante escolhido pelo
agente. `limpeza/pipeline.py::gerar_limpador` soma as duas máscaras
(`(mascara_intra + mascara_fd) > 0`) antes de entrar na cascata de correção.

**Por quê:** medido antes de desenhar, em
`docs/superpowers/specs/2026-09-20-deteccao-por-dependencia-funcional-design.md`,
seção 1–2. Um erro é **inalcançável intra-coluna** quando o mesmo valor sujo
aparece às vezes certo e às vezes errado — nenhuma função do valor sozinho
separa os dois casos. No `beers`, **0%** dos erros são inalcançáveis: é por
isso que o invariante do projeto (construído sobre o `beers`) é cego a esta
lacuna. No `environment`, **21,5%** (247/1.147) são — concentrados por
inteiro em duas colunas, `State` (**165 de 165**) e `Climate_Zone`
(**82 de 82**): não é que alguns erros escapem, é que todo erro dessas
colunas é estruturalmente invisível para qualquer regra `detectar(col)`. A
forma do erro é sempre a mesma — `State='MH'` numa linha cuja `City` é
`'Bangalore'` — uma violação de dependência funcional, não um valor
inválido. Detecção por desvio da moda condicionada, medida no `environment`
completo, fecha a lacuna **na detecção da POC** com **F1=1,00** nas duas
colunas (`City -> State`: 165 TP, 0 FP, 0 FN; `City -> Climate_Zone`: 82 TP,
0 FP, 0 FN) — é a métrica do run, não do limpador entregue; ver o próximo
parágrafo.

**A moda condicionada não é infalível — e não deve ser lida como tal.** No
`beers`, com a máscara intra realista (não o cenário teórico acima), a FD de
teste marca 9 células: 8 acertos e **1 falso positivo**, a linha da
`Blackrocks Brewery` — `State='MA'` é o valor **correto** segundo o `clean`,
contra seis outras linhas `MI` da mesma cervejaria. É heterogeneidade real do
dado (o nome da cervejaria não é um determinante de verdade ali), não um bug
na moda. O gate de 100% sobre as células rotuladas
(`correcao/fd.py::validar_fd`, mesmo gate que já existia para a correção — ver
"Gates de 100%") é o que limita o estrago: uma FD que erra qualquer célula
rotulada nunca chega a marcar nada.

**A via da FD viaja para o limpador empacotado — a lacuna registrada aqui foi
fechada.** Ver `docs/DECISOES.md#fd-no-limpador` para o desenho, os dois
invariantes e o achado do run pago do `environment`.

**Por que NÃO o contrato geral `detectar(df)`:** a seção 8 do `CLAUDE.md`
antecipa um projeto maior — trocar `detectar(col)` por uma forma cross-column
qualquer. Este trabalho não é aquele. Nenhum dos cinco datasets disponíveis
(`beers`, `hospital`, `rayyan`, `flights`, `environment`) tem erro
cross-column que não seja violação de FD — consistência aritmética, ordem
temporal ou faixa condicionada não aparecem em nenhum deles, então construir
a capacidade geral agora seria construir sem dado que a valide. E o custo
seria alto por um motivo concreto: a rota geral mudaria
`limpeza/empacotar.py::_ESTATICO`, que é copiado byte a byte para o limpador
gerado e termina a fixture congelada do invariante — mudar o contrato
obrigaria a regerar a fixture e o invariante, trocando a rede de segurança
justamente durante a mudança que ela deveria proteger.

**Onde:** `limpeza/deteccao/dependencia.py::detectar_dependencia` (a etapa) e
`_desvios_da_moda` (a marcação); `limpeza/correcao/fd.py` (`propor_fd`,
`moda_condicionada`, `validar_fd` — mecânica compartilhada com a camada de
correção); `limpeza/pipeline.py::gerar_limpador` (a combinação de máscaras);
`tests/test_pipeline.py::test_deteccao_por_fd_recebe_a_mascara_intra_e_nao_a_combinada`
(a ordem, e que a marca da FD sobrevive até a máscara que a cascata recebe) e
`tests/test_dependencia.py` (a etapa isolada, e a revalidação do gate da FD
reaproveitada em `test_cascata_revalida_o_gate_da_fd_reaproveitada`). O
empacotamento da FD está documentado em `#fd-no-limpador`, no `CLAUDE.md`
("Duas vias de detecção") e no `README.md` ("Duas vias de detecção"). A
medição completa está em
`docs/superpowers/specs/2026-09-20-deteccao-por-dependencia-funcional-design.md`.

<a id="fd-no-limpador"></a>
## A FD viaja para o limpador gerado

**Decisão:** o arquivo gerado passa a levar **dois** dicionários de FD, não
um. `DEPENDENCIAS` leva as FDs aprovadas no gate da *detecção* — as mesmas
que produziram `trabalho.dependencia` no pipeline — e alimenta `_mascara`:
para cada `coluna: (determinante, dependente)` de `DEPENDENCIAS`,
`_desvios_da_moda` (novo em `_ESTATICO`) marca quem se desvia da moda
condicionada do determinante, e a marca entra por OU na máscara que já vinha
de `_DETECTORES`. `FDS` continua existindo, sem mudança de papel: leva as
FDs que a cascata usa para *corrigir*, célula marcada por qualquer via. A
leitura de `_desvios_da_moda` é sempre sobre a máscara **intra**, nunca sobre
a máscara que está sendo acumulada — a mesma regra de anticircularidade que
`pipeline.gerar_limpador` já aplicava (`#deteccao-por-fd`) —, e é por isso
que `_aplicar_fd` (a correção) passa a receber essa máscara combinada: o
Important 5 da revisão anterior (POC e limpador calculando a moda da correção
sobre máscaras diferentes) se resolve sem código a mais, porque `aplicar` já
passava `m` adiante.

**Por quê dois dicionários, não um:** a cascata de correção pode criar uma FD
só para corrigir uma coluna que o run nunca usou para *marcar* — a proposta
de FD de correção não passa pelo mesmo gate de detecção, e as duas listas de
colunas aprovadas não coincidem por construção. Um único dicionário faria o
limpador detectar com FDs que o run não aprovou para detectar, inflando a
máscara do artefato além do que a métrica publicada descreve.
`DEPENDENCIAS` só recebe o que `trabalho.dependencia` guarda — preenchido só
depois do gate de 100%, ver `#deteccao-por-fd` — então uma coluna com entrada
em `FDS` mas sem `dependencia` fica fora de `DEPENDENCIAS`.

**A guarda contra as duas cópias divergirem:** a lógica da FD existe duas
vezes por construção — o limpador é autônomo, não importa nada deste
repositório —, uma em `deteccao/dependencia.py` (POC) e outra em
`_ESTATICO::_desvios_da_moda` (limpador). `tests/test_equivalencia.py` gera
um limpador de verdade com `empacotar.gerar_limpador`, carrega-o do disco e
compara sua `_mascara(df)` com a máscara que o pipeline da POC produziria
(`construir_mascara` + `detectar_dependencia`, combinadas do mesmo jeito que
`pipeline.gerar_limpador` combina), célula por célula, em quatro cenários —
máscara intra vazia, máscara intra realista, o empate de moda `Bangalore`
(`KA` vs. `AA`) e FDs encadeadas (`City → State` e `State → Country`, o único
cenário em que ler `m` em vez de `m_intra` mudaria o resultado). É o teste
que falha se alguém tocar `_mascara` para ignorar `DEPENDENCIAS` de novo, ou
trocar `m_intra` por `m` dentro do laço.

**Por que o invariante do `beers` não se moveu.** O run de 20/ago que gerou
`limpador_beers_congelado.py` é anterior à detecção por FD: não existe
`trabalho.dependencia` nele, então a fixture ganhou `DEPENDENCIAS = {}` na
metade de cima e o `_ESTATICO` novo na metade de baixo — nada mais mudou. Os
sete números continuam **495 / 121 / 121 / 1,0 / 0,2444 / 0,3929 / 71**
(`tests/test_invariante.py::test_f1_reparo_do_limpador_congelado`). É a prova
de que o maquinário novo não interfere quando `DEPENDENCIAS` está vazio — as
duas FDs que `FDS` já carregava (`city`, `state`) continuam só corrigindo,
como antes.

**O invariante do `environment`, e o que ele prova.** Run pago em 21/set,
`environment_dirty_300.csv`, todas as 11 colunas, OpenAI `gpt-4o-mini`,
iterações padrão, congelado em `limpador_environment_congelado.py`. O
`avaliar_limpador.py` sobre as 300 linhas trava
**erros=334, mudancas=44, tp=44, precisao=1.0, recall=0.1317, f1=0.2328,
flags=663** (`test_f1_reparo_do_limpador_environment_congelado`). No gate da
detecção, o LLM aprovou `City→Country` e `City→Climate_Zone`; reprovou
`City→State` e `Country→City`. O limpador saiu com
`DEPENDENCIAS = {Country: City, Climate_Zone: City}` e
`FDS = {Climate_Zone: City}`. A diferença de base entre o número publicado
pelo run e o que `avaliar_limpador` mede é a esperada e já documentada em
`#orcamento-vs-gabarito`: o run mede fora das linhas rotuladas (o holdout),
`avaliar_limpador` mede a tabela de 300 linhas inteira — nos dois,
`Monitoring_Station_ID` fica com correção de praticamente 100% e o resto
perto de 0%, o que é fidelidade ao run, não regressão.

**O achado: a detecção intra super-marcada anula a FD justamente onde ela
seria necessária.** `State` e `Climate_Zone` são as duas colunas cujo erro é
inalcançável intra-coluna (`#deteccao-por-fd`: 165/165 e 82/82 no
`environment` completo) — todo valor sujo também aparece correto em alguma
outra linha, então a única regra `detectar(col)` que acerta 100% do rotulado
é marcar **tudo**: o LLM escreveu `^[A-Z]{2}$` para `State` (todo código de
estado casa) e `Tropical|Temperate` para `Climate_Zone` (todo valor casa).
Marcar tudo anula a FD por construção: o duplo filtro de
`moda_condicionada`/`_desvios_da_moda` tira da moda toda linha já marcada, e
com a máscara intra em 300/300 não sobra nenhuma linha limpa para servir de
referência — é por isso que `City→State` também reprova no gate, pelo mesmo
motivo. Medido no run: a FD contribuiu **zero** marcas extras em `State` e em
`Climate_Zone`; a correção nessas duas colunas é **0%**. O limpador cumpre o
que prometia — reproduz o run, inclusive esse resultado —, mas no
`environment`, com LLM real, a via da FD **não contribuiu**: não porque o
empacotamento falhe, e sim porque a detecção intra super-marca justamente nas
colunas onde a FD seria necessária. Essa é uma lacuna de desenho conhecida e
nomeada — a interação entre o gate de 100% intra-coluna e a FD —, candidata a
próximo projeto; não foi resolvida aqui. Quem prova que o mecanismo da FD no
limpador funciona quando o gate aprova sem essa interação é o teste de
equivalência acima e
`tests/test_empacotar.py::test_limpador_com_fd_injetada_corrige_o_environment`,
que injeta `DEPENDENCIAS` com as duas colunas e mede o limpador corrigindo
54/54 erros de `State` e 26/26 de `Climate_Zone`.

**Onde:** `limpeza/empacotar.py::_ESTATICO` (`_mascara`, `_desvios_da_moda`,
`_aplicar_fd` recebendo a máscara combinada) e
`limpeza/empacotar.py::gerar_limpador` (coleta de `trabalho.dependencia` em
`DEPENDENCIAS`); `tests/test_equivalencia.py` (a guarda); `tests/test_empacotar.py::test_dependencias_so_leva_fd_aprovada_na_deteccao` e
`::test_limpador_com_fd_injetada_corrige_o_environment`;
`tests/fixtures/limpador_beers_congelado.py` e
`tests/fixtures/limpador_environment_congelado.py`;
`tests/test_invariante.py` (os dois invariantes). Spec:
`docs/superpowers/specs/2026-09-21-fd-no-limpador-design.md`.

<a id="teto-de-densidade-14"></a>
## Teto de densidade em 14%

**Decisão:** o `CLAUDE.md` §7 passa a publicar um teto explícito de densidade
de comentário, em **14%**. Antes ele só registrava a densidade medida, sem
teto; os 13% existiam como meta interna do plano de execução deste projeto, e
foram revistos para 14% durante ele. A densidade ao final da rodada de
documentação é 13,1% em 2.656 linhas; depois da rodada de correção que se
seguiu (mais comentários, mais um teste), 13,2% em 2.671 — o teto de 14%
segue com folga.

**Por quê:** `tests/medir_verbosidade.py` conta cada docstring como
`len(docstring.splitlines()) + 2` linhas de "comentário", e a própria política
de comentários deste projeto **exige** docstring de 1 linha em todo módulo,
classe e função. Um módulo pequeno e bem documentado já nasce com piso alto
só por cumprir a política: `limpeza/deteccao/dependencia.py`, com 53 linhas e
quatro docstrings de 1 linha (módulo + 3 funções) mais um comentário inline
de mecânica, mede **28,3%** — mais que o dobro do teto antigo, sem um único
parágrafo de prosa fora de lugar. Um projeto que manda documentar toda função
tem um piso imposto pelo próprio instrumento de medida. Trocar o instrumento
no meio do plano apagaria a comparabilidade com a medição anterior; subir o
teto e registrar o motivo é a opção honesta.

**Onde:** `CLAUDE.md`, seção 7, "Comentários" (densidade e teto);
`tests/medir_verbosidade.py::medir` (a fórmula `len(d.splitlines()) + 2`).

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

**Onde:** `limpeza/correcao/cascata.py::rodar_coluna` — os dois comentários
`# intacta escala`, um na camada de código e um na de FD, cada um alimentando
a lista `restantes` que vira `pendentes` da camada seguinte.

<a id="random-state-0"></a>
## random_state=0

**Decisão:** o KMeans de `amostragem.selecionar` roda sempre com
`random_state=0` fixo — nunca amostrado, nunca configurável por execução.

**Por quê:** a seleção de representantes fica determinística: uma reavaliação
reproduz exatamente os mesmos representantes de uma execução anterior. É o
que permite recalcular métricas (holdout, F1) sobre uma run já feita sem
chamar o LLM de novo — reavaliar custa zero de API.

**Onde:** `limpeza/amostragem.py::selecionar`, na chamada
`KMeans(n_clusters=k, random_state=0, n_init=10)`. A seleção é acionada por
`limpeza/amostragem.py::representantes`, chamado pela etapa 2 do pipeline
(`limpeza/pipeline.py::_processar_coluna`) — não há mais nenhuma amostragem
disparada do `main.py`.

<a id="tracing-desligado"></a>
## Tracing desligado

**Decisão:** `limpeza/config.py` escreve `"false"` nas quatro variáveis de
tracing do LangChain/LangSmith — `LANGSMITH_TRACING_V2`,
`LANGCHAIN_TRACING_V2`, `LANGSMITH_TRACING` e `LANGCHAIN_TRACING` — logo após
o `load_dotenv()`, sobrepondo o que já estiver no ambiente.

**Por quê:** `langsmith` entra no venv como dependência transitiva do
`langchain-core`, sem ninguém pedir, e qualquer uma daquelas quatro variáveis
com `"true"` faz cada chamada de LLM ser espelhada para a nuvem da LangChain.
Os prompts desta POC carregam **valores reais das células** do usuário: seriam
os dados inteiros saindo da rede, silenciosamente, mesmo com
`PROVEDOR=ollama`. `load_dotenv()` não resolve — ele não sobrepõe variável já
definida no sistema operacional, que é exatamente o caso de uma máquina onde
alguém habilitou tracing uma vez. É a mesma decisão de privacidade que faz
`PROVEDOR` desconhecido falhar alto em vez de cair para `openai`: caminho para
fora da rede não se ativa por omissão.

**Onde:** `limpeza/config.py`, o laço logo abaixo de `load_dotenv()`.

<a id="container-nao-root"></a>
## Container não-root

**Decisão:** o `Dockerfile` cria o usuário `limpeza` (uid fixo 10001), cede a
ele `/app` e `/dados` e termina com `USER limpeza`. O `ENTRYPOINT` roda sem
privilégio.

**Por quê:** o container é o primeiro contexto em que código escrito por LLM
executa fora da máquina de quem revisou o run — e o portão AST de
`limpeza/sandbox.py` reduz superfície, mas **não é sandbox**. Com uid 0 e o
volume de dados montado em escrita, um escape do portão escreveria em
`/dados` como root e sairia do container com as capacidades do daemon. O uid
não-root não fecha o buraco; tira o pior desfecho da mesa.

**O custo, medido:** Docker semeia um volume nomeado **vazio** com o dono e o
modo do diretório que existe na imagem naquele caminho — por isso o `mkdir -p
/dados` e o `chown` **antes** do `USER` são obrigatórios: sem eles o volume
nasceria `root:root` e a escrita falharia. Verificado: volume novo do compose
monta como `limpeza:limpeza` e a POC escreve. Um volume que **já tem
conteúdo** de um deploy anterior como root mantém o dono antigo e a escrita
falha com `Permission denied` — a correção é um `chown -R 10001:10001` de uma
vez só, que no Portainer sai como um container avulso e privilegiado. Está no
README.

**Onde:** `Dockerfile`, as três linhas do `RUN useradd ...` e o `USER
limpeza`.
