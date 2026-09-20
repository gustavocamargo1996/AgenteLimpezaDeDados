# Changelog

Mudanças relevantes desta POC. Formato baseado em
[Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).

Nada foi publicado. `0.1.0` é o que `limpeza/__init__.py` declara em `__version__`; não
há tag, release nem pacote distribuído. As entradas abaixo referenciam arquivo e
função, não commit.

## [Não publicado]

Duas frentes desde o último marco: **provedor local e container**
(19/set/2026) e a **simplificação do gerador** (30/ago/2026). O invariante de
`tests/test_invariante.py` (495 erros, 121 mudanças, 121 TP, precisão 1,0,
recall 0,2444, F1 0,3929, 71 flags no `beers` de 300 linhas) é o mesmo nas
duas: nenhuma delas moveu um número publicado.

### Adicionado — provedor local e container (19/set/2026)

- **Seletor de provedor: OpenAI ou Ollama, num repositório só.**
  `limpeza/llm.py` é o **único** lugar que sabe que existe mais de um
  provedor; os quatro construtores de agente (`deteccao/regra.py`,
  `correcao/regras.py` duas vezes, `correcao/fd.py`) viraram uma chamada a
  `llm.construir(schema, papel)` cada. A escolha sai do `.env`: `PROVEDOR`,
  `OLLAMA_URL`, `MODELO_DETECCAO`/`_ESPECIFICADOR`/`_CODIGO`/`_FD` e
  `TIMEOUT_LLM`. Precedência de modelo: explícito → papel → `MODELO_LLM`.
- **`PROVEDOR` desconhecido falha alto**, com `ProvedorDesconhecido`. Não cai
  para `openai` em silêncio: um typo que caísse mandaria a tabela do usuário
  para fora da rede, que é exatamente o que o provedor local existe para
  impedir.
- **Timeout por provedor**: 180s no `openai`, 900s no `ollama` — uma chamada
  em CPU já levou 808s nesta POC. `ChatOllama` 1.1 **não aceita** `timeout`
  nem `max_retries` no construtor (verificado por inspeção), então o timeout
  desce por `client_kwargs={"timeout": N}` e o ramo local **não tem retry**;
  quem degrada quando uma chamada falha é `pipeline.py::_processar_coluna`, a
  fronteira de resiliência que já existia.
- **`escolher_modelo.py`** — mede se um modelo serve **antes** de gastar um
  run inteiro: invoca os quatro papéis com os prompts reais e amostras fixas
  do `beers`, e conta duas coisas por papel, schema preenchido e código aceito
  pelo portão AST. O veredito olha só a linha da detecção: um modelo que não
  gera `detectar(col)` aceito pelo portão produz limpador vazio, por melhor
  que seja nos outros três papéis.
- **A sondagem que autorizou este trabalho** foi exatamente esse teste, com
  `qwen2.5:3b`, três repetições por agente e os prompts reais: a **detecção**
  preencheu o schema 3/3 mas teve o código **rejeitado pelo portão AST 3/3**,
  sempre por conter mais de uma definição; o **tradutor de código** foi 3/3 no
  schema e 3/3 no portão; a **dependência funcional**, 3/3; o
  **especificador**, 2/3. Ou seja: o gargalo de um modelo pequeno é
  justamente o papel que decide se sai limpador.
- **Imagem da POC** (`Dockerfile` + `.dockerignore`), com os dois arquivos do
  MiniLM assados dentro e `MODELO_EMBEDDING=/modelo`: **732 MB de conteúdo,
  255 MB comprimido** (`docker images` relata 1,03 GB, que é métrica de disco
  não deduplicada do snapshotter containerd, não o tamanho transportável). A
  camada do `pip install` responde por 591 MB e o MiniLM por 45,2 MB. Os CSVs
  não entram na imagem: entram por volume. Smoke do `Embedder` dentro do
  container: `similar` 0,5428 contra `distante` 0,1821.
- **Stack do Portainer** (`docker-compose.yml`): `ollama` (modelos no volume
  `ollama-modelos`, que sobrevive a redeploy) e `poc` como **job one-shot**
  (`restart: "no"`). `SUJO`, `LIMPO` e `SAIDA` viraram o default de
  `--sujo`/`--limpo`/`--saida` — na UI do Portainer editar variável é fácil e
  editar comando é desconfortável, e pela linha de comando nada mudou.
- **Suíte de 87 para 114 testes**, todos sem API. Densidade de comentário:
  12,8% em 2.590 linhas.
- **Documentação**: seção nova no `CLAUDE.md` (provedores, precedência,
  timeout, o critério de escolha de modelo), seções "Rodando com modelo local"
  e "Em container / Portainer" no `README.md` e as variáveis novas no
  `.env.example`.

### Corrigido — revisão final antes do merge (19/set/2026)

- **O serviço `arquivos` publicava os dados do usuário numa web app
  arquivada.** A imagem `filebrowser/filebrowser` foi arquivada em
  01/set/2026 (sem releases, sem correção para advisory aberto), estava em
  `latest`, publicava em `0.0.0.0:8080` sem restrição de bind e servia o
  volume com o CSV sujo, o CSV limpo e `correcoes.csv` — a tabela inteira,
  célula a célula. **Comentada no `docker-compose.yml`**, com o motivo no
  próprio arquivo. O ciclo de trabalho do `README.md` passou a usar o
  navegador de volumes do Portainer.
- **A sobreposição de modelo por papel estava morta.**
  `pipeline.py::_construir_agentes` passava `config.MODELO_LLM` aos quatro
  construtores; como esse valor nunca é vazio, o nível 1 da precedência
  disparava sempre e `MODELOS_POR_PAPEL` nunca era consultado —
  `MODELO_DETECCAO` e os três irmãos eram inertes, com a suíte verde. Agora
  `_construir_agentes` chama os quatro **sem** argumento de modelo, e
  `main.py --modelo` tem `default=None`: só escreve `config.MODELO_LLM`
  quando é passado. A precedência real virou `--modelo` →
  `MODELO_<PAPEL>` → `MODELO_LLM`. O teste novo é sobre
  `_construir_agentes`, não sobre `llm.construir`: era essa a lacuna.
- **Tracing do LangSmith desligado explicitamente.** `langsmith` entra como
  dependência transitiva do `langchain-core`, e qualquer uma de
  `LANGSMITH_TRACING_V2`, `LANGCHAIN_TRACING_V2`, `LANGSMITH_TRACING` ou
  `LANGCHAIN_TRACING` com `"true"` no ambiente do sistema — que
  `load_dotenv()` não sobrepõe — mandaria todo prompt, com células reais, para
  a nuvem da LangChain, mesmo com `PROVEDOR=ollama`. `limpeza/config.py`
  escreve `"false"` nas quatro. Ver `docs/DECISOES.md#tracing-desligado`.
- **Os dois arquivos do MiniLM passaram a ser versionados**
  (`all-MiniLM-L6-v2/onnx/model_O4.onnx` e `tokenizer.json`, Apache 2.0, da
  `sentence-transformers`). O Portainer builda clonando o repositório e a
  pasta estava no `.gitignore`: o `COPY` do `Dockerfile` falhava, e as duas
  alternativas documentadas exigiam shell no servidor ou registry —
  exatamente o que o usuário-alvo não tem. Agora o build por Git funciona
  direto.
- **Falha total do LLM não avisava.** Com todos os agentes falhando o run
  completava, escrevia um limpador vazio e retornava 0 — no Portainer, um job
  verde. `pipeline.py::_avisar_limpador_vazio` imprime `AVISO: 0/N colunas
  receberam regra de deteccao` quando nenhuma coluna saiu com detector
  diferente de `DETECTA_NADA`. O código de saída **não** mudou.
- **O container roda como não-root** (usuário `limpeza`, uid 10001): é o
  primeiro contexto em que código escrito por LLM executa com o volume de
  dados montado em escrita. O `mkdir -p /dados` e o `chown` antes do `USER`
  são o que faz o volume nomeado nascer com o dono certo. Ver
  `docs/DECISOES.md#container-nao-root`.
- **O 11º ponto da costura cross-column foi eliminado, não documentado.**
  `escolher_modelo.py::CASOS` repetia `nome_funcao="detectar"`,
  `nome_argumento="col"`, `series_mode=True`; os dois `alvo` viraram
  **callables** (`regra.materializar` e `sandbox.validar`), que é o que o
  pipeline realmente chama. A lista do `CLAUDE.md` seção 8 voltou a 10
  pontos.
- **Mensagem de exceção unificada** em `limpeza/erros.py::descrever`: tipo
  **e** mensagem, truncada em 200 caracteres. `pipeline.py::_processar_coluna`
  e `correcao/cascata.py` registravam só `type(exc).__name__` — com Ollama,
  "404 model not found", "connection refused" e "timeout" viravam a mesma
  linha, e o log é a única janela do usuário no Portainer.
- **`TIMEOUT_LLM` tem um leitor só.** `config.py` lê a variável (valor
  não-inteiro ou `<= 0` é ignorado, sem traceback) e `llm.py` consulta apenas
  `config.TIMEOUT_LLM`; antes `llm.timeout_do_provedor` chamava
  `os.getenv` duas vezes e `config.TIMEOUT_LLM` era uma constante fixa que
  ignorava a variável de mesmo nome.
- Miudezas do mesmo passe: import de `config` não usado em
  `deteccao/regra.py`; `--repeticoes` sem `help=`; `AS base` nomeando um
  estágio nunca referenciado no `Dockerfile`; `tests/` fora do
  `.dockerignore`; acentos em docstrings de `tests/test_pipeline.py`.

**Simplificação do gerador de limpadores (30/ago/2026)** — as duas seções
abaixo, `Removido` e `Alterado`, são dela. O objetivo declarado era reduzir a
superfície do código sem mover um único número publicado — e o
invariante de `tests/test_invariante.py` (495 erros, 121 mudanças, 121 TP,
precisão 1,0, recall 0,2444, F1 0,3929, 71 flags no `beers` de 300 linhas)
continua idêntico ao do início.

### Removido

- **Modo `blind` e a comparação entre modos.** O `main.py` deixou de ter
  `--modo`: o único regime é o de orçamento de rotulagem. O `comparativo.md`
  e a máquina que o produzia saíram junto.
- **Camada 3 da cascata** (fallback por célula via LLM): módulo
  `fallback_celula.py`, a flag `USAR_FALLBACK`, o argumento
  `--limite-fallback` e a chave `contagem["fallback"]`. Medição de 7 execuções
  mostrou ~0–0,5% de acerto ao custo de milhares de chamadas de LLM, e era a
  única camada sem gate de 100%. Nunca participou de nenhum número publicado.
  Ver `docs/DECISOES.md#veredito-contra-o-fallback`.
- **Acoplamento ao `ZERODC_DIR`.** A entrada passou a ser caminho de arquivo
  (`--sujo`/`--limpo`); a variável de ambiente e a resolução de dataset por
  nome de pasta deixaram de existir.
- **Módulos de consumidor único**, fundidos nos seus chamadores:
  `embeddings.py` (para `amostragem.py`), `avaliacao.py` (para `metricas.py`),
  `identificador_regras.py` e `gerador_codigo.py` (para `correcao/regras.py`).
- **`verificar_ambiente.py` e `verificar_e2e.py`**, convertidos na suíte
  `pytest` de `tests/`.

### Alterado

- **`poc/` virou `limpeza/`**, com os subpacotes `deteccao/` (`regra`,
  `oraculo`, `refino`, `mascara`) e `correcao/` (`cascata`, `regras`, `fd`). O
  `deteccao.py` de 776 linhas virou quatro módulos; nenhum arquivo do pacote
  passa de **297 linhas** (era 1.118 o maior).
- **Espinha explícita em `limpeza/pipeline.py::gerar_limpador`**: as etapas do
  gerador, em ordem, num só lugar. `main.py` ficou só com CLI e impressão.
- **Tipos nomeados em `limpeza/tipos.py`** (`Coluna`, `Tabela`, `Amostra`,
  `Detector`, `Correcao`, `Trabalho`) no lugar dos dicionários que viajavam
  entre as etapas. `Detector` guarda `codigo`/`funcao` **sem** interpretar a
  assinatura, para não espalhar a premissa intra-coluna.
- **Loop de refinamento da detecção com oráculo virou o default**
  (`ITERACOES_DETECCAO = 5`), em vez de um modo opcional: é ele que produz os
  números publicados.
- **Resiliência concentrada numa fronteira só**, `pipeline.py::_processar_coluna`.
  Exceções caíram de **22 para 16** e loops de **68 para 40**; cada `except`
  restante tem um comportamento de degradação nomeado.
- **Comentários podados**: mecânica no código, porquê em `docs/DECISOES.md`.
  O pacote saiu de **4.969 para 2.497 linhas** e a densidade de comentário de
  **23,7% para 11,1%** (medido por `tests/medir_verbosidade.py`). Docstrings
  de 1 linha, comentário inline com teto de 2.
- **Suíte de testes de 1 para 87.** Inclui o invariante do limpador congelado,
  a trava byte a byte de `empacotar._ESTATICO` contra a fixture
  (`tests/test_estatico_congelado.py`) e um alarme do modelo ONNX real que
  pula quando o modelo não está no disco.
- **Documentação reescrita**: `CLAUDE.md` novo (arquitetura, tipos,
  invariante, políticas, costura cross-column, zonas de risco); `README.md`
  reescrito para a CLI atual; `docs/DECISOES.md` com todas as referências de
  **Onde** apontando para símbolos que existem.

## [0.1.0] — não publicado

Construção da POC (22–23/jul/2026).

> **Leia como história, não como mapa.** As entradas abaixo descrevem o código
> como ele era em julho de 2026 e ficam preservadas de propósito. Os caminhos
> `poc/...` não existem mais: o pacote é `limpeza/`. Tradução dos módulos
> citados aqui para os de hoje:
>
> | Citado abaixo | Hoje |
> |---|---|
> | `poc/dados.py` | `limpeza/dados.py` |
> | `poc/amostragem.py`, `poc/embeddings.py` | `limpeza/amostragem.py` |
> | `poc/identificador_regras.py`, `poc/gerador_codigo.py` | `limpeza/correcao/regras.py` |
> | `poc/avaliacao.py` | `limpeza/metricas.py` |
> | `poc/sandbox.py`, `poc/esquemas.py`, `poc/relatorio.py`, `poc/config.py` | mesmo nome em `limpeza/` |
>
> Além do rename, estas coisas **não existem mais** (ver `[Não publicado]`):
> os modos `blind`/`budget` e o `comparativo.md`; `main.py --reavaliar`;
> `main.py:_preparar_itens` (a lógica de ambiguidade vive em
> `limpeza/correcao/cascata.py::_itens`); `dados.particionar` e as duas escalas
> de holdout `generalizacao`/`cobertura`, substituídas pelo F1 de reparo de
> `avaliar_limpador.py`. A seção "Variância entre execuções" do README saiu com
> os modos que ela media, mas a advertência que ela carregava continua visível
> no README (em "As métricas de reparo"); a medição de julho está preservada
> em **Documented**, logo abaixo.

### Added

- Pipeline de dois agentes em série: `poc/identificador_regras.py` (agente 1 —
  cadeia de pensamento + regra em JSON, numa única chamada) e `poc/gerador_codigo.py`
  (agente 2 — JSON → função Python, com até `MAX_TENTATIVAS_CODIGO = 3` tentativas
  realimentadas pela rejeição). Contrato pydantic entre os dois em `poc/esquemas.py`.
- Modos `blind` e `budget` (`config.MODOS`), com `--modo ambos` como default do
  `main.py`. `blind` mostra só os valores sujos; `budget` mostra os pares
  `sujo → correto` das células representativas.
- Seleção de representantes por KMeans sobre os **valores distintos** da coluna
  (`poc/amostragem.py`): o mais atípico e o mais típico de cada grupo,
  `random_state=0`, `N_CLUSTERS = 6`, teto de `MAX_REPRESENTANTES = 12`.
- Embeddings locais MiniLM em ONNX (`poc/embeddings.py`), via `tokenizers` +
  `onnxruntime` — sem torch, sem download, sem rede.
- Portão AST + namespace restrito antes de executar o código gerado
  (`poc/sandbox.py`): exatamente uma `def corrigir(valor):`, sem import, sem chamada a
  `eval`/`exec`/`open`/`getattr` e afins, sem atributo dunder, builtins por allowlist.
- Artefatos por execução em `runs/<carimbo>__<modo>/` (`poc/relatorio.py`):
  `cadeias.md`, `regras.json`, `metricas.json`, `codigo.json` e `corretores.py`, mais
  `comparativo.md` na raiz de `runs/` quando os dois modos rodam.
- `codigo.json` por execução (`poc/relatorio.py:escrever`), guardando código, nota do
  tradutor, tentativas e rejeições de cada coluna. É o que viabiliza o `--reavaliar`:
  o `corretores.py` renomeia as funções para `corrigir_<coluna>` e não volta limpo
  para `sandbox.materializar`.
- `main.py --reavaliar PASTA[,PASTA...]`: recalcula as métricas de execuções passadas
  e reescreve os artefatos sem nenhuma chamada de LLM — lê `regras.json` e
  `codigo.json` do disco e reproduz os mesmos representantes, já que a seleção do
  KMeans é determinística e independente do modo. Aceita execuções antigas sem
  `codigo.json`, reconstruindo as funções a partir do `corretores.py` legado
  (`main.py:carregar_codigos`).
- `verificar_ambiente.py`: teste de fumaça offline (caminhos, carga dos CSVs,
  embeddings, KMeans, portão AST, presença da chave) que imprime os representantes que
  o agente 1 receberia, antes de gastar API.

### Documented

- **Variância entre execuções** (então na seção homônima do README; hoje o
  README carrega a advertência e esta entrada guarda a medição). Duas
  execuções de 23/07 com entrada idêntica — mesmo dataset, modelo, prompt e os mesmos
  representantes (KMeans `random_state=0`) — divergiram: `city` no modo `budget` fez
  100% às 11h33 e 11,3% às 17h11, e `state` no `blind` alternou entre `erro=sim` e
  `erro=NAO`. `abv`, `ibu` e `ounces` repetiram. `temperature=0` não garante
  determinismo. Causa do caso `city` legível no código gerado: numa execução o agente
  generalizou o sufixo (`\s+[A-Z]{2}`, cobre 127/127), na outra decorou os dois
  exemplos vistos (`PA|IN`, cobre 17/127). Consequência então registrada no README: os
  números publicados são de **uma execução**, não propriedades do método; medir faixa
  exigiria k repetições, que esta POC não implementa.

### Fixed

- **Teste de fumaça antes de aceitar o código gerado** (`poc/sandbox.py:testar_fumaca`,
  acionado em `poc/gerador_codigo.py:traduzir`). O portão AST valida estrutura e não
  pega defeito de runtime: na execução de 23/07 o agente 1 emitiu um `condicao_regex`
  com `(?i)` repetido no meio da expressão, o agente 2 copiou fiel, o AST aprovou e a
  função lançou `re.error` em **todas** as 3.977 células de `ounces` no modo `blind` —
  reportando 0% de acerto e 0% de dano, visualmente idêntico a uma regra apenas
  ineficaz. Agora a função é chamada uma vez por exemplo da própria spec antes de ser
  aceita; se explodir, vira rejeição com feedback e o laço de retry a devolve ao LLM.
- **`excecoes` exposto** no `cadeias.md` e na impressão do `main.py`. A contagem já
  existia em `poc/avaliacao.py` e não aparecia em lugar nenhum — foi o que escondeu o
  defeito acima.

### Changed

- **Carga literal do CSV** em `poc/dados.py:carregar`: passou a ler com
  `keep_default_na=False, na_values=[]`. Com o default do pandas, `"N/A"` vira `NaN` e
  fica indistinguível da célula vazia do arquivo limpo — na coluna `ibu` do `beers`
  isso apagava as 1.005 células erradas antes de qualquer algoritmo rodar. A sentinela
  de ausência voltou a ser texto que o agente pode detectar.
- **Ambiguidade declarada em vez de reduzida à moda** em `main.py:_preparar_itens`:
  no modo `budget`, valor sujo que corresponde a mais de um valor limpo passou a ser
  enviado ao agente 1 como `AMBIGUO`, com a contagem de valores corretos distintos e
  até 4 exemplos, em vez de virar um par `sujo → limpo` único. É o caso das 127
  células vazias de `state`. A renderização do aviso está em
  `poc/identificador_regras.py:_formatar_amostra`.
- **Duas escalas de holdout** em `poc/avaliacao.py`, no lugar de uma:
  `generalizacao` (só as linhas cujo valor sujo o agente nunca viu) e `cobertura`
  (a coluna inteira menos as linhas efetivamente exibidas), ambas medindo acerto e
  dano. A partição correspondente está em `poc/dados.py:particionar`. Quando a
  generalização degenera — toda a classe de erro concentrada num único valor mostrado,
  como `ibu` e `state` no `beers` — a medida é marcada como `NAO MENSURAVEL` com
  instrução de julgar pela cobertura, em vez de ficar sem número algum.
