# Changelog

Mudanças relevantes desta POC. Formato baseado em
[Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).

Nada foi publicado. `0.1.0` é o que `poc/__init__.py` declara em `__version__`; não há
tag, release nem pacote distribuído. Não existe histórico de versionamento neste
diretório — as entradas abaixo referenciam arquivo e função, não commit.

## [0.1.0] — não publicado

Construção da POC (22–23/jul/2026).

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

- **Variância entre execuções** (README, seção "Variância entre execuções"). Duas
  execuções de 23/07 com entrada idêntica — mesmo dataset, modelo, prompt e os mesmos
  representantes (KMeans `random_state=0`) — divergiram: `city` no modo `budget` fez
  100% às 11h33 e 11,3% às 17h11, e `state` no `blind` alternou entre `erro=sim` e
  `erro=NAO`. `abv`, `ibu` e `ounces` repetiram. `temperature=0` não garante
  determinismo. Causa do caso `city` legível no código gerado: numa execução o agente
  generalizou o sufixo (`\s+[A-Z]{2}`, cobre 127/127), na outra decorou os dois
  exemplos vistos (`PA|IN`, cobre 17/127). Consequência registrada no README: os
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
