# CLAUDE.md — orientação para sessões futuras

Leia este arquivo antes de tocar em qualquer coisa. Ele descreve a árvore que
existe hoje, não a que existia antes do refactor de simplificação
(2026-08-30). Quando ele divergir do código, **o código ganha** — e corrigir
este arquivo faz parte da tarefa que causou a divergência.

---

## 1. O que é

Entra um CSV **sujo** e um CSV **limpo** de referência. Saem três coisas:

1. `limpador_<dataset>_<carimbo>.py` — um script Python autônomo, sem
   dependência deste repositório, aplicável a qualquer tabela da mesma origem;
2. a **cadeia de pensamento** que gerou cada regra, em Markdown legível;
3. a **qualidade** desse limpador no dataset usado, em precisão/recall/F1 de
   reparo.

**O produto é o PROGRAMA de limpeza, não a tabela limpa.** A tabela corrigida
é subproduto: serve para medir. Se uma mudança melhora a tabela de saída mas
piora o limpador gerado, é regressão.

### Os três papéis do `--limpo`

Esta distinção é o coração do projeto. O CSV limpo tem **três papéis
distintos e não intercambiáveis**:

| Papel | O que consome | Quanto usa |
|---|---|---|
| **Orçamento de rotulagem** | os representantes do KMeans, rotulados par a par (`sujo → limpo`) | ~10 células por coluna — simula um humano rotulando |
| **Gates da cascata** | as mesmas células rotuladas | uma regra só é aplicada se acertar **100%** delas |
| **Gabarito da métrica** | a tabela `clean` completa | só para **medir**, nunca para detectar |

Colapsar os três é o erro clássico. Só o terceiro exige o `clean` inteiro; os
dois primeiros operam sobre as poucas linhas efetivamente mostradas ao agente.
**O `clean` nunca entra na detecção nem na correção fora do orçamento
rotulado** — se entrar, a POC passa a medir memorização e todos os números
publicados viram ficção. Ver `docs/DECISOES.md#orcamento-vs-gabarito`.

---

## 2. Como rodar

O `.venv` deste repositório foi criado com **uv**, que não instala `pip` dentro
do venv: `.venv/Scripts/python -m pip` responde `No module named pip`. Não é
ambiente quebrado — instale com `uv pip install -r requirements.txt` (com
`VIRTUAL_ENV` apontando para `.venv`, ou de dentro do diretório do projeto).
Um venv da biblioteca padrão (`python -m venv`) traz `pip` normalmente.

```bash
uv pip install -r requirements.txt
cp .env.example .env    # e cole sua OPENAI_API_KEY
```

Gerar um limpador (**gasta chamadas de API paga** — cada coluna dispara o
agente de detecção, o loop de refino com oráculo e a cascata de correção):

```bash
.venv/Scripts/python main.py --sujo caminho/beers_dirty.csv --limpo caminho/beers_clean.csv
```

Argumentos: `--colunas` (lista por vírgula ou `todas`, o default), `--modelo`,
`--saida`, `--iteracoes-deteccao`, `--amostras-iter`. Confira com `--help`.

Avaliar um limpador já gerado — **custo zero de API**, é o caminho para
iterar:

```bash
.venv/Scripts/python avaliar_limpador.py \
  --limpador tests/fixtures/limpador_beers_congelado.py \
  --sujo tests/fixtures/beers_dirty_300.csv \
  --limpo tests/fixtures/beers_clean_300.csv
```

Testes (nenhum gasta API — usam dublês de LLM e o limpador congelado):

```bash
.venv/Scripts/python -m pytest tests/ -v
```

### Em container

A imagem traz o MiniLM assado dentro e entra num stack com o Ollama: **o dado
do usuário não sai da rede** (o stack ainda puxa imagem e modelo da internet —
é a saída do *dado* que não existe). Os dois arquivos do modelo
(`all-MiniLM-L6-v2/onnx/model_O4.onnx` e `all-MiniLM-L6-v2/tokenizer.json`)
**são versionados**, para o build por Git do Portainer funcionar sem shell no
servidor. O container roda como usuário não-root (uid 10001, ver
`docs/DECISOES.md#container-nao-root`). O README tem o ciclo de trabalho no
Portainer; aqui fica só o essencial:

```bash
docker build -t limpeza-poc .
docker run --rm limpeza-poc --help
docker compose up -d ollama
```

O serviço `poc` do `docker-compose.yml` é um **job one-shot** (`restart: "no"`):
sobe, roda `main.py` uma vez com `SUJO`/`LIMPO`/`SAIDA` do ambiente, e para.

---

## 3. A arquitetura

```
main.py                     CLI: lê argumentos, chama a espinha, imprime
avaliar_limpador.py         CLI: F1 de reparo de um limpador já gerado (sem API)
escolher_modelo.py          CLI: mede se um modelo serve, antes de gastar um run

limpeza/
  pipeline.py               A ESPINHA — as etapas em ordem. Comece a ler aqui.
  tipos.py                  Coluna, Tabela, Amostra, Detector, Correcao, Trabalho
  config.py                 tudo que é ajustável (modelo, clusters, iterações)
  esquemas.py               contratos pydantic entre os agentes LLM
  llm.py                    o único lugar que sabe que há mais de um provedor
  erros.py                  formato único de exceção no log (tipo + mensagem)

  dados.py                  etapa 1 — carga literal dos dois CSVs → Tabela
  amostragem.py             etapa 2 — MiniLM ONNX + KMeans → representantes + rótulos
  deteccao/                 etapa 3 — onde estão os erros
    regra.py                  agente de detecção 1-passe → Detector
    oraculo.py                escolhe o que rotular na próxima iteração
    refino.py                 loop de active learning que melhora a regra
    mascara.py                aplica detectar(col) por coluna → DataFrame[bool]
    dependencia.py            etapa 3b — FD sobre a máscara intra: marca quem
                              desvia da moda condicionada do grupo
  correcao/                 etapa 4 — como consertar
    cascata.py                código → FD → flag, com gate de 100% em cada camada
    regras.py                 agente especificador + agente tradutor (JSON → Python)
    fd.py                     dependência funcional: MI, proposta, gate, aplicação
  sandbox.py                portão AST + namespace restrito (roda em toda etapa que gera código)
  empacotar.py              etapa 5 — escreve o limpador_<nome>.py autônomo
  metricas.py               etapa 6 — P/R/F1 de detecção, acerto/dano de correção
  relatorio.py              artefatos por execução (cadeias_deteccao.md, cascata.md, ...)

Dockerfile                  imagem da POC, com os dois arquivos do MiniLM dentro
docker-compose.yml          stack: ollama + a POC como job one-shot
```

A numeração é a das chamadas em `gerar_limpador`, logo abaixo, e os docstrings
de módulo repetem a mesma. Se as duas divergirem, a ordem das chamadas é que
vale.

### A ordem das etapas, sem intermediário

Este é o corpo de `limpeza/pipeline.py::gerar_limpador`, colado inteiro. Quem
ler isto sabe a ordem — não há um segundo lugar onde a sequência more:

```python
def gerar_limpador(caminho_sujo, caminho_limpo, colunas=None,
                   saida=None) -> tuple[Path, dict]:
    """Gera um limpador.py a partir do sujo e do orcamento de rotulos."""
    tabela = dados.carregar(caminho_sujo, caminho_limpo, colunas)
    agentes = _construir_agentes()
    carimbo = datetime.now().strftime("%Y-%m-%d_%H%M")
    saida = Path(saida) if saida else relatorio.criar_pasta("e2e", carimbo)
    saida.mkdir(parents=True, exist_ok=True)
    _anunciar(tabela)

    trabalhos = []
    for coluna in tabela.colunas:
        trabalhos.append(_processar_coluna(coluna, agentes))
    _avisar_limpador_vazio(trabalhos)

    mascara_intra = deteccao.construir_mascara(trabalhos, tabela)
    mascara_fd = deteccao.detectar_dependencia(
        trabalhos, tabela, mascara_intra, agentes["fd"])
    # A etapa nova recebe a mascara ANTERIOR: e' isso que evita a circularidade.
    mascara = ((mascara_intra + mascara_fd) > 0).astype(int)
    corrigido = correcao.rodar_cascata(trabalhos, tabela, mascara, agentes)
    limpador = empacotar.gerar_limpador(
        trabalhos, saida / f"limpador_{tabela.nome}_{carimbo}.py", tabela.nome
    )
    medida = metricas.avaliar(trabalhos, tabela, mascara, corrigido)
    relatorio.escrever(saida, tabela.nome, trabalhos, mascara, corrigido, medida)
    return limpador, medida
```

`_processar_coluna` (amostrar → detectar → refinar) é a única fronteira de
resiliência do run: ver a seção 7.

### Duas vias de detecção

A detecção tem duas etapas em série, não uma. A **intra-coluna**
(`deteccao.construir_mascara`) vê o valor isolado: `detectar(col)` olha uma
célula sem saber o resto da linha. A **FD** (`deteccao.detectar_dependencia`,
`deteccao/dependencia.py`) vê a linha: propõe um determinante por
informação mútua, um agente LLM escolhe qual candidata determina a coluna, e
quem se desvia da **moda condicionada** do grupo (o valor mais comum entre as
linhas com o mesmo determinante) é marcado.

A FD roda **depois** e recebe `mascara_intra` — a máscara **anterior**, nunca
a combinada. Isso não é ordem arbitrária: é o que evita a circularidade.
`fd.moda_condicionada` só soma ao pool linhas onde a detecção é 0 nas duas
colunas; se a FD recebesse a própria saída (ou a máscara já somada com a
dela), a moda de um grupo passaria a depender de quais células aquele mesmo
grupo já havia marcado — um resultado calculado a partir de si mesmo.
`tests/test_pipeline.py::test_deteccao_por_fd_recebe_a_mascara_intra_e_nao_a_combinada`
é o teste que trava essa ordem.

A moda só encontra erro que é **minoria no grupo** — se a maioria das linhas
de um determinante estiver errada da mesma forma, a moda aprende o erro, não
a correção. E mesmo quando o erro é minoria genuína a moda pode marcar uma
linha certa: no `beers`, a FD de teste marca 9 células — 8 acertos e 1 falso
positivo (a linha do `Blackrocks Brewery`, que é `MA` correto contra seis
outras linhas `MI` da mesma cervejaria — heterogeneidade real do dado, não
erro de digitação). O gate de 100% sobre as células rotuladas
(`fd.validar_fd`) é a proteção: uma FD que erra qualquer célula rotulada
nunca chega a marcar nada. Isso não garante que a FD foi testada — se
nenhuma célula rotulada da coluna estiver errada, o gate compara sujo com
limpo sem ter erro nenhum para pegar, e aprova por vacuidade
(`docs/DECISOES.md#fd-no-limpador`, o caso de `Climate_Zone` no
`environment`).

**A via da FD viaja para o limpador.** O arquivo gerado leva dois dicionários
— `DEPENDENCIAS` (as FDs que passaram no gate da *detecção*, vindas de
`trabalho.dependencia`) e `FDS` (as FDs de *correção*, que já existiam). São
dois porque a cascata pode criar uma FD só para corrigir uma coluna que o run
nunca usou para marcar; misturar os dois faria o limpador detectar com uma FD
que o run não aprovou para isso. `_ESTATICO::_mascara` soma, à máscara intra,
os desvios da moda condicionada de cada entrada de `DEPENDENCIAS`
(`_desvios_da_moda`, novo) — sempre lendo a máscara **intra**, nunca a que
está sendo acumulada, a mesma regra de anticircularidade do pipeline.
`tests/test_equivalencia.py` compara, célula por célula, a máscara que a POC
produz com a de um limpador gerado de verdade e carregado do disco: é o teste
que impede as duas cópias da lógica — a da POC e a de `_ESTATICO` — de
divergirem. Ver `docs/DECISOES.md#fd-no-limpador`.

---

## 4. Os tipos

Tudo que viaja entre as etapas está em `limpeza/tipos.py`, e nada mais viaja.

| Tipo | O que carrega | Cuidado |
|---|---|---|
| `Coluna` | `sujo`/`limpo` como `pd.Series`, `valores_distintos`, `contagem` | é a coluna, não o dataset |
| `Tabela` | `sujo`/`limpo` como `pd.DataFrame`, `nome`, `colunas` | aqui sim é o dataset inteiro |
| `Amostra` | `representantes` (valores distintos), `linhas` (índices), `rotulados` | `linhas` **é também o holdout da métrica** — medir nelas mede memorização |
| `Detector` | `codigo` (str), `funcao` (callable), `cadeia`, `orcamento`, `historico` | ver o aviso abaixo |
| `Correcao` | `passos` (o que o limpador vai empacotar), `trilha` (diagnóstico), `cadeia` | `passos` é o que sobrevive no arquivo gerado |
| `Trabalho` | uma `Coluna` + tudo que se descobriu sobre ela | é o acumulador que atravessa o pipeline inteiro |

**`Detector` guarda `codigo` e `funcao` sem inspecionar a assinatura, de
propósito.** Ele não sabe — e não pode passar a saber — que a função se chama
`detectar` nem que ela recebe uma coluna. Essa ignorância é o que permite
trocar o contrato de detecção sem tocar no tipo. Ver a seção 8.

**`Trabalho.dependencia` só é preenchido quando a FD passa no gate de 100%**
(`fd.validar_fd`) — é a `DependenciaFuncional` aprovada, com `determinante`,
`dependente` e `justificativa`. Quando a FD é proposta e reprova o gate, ou
quando nenhuma candidata passa no limiar de informação mútua,
`Trabalho.dependencia` fica `None`; `relatorio.py` só publica a seção da FD no
`cadeias_deteccao.md` quando o campo não é `None`. Um `Trabalho` sem
`dependencia` não é erro — é a maioria das colunas, já que a maior parte não
tem determinante forte o bastante.

---

## 5. Provedores de LLM

`limpeza/llm.py` é o **único** lugar do repositório que sabe que existe mais de
um provedor. Os quatro construtores de agente — `deteccao/regra.py`,
`correcao/regras.py` (duas vezes: especificador e tradutor de código) e
`correcao/fd.py` — chamam `llm.construir(schema, papel, modelo=modelo)` e não
importam cliente nenhum. O teste que vale é este:

```bash
git grep -n "ChatOpenAI\|ChatOllama" -- "limpeza/"
```

Se ele responder qualquer coisa além de `limpeza/llm.py`, o seletor virou dois
lugares. **Não acrescente um segundo.**

A superfície do módulo é pequena de propósito:

| Função | O que faz |
|---|---|
| `construir(schema, papel, modelo=None)` | o que os agentes chamam: devolve o cliente já amarrado ao schema pydantic |
| `cliente(provedor, modelo, segundos)` | o `if` do provedor, cru, sem schema |
| `modelo_do_papel(papel)` | `MODELO_<PAPEL>` se existir, senão `MODELO_LLM` |
| `timeout_do_provedor(provedor)` | `config.TIMEOUT_LLM` se existir, senão o default do provedor |
| `ProvedorDesconhecido(ValueError)` | o que `cliente` levanta num `PROVEDOR` que não conhece |

Os papéis são quatro, e o nome é a chave de tudo: `deteccao`,
`especificador`, `codigo`, `fd`.

### Precedência de modelo

Explícito → papel → geral:

1. o argumento `modelo=` de `construir` — `escolher_modelo.py`, e
   `main.py --modelo`, que escreve `config.MODELO_LLM` **só quando é passado**;
2. `MODELO_DETECCAO`, `MODELO_ESPECIFICADOR`, `MODELO_CODIGO`, `MODELO_FD` —
   a sobreposição por papel, montada em `config.MODELOS_POR_PAPEL`. Vazio não
   conta como valor: cai para o geral;
3. `MODELO_LLM` — o geral.

**`pipeline.py::_construir_agentes` chama os quatro construtores SEM argumento
de modelo**, de propósito: é o `None` fluindo até `llm.modelo_do_papel` que
mantém o nível 2 vivo. Passar `config.MODELO_LLM` ali — como já esteve — faz o
nível 1 disparar sempre e mata a sobreposição por papel em silêncio, com a
suíte verde. `tests/test_pipeline.py::test_construir_agentes_respeita_a_sobreposicao_por_papel`
é o teste que trava isso; teste sobre `llm.construir` sozinho **não** pega.

### Timeout, e o que o `ChatOllama` não tem

O default é **180s no `openai` e 900s no `ollama`**
(`config.TIMEOUT_POR_PROVEDOR`); a variável `TIMEOUT_LLM`, lida uma única vez
em `config.py` (valor não-inteiro ou `<= 0` é ignorado), sobrepõe os dois. Os 900s não
são folga decorativa: uma chamada em CPU já levou 808s nesta POC.

**`ChatOllama` 1.1 não aceita `timeout` nem `max_retries` no construtor** —
verificado por inspeção da assinatura, não suposto. O timeout desce por
`client_kwargs={"timeout": segundos}`, que o `httpx` recebe. Retry **não
existe** no ramo Ollama (o ramo OpenAI tem `max_retries=2`), e não precisa
existir: quando uma chamada falha, quem degrada é
`pipeline.py::_processar_coluna` — a coluna fica com a regra de 1-passe, ou
sem regra, e o run continua.

### `PROVEDOR` desconhecido falha alto

`cliente` levanta `ProvedorDesconhecido` em vez de cair para `openai`. Isso
**não** é rigor estético: é a restrição de privacidade do projeto. Um typo em
`PROVEDOR` que caísse silenciosamente para o padrão mandaria a tabela do
usuário para fora da rede — exatamente o que o provedor local existe para
impedir. Falhar no primeiro agente, com o valor errado na mensagem, é o
comportamento correto.

### `escolher_modelo.py`: o modelo serve?

Mede se um modelo consegue preencher os quatro schemas **antes** de gastar um
run inteiro. Usa os prompts reais e amostras fixas do `beers`, três repetições
por papel:

```bash
.venv/Scripts/python escolher_modelo.py --modelo qwen2.5-coder:14b
```

O provedor é o do `.env` (`PROVEDOR=ollama` para medir um modelo local), e
`--modelo` passa por cima de `MODELO_LLM` e das sobreposições por papel nos
quatro de uma vez. `--repeticoes` muda o denominador, que é 3 por padrão.

Sai uma linha por papel com dois números: quantas vezes o schema foi
preenchido e, nos dois papéis que geram código, quantas vezes o portão AST
aceitou o resultado. O portão de cada caso é um **callable** em `CASOS` —
`regra.materializar` na detecção e `sandbox.validar` no código —, nunca um
dicionário de `nome_funcao`/`nome_argumento`: é assim que a ferramenta mede o
degrau real do pipeline sem virar um 11º ponto da lista da seção 8.

**O critério de aprovação é a linha da detecção.** Um modelo que preenche o
schema mas cujo `detectar(col)` o portão rejeita produz um limpador **sem
detecção intra-coluna** — por melhor que seja nos outros três papéis. A
cadeia é mecânica: sem regra aceita, `_processar_coluna` cai para
`detector_nulo` (`DETECTA_NADA`, que é `col.isin([])`); um detector que não
marca nada dá máscara intra toda zero. Isso não faz o limpador sair vazio: se
alguma FD passou no gate da detecção, ela ainda soma marcas à máscara e a
cascata ainda corrige por ela — é o cenário que
`tests/test_empacotar.py::test_limpador_com_fd_injetada_corrige_o_environment`
exercita. É por isso que `_imprimir` decide o veredito olhando só
`placar["deteccao"]["portao_ok"]`: os outros três papéis não salvam um run
cuja detecção intra-coluna não passou no portão.

---

## 6. Invariante de aceitação

São **duas** fixtures e dois invariantes, um por dataset, ambos em
`tests/test_invariante.py`.

`limpador_beers_congelado.py`, aplicado às 300 linhas de `beers_dirty_300.csv`:

| Métrica | Valor travado |
|---|---|
| `erros` | 495 |
| `mudancas` | 121 |
| `tp` | 121 |
| `precisao` | 1.0 |
| `recall` | 0.2444 |
| `f1` | 0.3929 |
| `flags` | 71 |

`limpador_environment_congelado.py`, aplicado às 300 linhas de
`environment_dirty_300.csv`:

| Métrica | Valor travado |
|---|---|
| `erros` | 334 |
| `mudancas` | 44 |
| `tp` | 44 |
| `precisao` | 1.0 |
| `recall` | 0.1317 |
| `f1` | 0.2328 |
| `flags` | 663 |

```bash
.venv/Scripts/python -m pytest tests/test_invariante.py -v
```

> **ALTERAR ESSES NÚMEROS É BUG, NÃO MELHORIA.**
>
> Não são uma meta a superar: são a assinatura de que a semântica da cascata,
> da máscara e do empacotamento continua a mesma. Um refactor que faz o
> `recall` subir mudou o comportamento sem querer — e mudou onde não estava
> olhando. Se você acredita de verdade que a mudança é uma melhoria, ela é uma
> mudança de comportamento: precisa de decisão explícita do usuário, de uma
> entrada nova em `docs/DECISOES.md` e de uma fixture nova. Nunca de um
> `assert` reescrito para bater com o que saiu.

`tests/test_estatico_congelado.py` guarda a outra ponta, para **as duas**
fixtures: a constante `_ESTATICO` de `limpeza/empacotar.py` tem de continuar
batendo byte a byte com o final de cada uma. Se ela divergir, o limpador que
a POC gera hoje deixou de ser o mesmo que o invariante mede — e os números
acima passam a mentir sobre o run.

O invariante do `environment` **trava o limpador que reproduziu o run**,
conferido contra os artefatos do run (`mascara.csv`, `correcoes.csv`) no
momento do congelamento, 0 divergências célula por célula — não é o
invariante sozinho que prova isso: zerar `DEPENDENCIAS`/`FDS` na fixture
deixa `test_invariante.py`/`test_estatico_congelado.py` verdes do mesmo jeito,
porque nenhum dos sete números se move sem FD. E o invariante não prova que a
FD contribuiu no run: o gate de 100% (determinístico) reprovou `City→State` e
as regras intra-coluna de `State`/`Climate_Zone` marcam todas as 300 células,
o que anula a FD por construção (ver `docs/DECISOES.md#fd-no-limpador`). Quem
prova o mecanismo da FD no limpador — que ela detecta e corrige quando o
gate aprova — é `tests/test_equivalencia.py` e
`tests/test_empacotar.py::test_limpador_com_fd_injetada_corrige_o_environment`.

---

## 7. Políticas

### Comentários

- Docstring de **1 linha**. Se não cabe em uma linha, o raciocínio longo vai
  para `docs/DECISOES.md` e o código guarda um link para a âncora.
- Comentário inline: teto de **2 linhas**.
- **Não** comentar: histórico ("antes isso era..."), medição ("reduz 40% do
  tempo"), comparação com o ZeroDC, plano de IA ("aqui poderíamos...").
  Comentário descreve a mecânica do que está ali, ou não existe.
- Densidade atual: **13,0%** em 2.699 linhas, teto **14%** (ver
  `docs/DECISOES.md#teto-de-densidade-14`). `tests/medir_verbosidade.py`
  mede; use antes de commitar uma tarefa que mexe em muitos arquivos.

### Exceções

São **20**, sendo 17 no código do repositório e 3 dentro da string `_ESTATICO` de
`empacotar.py` — estes últimos rodam no limpador gerado, não aqui. A regra é:
exceção existe onde a falha tem um comportamento de degradação **nomeado**,
nunca como rede genérica. Cite função, não número de linha: linha muda, função
não.

| Onde | O que degrada, e para quê |
|---|---|
| **`pipeline.py::_processar_coluna`** (2) | **a fronteira de resiliência do run.** Detecção falhou → `detector_nulo`, a coluna não é marcada; refino falhou → fica a regra de 1-passe, que já foi validada. Não acrescente uma segunda fronteira paralela. |
| `config.py::_segundos` | `TIMEOUT_LLM` com lixo é ignorado e cai no default do provedor — variável de ambiente ruim não vira traceback no start do container |
| `correcao/cascata.py::rodar_cascata` | uma coluna ruim não derruba o run; as células marcadas dela viram flag |
| `correcao/cascata.py::rodar_coluna` | regra de correção que explode numa célula deixa a célula intacta, e ela escala |
| `deteccao/dependencia.py::detectar_dependencia` | mesma política de `rodar_cascata`, não uma segunda fronteira: uma coluna cuja FD explode (proposta do agente, gate ou moda) fica **sem marca de FD**, com log, e o run continua para a próxima coluna |
| `deteccao/mascara.py::aplicar_detectores` | `detectar(col)` pode explodir na chamada ou na coerção; a coluna fica zerada com log, sem crash |
| `deteccao/oraculo.py::_classificar` | idem, no laço de refino: classificação que quebra vira "tudo falso" |
| `deteccao/refino.py::_tentar_update` (2) | revisão que não passa no portão mantém a regra residente e segue para a próxima iteração |
| `correcao/regras.py::traduzir` | código rejeitado vira feedback para o prompt de reparo — é o retry funcionando, não erro engolido |
| `sandbox.py::validar`, `sandbox.py::testar_fumaca` (3) | o portão *é* o lugar onde código gerado falha |
| `empacotar.py::_ESTATICO` (3) | o limpador gerado roda **sem** sandbox na máquina do usuário; regra que explode numa célula deixa a célula intacta |
| `main.py::main` | `DadosInvalidos` é erro do usuário: mensagem curta, sem traceback |
| `escolher_modelo.py::medir` (2) | é a ferramenta de medir um modelo: chamada que falha conta como schema não preenchido, código rejeitado conta como portão não passado. Aqui a falha **é** o dado. |

### Loops

São **50** (incluindo `avaliar_limpador.py` e `escolher_modelo.py`) — três a mais
que antes de `fd-no-limpador`: `_mascara` e `_desvios_da_moda`, novos em
`_ESTATICO`, mais o `for` que `gerar_limpador` usa para coletar
`trabalho.dependencia` em `DEPENDENCIAS`. Colapse loops **paralelos** —
dois `for` sobre a mesma sequência viram um. Mantenha os que **carregam
estado** entre iterações: o escalonamento célula a célula da cascata
(`pendentes`/`restantes`) e o loop de refino são estado acumulado, e fundi-los
com outra coisa esconde a mecânica que eles implementam. Os dois loops de
`deteccao/dependencia.py` — o `for trabalho` de `detectar_dependencia`
(acumula `mascara` coluna a coluna) e o `for valor_det` de `_desvios_da_moda`
(varre os grupos distintos do determinante) — são da mesma família: nenhum
dos dois é paralelo a outro `for` da função.

---

## 8. A costura cross-column (o próximo projeto)

A detecção hoje é **intra-coluna**: o contrato é
`detectar(col) -> pd.Series[bool]` — recebe só a própria coluna e devolve um
booleano por linha, sem olhar as outras colunas. O **próximo projeto do
usuário** é trocar esse contrato por uma forma cross-column.

Essa restrição está concentrada em **10 pontos, 7 arquivos**. Ela está
concentrada de propósito: se uma tarefa espalhar esse conhecimento, o próximo
projeto passa a ter que redescobrir o contrato em vez de trocá-lo num lugar
só.

| # | Onde | O quê | Já preparado para a troca? |
|---|---|---|---|
| 1 | `limpeza/sandbox.py::validar` | `nome_funcao`/`nome_argumento` parametrizados | **sim** — troca sem tocar na lógica |
| 2 | `limpeza/sandbox.py::_ATRIBUTOS_SERIE` | allowlist de 17 atributos que bloqueia cross-row e travessia de `pd` | não — precisa de um modo novo |
| 3 | `limpeza/sandbox.py::materializar` | `pd` fora do namespace do exec | não |
| 4 | `limpeza/deteccao/regra.py` | o prompt declara `detectar(col)` (4 ocorrências) | não |
| 5 | `limpeza/deteccao/regra.py::DETECTA_NADA` | sentinela `"def detectar(col): ..."` | não |
| 6 | `limpeza/deteccao/refino.py` | `SISTEMA_UPDATE`/`HUMANO_UPDATE` repetem o contrato `detectar(col) -> pd.Series[bool]`, "NAO ve outras colunas" e a proibição de cross-row | não — o loop reescreve a regra a cada iteração |
| 7 | `limpeza/deteccao/oraculo.py` (`_classificar`, `_amostrar_oraculo`) | chama `funcao(pd.Series(valores))` sobre valores distintos, não sobre linhas | não — premissa intra-coluna na lógica, não só no prompt |
| 8 | `limpeza/deteccao/mascara.py` | aplica `detectar(col)` uma vez por coluna, posicional | não |
| 9 | `limpeza/esquemas.py::RegraDeteccao` | docstring da classe e `description` do campo `codigo` repetem o contrato | não |
| 10 | `limpeza/empacotar.py` | cópia congelada da mesma aplicação, no limpador gerado | não |

**Continua 10 pontos, 7 arquivos — a detecção por FD (`deteccao/dependencia.py`)
não é um 11º ponto.** Ela não conhece o contrato `detectar(col)`: não gera
código, não passa pelo portão AST, não lê nem escreve a assinatura de nenhuma
função — recebe a `Tabela` inteira e a máscara intra já pronta, e decide por
informação mútua e por um agente que escolhe entre nomes de coluna, nunca por
`ast.parse` de uma função gerada. `git grep -n 'nome_funcao="detectar"' --
"limpeza/"` acha só `limpeza/deteccao/regra.py:103` — nenhuma ocorrência em
`deteccao/dependencia.py` nem em `correcao/fd.py`.

**Regra para qualquer tarefa daqui em diante:** esses 10 pontos continuam
sendo os *únicos* lugares que sabem que a detecção é intra-coluna. Em
particular, `Detector` guarda `codigo`/`funcao` sem interpretar a assinatura
(seção 4). A máscara, porém, **não** é `DataFrame[bool]` dentro da POC: ela
nasce `dtype=int` em `deteccao/mascara.py`, e `correcao/cascata.py` e
`metricas.py` a comparam com `== 1`; só o limpador empacotado
(`empacotar.py::_mascara`) reconstrói a sua própria versão `bool`. Não
acrescente um décimo primeiro ponto.

Lista original: `docs/superpowers/specs/2026-08-30-simplificacao-limpador-design.md`,
seção 9. Racional: `docs/DECISOES.md#deteccao-intra-coluna`.

---

## 9. Onde não mexer sem ler antes

**`limpeza/sandbox.py`** — é o portão de segurança. Todo código Python escrito
por LLM passa por ele antes de ser executado: allowlist de builtins, proibição
de import, de dunder e de `eval`/`exec`/`open`/`getattr`; namespace do exec
sempre descartável, nunca `globals()`. Afrouxar qualquer uma dessas listas é
uma decisão de segurança, não uma conveniência de implementação. Leia
`docs/DECISOES.md#portao-ast` e `#modo-serie` antes.

**`limpeza/empacotar.py::_ESTATICO`** — é copiada **byte a byte** para o
limpador gerado, e **as duas** fixtures congeladas —
`tests/fixtures/limpador_beers_congelado.py` e
`tests/fixtures/limpador_environment_congelado.py` — terminam exatamente com
ela. `tests/test_estatico_congelado.py` trava isso para as duas.
Se você editar `_ESTATICO`, o teste quebra — e a resposta certa quase nunca é
atualizar a fixture, porque a fixture é o que `test_invariante.py` mede.
Quando for mesmo o caso, atualize **as duas**, e só na metade de baixo (a
metade de cima — `DEPENDENCIAS`, `FDS`, `_DETECTORES`, `_CORRETORES` — é
específica de cada run e não muda). Divergir de `_ESTATICO` faz o limpador
mentir sobre o run.

**Sobre o teste de embeddings:** `tests/test_embeddings.py` exercita o modelo
MiniLM ONNX **real** e usa `pytest.mark.skipif` quando o modelo não está no
disco (`MODELO_EMBEDDING` no `.env`). Numa máquina sem o modelo ele **pula** —
`skip` não é cobertura. Se você mexeu em `amostragem.py::Embedder` e a suíte
passou verde, confirme que esse teste rodou (`pytest -rs` lista os pulados)
antes de dizer que está coberto.

---

## 10. Idioma

**Português do Brasil em tudo:** nomes de função e variável, docstrings,
comentários, mensagens de log, documentos, mensagens de commit — e as
respostas ao usuário nesta conversa.

Exceção única: identificadores que vêm de biblioteca externa
(`random_state`, `keep_default_na`, `pd.Series`) e as colunas do dataset, que
são dados de entrada. Não traduza esses.

O código evita acentos em identificadores e em strings de log (o stdout do
Windows abre em cp1252); os **documentos** usam acentuação normal.
