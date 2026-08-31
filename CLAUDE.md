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

---

## 3. A arquitetura

```
main.py                     CLI: lê argumentos, chama a espinha, imprime
avaliar_limpador.py         CLI: F1 de reparo de um limpador já gerado (sem API)

limpeza/
  pipeline.py               A ESPINHA — as etapas em ordem. Comece a ler aqui.
  tipos.py                  Coluna, Tabela, Amostra, Detector, Correcao, Trabalho
  config.py                 tudo que é ajustável (modelo, clusters, iterações)
  esquemas.py               contratos pydantic entre os agentes LLM

  dados.py                  etapa 1 — carga literal dos dois CSVs → Tabela
  amostragem.py             etapa 2 — MiniLM ONNX + KMeans → representantes + rótulos
  deteccao/                 etapa 3 — onde estão os erros
    regra.py                  agente de detecção 1-passe → Detector
    oraculo.py                escolhe o que rotular na próxima iteração
    refino.py                 loop de active learning que melhora a regra
    mascara.py                aplica detectar(col) por coluna → DataFrame[bool]
  correcao/                 etapa 4 — como consertar
    cascata.py                código → FD → flag, com gate de 100% em cada camada
    regras.py                 agente especificador + agente tradutor (JSON → Python)
    fd.py                     dependência funcional: MI, proposta, gate, aplicação
  sandbox.py                portão AST + namespace restrito (roda em toda etapa que gera código)
  empacotar.py              etapa 5 — escreve o limpador_<nome>.py autônomo
  metricas.py               etapa 6 — P/R/F1 de detecção, acerto/dano de correção
  relatorio.py              artefatos por execução (cadeias_deteccao.md, cascata.md, ...)
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

    mascara = deteccao.construir_mascara(trabalhos, tabela)
    corrigido = correcao.rodar_cascata(trabalhos, tabela, mascara, agentes)
    limpador = empacotar.gerar_limpador(
        trabalhos, saida / f"limpador_{tabela.nome}_{carimbo}.py", tabela.nome
    )
    medida = metricas.avaliar(trabalhos, tabela, mascara, corrigido)
    relatorio.escrever(saida, tabela.nome, trabalhos, mascara, corrigido, medida)
    return limpador, medida
```

`_processar_coluna` (amostrar → detectar → refinar) é a única fronteira de
resiliência do run: ver a seção 6.

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
trocar o contrato de detecção sem tocar no tipo. Ver a seção 7.

---

## 5. Invariante de aceitação

`tests/test_invariante.py` aplica o limpador **congelado** de
`tests/fixtures/limpador_beers_congelado.py` às 300 linhas de
`beers_dirty_300.csv` e trava o resultado:

| Métrica | Valor travado |
|---|---|
| `erros` | 495 |
| `mudancas` | 121 |
| `tp` | 121 |
| `precisao` | 1.0 |
| `recall` | 0.2444 |
| `f1` | 0.3929 |
| `flags` | 71 |

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

`tests/test_estatico_congelado.py` guarda a outra ponta: a constante
`_ESTATICO` de `limpeza/empacotar.py` tem de continuar batendo byte a byte com
o final da fixture. Se ela divergir, o limpador que a POC gera hoje deixou de
ser o mesmo que o invariante mede — e o número acima passa a mentir sobre o
run.

---

## 6. Políticas

### Comentários

- Docstring de **1 linha**. Se não cabe em uma linha, o raciocínio longo vai
  para `docs/DECISOES.md` e o código guarda um link para a âncora.
- Comentário inline: teto de **2 linhas**.
- **Não** comentar: histórico ("antes isso era..."), medição ("reduz 40% do
  tempo"), comparação com o ZeroDC, plano de IA ("aqui poderíamos...").
  Comentário descreve a mecânica do que está ali, ou não existe.
- Densidade atual: **11,1%** em 2.497 linhas. `tests/medir_verbosidade.py`
  mede; use antes de commitar uma tarefa que mexe em muitos arquivos.

### Exceções

São **16**, sendo 13 no código do pipeline e 3 dentro da string `_ESTATICO` de
`empacotar.py` — estes últimos rodam no limpador gerado, não aqui. A regra é:
exceção existe onde a falha tem um comportamento de degradação **nomeado**,
nunca como rede genérica. Cite função, não número de linha: linha muda, função
não.

| Onde | O que degrada, e para quê |
|---|---|
| **`pipeline.py::_processar_coluna`** (2) | **a fronteira de resiliência do run.** Detecção falhou → `detector_nulo`, a coluna não é marcada; refino falhou → fica a regra de 1-passe, que já foi validada. Não acrescente uma segunda fronteira paralela. |
| `correcao/cascata.py::rodar_cascata` | uma coluna ruim não derruba o run; as células marcadas dela viram flag |
| `correcao/cascata.py::rodar_coluna` | regra de correção que explode numa célula deixa a célula intacta, e ela escala |
| `deteccao/mascara.py::aplicar_detectores` | `detectar(col)` pode explodir na chamada ou na coerção; a coluna fica zerada com log, sem crash |
| `deteccao/oraculo.py::_classificar` | idem, no laço de refino: classificação que quebra vira "tudo falso" |
| `deteccao/refino.py::_tentar_update` (2) | revisão que não passa no portão mantém a regra residente e segue para a próxima iteração |
| `correcao/regras.py::traduzir` | código rejeitado vira feedback para o prompt de reparo — é o retry funcionando, não erro engolido |
| `sandbox.py::validar`, `sandbox.py::testar_fumaca` (3) | o portão *é* o lugar onde código gerado falha |
| `empacotar.py::_ESTATICO` (3) | o limpador gerado roda **sem** sandbox na máquina do usuário; regra que explode numa célula deixa a célula intacta |
| `main.py::main` | `DadosInvalidos` é erro do usuário: mensagem curta, sem traceback |

### Loops

São **40** (incluindo `avaliar_limpador.py`). Colapse loops **paralelos** —
dois `for` sobre a mesma sequência viram um. Mantenha os que **carregam
estado** entre iterações: o escalonamento célula a célula da cascata
(`pendentes`/`restantes`) e o loop de refino são estado acumulado, e fundi-los
com outra coisa esconde a mecânica que eles implementam.

---

## 7. A costura cross-column (o próximo projeto)

A detecção hoje é **intra-coluna**: o contrato é
`detectar(col) -> pd.Series[bool]` — recebe só a própria coluna e devolve um
booleano por linha, sem olhar as outras colunas. O **próximo projeto do
usuário** é trocar esse contrato por uma forma cross-column.

Essa restrição está concentrada em **7 pontos, 3 arquivos**. Ela está
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
| 6 | `limpeza/deteccao/mascara.py` | aplica `detectar(col)` uma vez por coluna, posicional | não |
| 7 | `limpeza/empacotar.py` | cópia congelada da mesma aplicação, no limpador gerado | não |

**Regra para qualquer tarefa daqui em diante:** esses 7 pontos continuam sendo
os *únicos* lugares que sabem que a detecção é intra-coluna. Em particular,
`Detector` guarda `codigo`/`funcao` sem interpretar a assinatura (seção 4), e a
máscara é consumida como `DataFrame[bool]` pelas etapas 4, 5 e 6, que não
precisam saber como ela foi construída. Não acrescente um oitavo ponto.

Lista original: `docs/superpowers/specs/2026-08-30-simplificacao-limpador-design.md`,
seção 9. Racional: `docs/DECISOES.md#deteccao-intra-coluna`.

---

## 8. Onde não mexer sem ler antes

**`limpeza/sandbox.py`** — é o portão de segurança. Todo código Python escrito
por LLM passa por ele antes de ser executado: allowlist de builtins, proibição
de import, de dunder e de `eval`/`exec`/`open`/`getattr`; namespace do exec
sempre descartável, nunca `globals()`. Afrouxar qualquer uma dessas listas é
uma decisão de segurança, não uma conveniência de implementação. Leia
`docs/DECISOES.md#portao-ast` e `#modo-serie` antes.

**`limpeza/empacotar.py::_ESTATICO`** — é copiada **byte a byte** para o
limpador gerado, e a fixture congelada
`tests/fixtures/limpador_beers_congelado.py` termina exatamente com ela.
`tests/test_estatico_congelado.py` trava isso. Se você editar `_ESTATICO`, o
teste quebra — e a resposta certa quase nunca é atualizar a fixture, porque a
fixture é o que `test_invariante.py` mede. Divergir de `_ESTATICO` faz o
limpador mentir sobre o run.

**Sobre o teste de embeddings:** `tests/test_embeddings.py` exercita o modelo
MiniLM ONNX **real** e usa `pytest.mark.skipif` quando o modelo não está no
disco (`MODELO_EMBEDDING` no `.env`). Numa máquina sem o modelo ele **pula** —
`skip` não é cobertura. Se você mexeu em `amostragem.py::Embedder` e a suíte
passou verde, confirme que esse teste rodou (`pytest -rs` lista os pulados)
antes de dizer que está coberto.

---

## 9. Idioma

**Português do Brasil em tudo:** nomes de função e variável, docstrings,
comentários, mensagens de log, documentos, mensagens de commit — e as
respostas ao usuário nesta conversa.

Exceção única: identificadores que vêm de biblioteca externa
(`random_state`, `keep_default_na`, `pd.Series`) e as colunas do dataset, que
são dados de entrada. Não traduza esses.

O código evita acentos em identificadores e em strings de log (o stdout do
Windows abre em cp1252); os **documentos** usam acentuação normal.
