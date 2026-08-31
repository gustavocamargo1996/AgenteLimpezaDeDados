# POC — gerador de limpadores de dados

Entra um CSV **sujo** e um CSV **limpo** de referência. Sai um
`limpador_<dataset>_<carimbo>.py`: um script Python autônomo, sem dependência
deste repositório, que aplica as regras aprendidas a qualquer tabela da mesma
origem.

**O produto é o programa de limpeza, não a tabela limpa.** A tabela corrigida
é subproduto — serve para medir a qualidade do limpador.

Cada execução entrega três coisas:

1. o **limpador** autônomo (`limpador_<dataset>_<carimbo>.py`);
2. a **cadeia de pensamento** que gerou cada regra (`cadeias_deteccao.md`, `cascata.md`);
3. a **qualidade** do limpador no dataset usado, em precisão/recall/F1 de reparo.

## Os três papéis do `--limpo`

O CSV de referência tem três papéis distintos, e a distinção é o centro do
método:

| Papel | O que consome | Quanto usa |
|---|---|---|
| **Orçamento de rotulagem** | os representantes do KMeans, rotulados par a par | ~10 células por coluna — simula um humano rotulando |
| **Gates da cascata** | as mesmas células rotuladas | uma regra só é aplicada se acertar **100%** delas |
| **Gabarito da métrica** | a tabela limpa completa | só para **medir**, nunca para detectar |

Só o terceiro papel exige o CSV limpo inteiro. Os dois primeiros operam sobre
as poucas linhas efetivamente mostradas ao agente — é por isso que o custo
escala com o número de rótulos, não com o tamanho da tabela. O CSV limpo nunca
entra na detecção nem na correção fora do orçamento rotulado; se entrasse, a
POC mediria memorização.

## Como funciona

```
CSV sujo + CSV limpo
  → carga literal (keep_default_na=False)
  → embeddings MiniLM ONNX local → KMeans sobre os valores DISTINTOS
  → representantes (o mais atípico e o mais típico de cada grupo) + rótulos
  → DETECÇÃO: agente escreve detectar(col), refinada por N iterações com oráculo
  → máscara 0/1 do dataset
  → CORREÇÃO em cascata: regra de código → dependência funcional → flag
       (cada camada com gate de 100% no rotulado; célula intacta escala)
  → limpador autônomo empacotado + métricas + cadeias
```

Nenhuma célula recebe valor inventado: o que nenhuma camada resolve fica
sinalizado com o valor sujo original.

## Rodando

O `.venv` deste repositório foi criado com [uv](https://docs.astral.sh/uv/),
que **não instala `pip` dentro do venv** — `.venv/Scripts/python -m pip` falha
com `No module named pip`. Use o `uv`:

```bash
uv venv .venv
uv pip install -r requirements.txt
cp .env.example .env    # e cole sua OPENAI_API_KEY
```

Sem `uv`, o caminho da biblioteca padrão funciona igual (aí `pip` vem junto):

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Os comandos abaixo chamam `.venv/Scripts/python` direto, sem ativar o venv.

O `.env` também aponta `MODELO_EMBEDDING` para a pasta local do MiniLM
(`tokenizer.json` + `onnx/model_O4.onnx`) — o modelo roda offline, sem torch e
sem download em tempo de execução.

Gerar um limpador (**gasta chamadas de API paga**):

```bash
.venv/Scripts/python main.py --sujo caminho/beers_dirty.csv --limpo caminho/beers_clean.csv
```

Opções:

```bash
--colunas ounces,state        # subconjunto; default 'todas'
--modelo gpt-4o               # outro modelo OpenAI
--saida runs                  # pasta base onde o run é escrito
--iteracoes-deteccao 5        # 1 = 1-passe sem loop; N>1 = active learning
--amostras-iter 2             # valores que o oráculo rotula por iteração
```

Cada execução escreve `runs/<carimbo>__e2e/` com o limpador gerado,
`cadeias_deteccao.md` e `cascata.md` (para ler), `deteccao_metricas.json` e
`correcao_metricas.json` (para diffar), e `mascara.csv` + `correcoes.csv` (as
duas tabelas).

## Avaliando um limpador — sem gastar API

`avaliar_limpador.py` aplica um limpador já gerado a um par sujo/limpo e mede o
F1 de reparo. Não faz nenhuma chamada de LLM:

```bash
.venv/Scripts/python avaliar_limpador.py \
  --limpador tests/fixtures/limpador_beers_congelado.py \
  --sujo tests/fixtures/beers_dirty_300.csv \
  --limpo tests/fixtures/beers_clean_300.csv
```

## Testes

```bash
.venv/Scripts/python -m pytest tests/ -v
```

87 testes, nenhum gasta API — os agentes LLM são substituídos por dublês e o
invariante mede um limpador congelado. Um teste (`tests/test_embeddings.py`)
exercita o modelo ONNX real e **pula** quando o modelo não está no disco; use
`pytest -rs` para ver os pulados.

## As métricas de reparo

Medidas por coluna e no total, sobre a tabela inteira:

- **erros** — células onde `sujo != limpo` (a verdade);
- **mudanças** — células que o limpador alterou;
- **TP** — mudanças que acertaram o valor limpo;
- **precisão** = TP / mudanças — *das que mexi, quantas acertei*;
- **recall** = TP / erros — *dos erros que existiam, quantos consertei*;
- **F1** — a média harmônica das duas;
- **flags** — células marcadas como erradas que nenhuma camada conseguiu
  consertar. Não são erro: são o método recusando-se a chutar.

Precisão e recall pedem leitura conjunta. O gate de 100% da cascata empurra a
precisão para perto de 1,0 de propósito: entre aplicar uma regra
parcialmente certa e não aplicar nada, a cascata prefere não aplicar e deixar a
célula escalar. O preço disso é recall baixo, e é um preço escolhido — sem ele
a POC estragaria células que já estavam corretas.

O invariante congelado do `beers` (300 linhas) mede exatamente isso: 495 erros,
121 mudanças, 121 TP, precisão 1,0, recall 0,2444, F1 0,3929, 71 flags. Esses
números estão travados em `tests/test_invariante.py` — alterá-los é bug, não
melhoria.

## Segurança

O pipeline executa código Python gerado por LLM. Antes de qualquer execução, o
código passa por um **portão AST** (`limpeza/sandbox.py`): allowlist explícita
de builtins, proibição de imports, de atributos dunder e de chamadas como
`eval`/`exec`/`open`/`getattr`; o namespace do `exec` é sempre descartável,
nunca `globals()`.

O portão reduz superfície mas **não é sandbox** — builtins seguem alcançáveis
em CPython por código determinado. Rode sobre dados públicos e revise o
limpador gerado antes de usar em produção: o arquivo final é Python legível
que roda **sem** sandbox.

A chave de API vive em `.env`, que está no `.gitignore`.

## Fora de escopo

FAISS e RAG de tuplas vizinhas · detecção cross-column (é o próximo projeto,
não um descarte — ver `CLAUDE.md` seção 7) · fallback por célula via LLM
(medido e rejeitado, ver `docs/DECISOES.md#veredito-contra-o-fallback`) ·
destilação professor→aluno · detecção de erro a partir do CSV limpo (ele serve
para *medir*, nunca para detectar).

## Onde ler mais

- **`CLAUDE.md`** — arquitetura, tipos, invariantes e políticas. É o ponto de
  partida para editar o código.
- **`docs/DECISOES.md`** — o porquê medido de cada parâmetro e cada gate. O
  código só carrega a mecânica; o raciocínio mora aqui.
- **`CHANGELOG.md`** — o que mudou e quando.

## Licença

[MIT](LICENSE). O código desta POC é original. O dataset `beers` e o modelo
MiniLM local são fontes externas somente-leitura, **não** redistribuídas aqui —
as condições de uso são as dos repositórios de origem.
