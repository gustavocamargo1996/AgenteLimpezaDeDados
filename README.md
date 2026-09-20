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

A chave só é necessária com `PROVEDOR=openai`, que é o default. Com um modelo
local não há chave nenhuma — ver "Rodando com modelo local", abaixo.

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
--modelo gpt-4o               # outro modelo, no provedor ativo
--saida runs                  # pasta base onde o run é escrito
--iteracoes-deteccao 5        # 1 = 1-passe sem loop; N>1 = active learning
--amostras-iter 2             # valores que o oráculo rotula por iteração
```

Cada execução escreve `runs/<carimbo>__e2e/` com o limpador gerado,
`cadeias_deteccao.md` e `cascata.md` (para ler), `deteccao_metricas.json` e
`correcao_metricas.json` (para diffar), e `mascara.csv` + `correcoes.csv` (as
duas tabelas).

## Rodando com modelo local

A POC fala com dois provedores, e quem escolhe é o `.env`. Com
`PROVEDOR=ollama` nenhuma chave é usada e **nenhum dado sai da rede** — é para
isso que este caminho existe.

| Variável | Default | Para quê |
|---|---|---|
| `PROVEDOR` | `openai` | `openai` ou `ollama`. Valor desconhecido **falha**: não cai para `openai` em silêncio, porque isso mandaria dado para fora da rede |
| `OLLAMA_URL` | `http://localhost:11434` | onde o Ollama atende |
| `MODELO_LLM` | `gpt-4o-mini` | o modelo geral, dos quatro papéis |
| `MODELO_DETECCAO`, `MODELO_ESPECIFICADOR`, `MODELO_CODIGO`, `MODELO_FD` | vazio | sobrepõem o geral **num papel só**; vazio usa o geral |
| `TIMEOUT_LLM` | 180s no `openai`, 900s no `ollama` | segundos de uma chamada. Os 900s não são folga: uma chamada em CPU já levou 808s |

Com o Ollama de pé na máquina, baixe o modelo e aponte o `.env` para ele:

```bash
ollama pull qwen2.5-coder:14b
```

```
PROVEDOR=ollama
MODELO_LLM=qwen2.5-coder:14b
OLLAMA_URL=http://localhost:11434
```

Daí em diante os comandos são os mesmos — `main.py` não sabe qual provedor
está atrás.

### Antes de gastar um run: o modelo serve?

Modelo pequeno costuma preencher o schema e mesmo assim escrever código que o
portão AST rejeita. `escolher_modelo.py` mede isso em minutos, com os prompts
reais e amostras fixas do `beers`, sem tocar no dataset do usuário:

```bash
.venv/Scripts/python escolher_modelo.py --modelo qwen2.5-coder:14b
```

Sai uma linha por papel — quantas vezes o schema foi preenchido, e nos dois
papéis que geram código quantas vezes o portão aceitou:

```
  papel               schema    portao AST
  deteccao             3/3             0/3
  especificador        2/3               -
  codigo               3/3             3/3
  fd                   3/3               -

  NAO serve: sem deteccao aceita pelo portao, o limpador sai vazio.
```

**O critério é a linha da detecção.** Um modelo que não gera `detectar(col)`
aceito pelo portão produz limpador vazio, por melhor que seja nos outros três
papéis: sem regra de detecção não há máscara, e sem máscara a cascata não tem
célula para corrigir. O placar acima é real — é o do `qwen2.5:3b`, e é por
isso que ele não foi adotado.

`--repeticoes` muda o denominador (3 por padrão) e `--modelo` vale para os
quatro papéis de uma vez.

## Em container / Portainer

A imagem traz a POC e o MiniLM dentro (732 MB de conteúdo, 255 MB
comprimido); os CSVs **não** entram nela — entram por volume, para que dado
sensível não fique preso num artefato que se copia por engano.

### O build precisa do modelo no contexto

O `Dockerfile` copia dois arquivos que **não estão versionados** (a pasta está
no `.gitignore`). Sem eles o build falha no `COPY`. Copie-os para o contexto
antes:

```bash
mkdir -p all-MiniLM-L6-v2/onnx
cp "$MODELO_EMBEDDING/onnx/model_O4.onnx" all-MiniLM-L6-v2/onnx/
cp "$MODELO_EMBEDDING/tokenizer.json"     all-MiniLM-L6-v2/
docker build -t limpeza-poc .
```

(`$MODELO_EMBEDDING` é a pasta local do MiniLM, a mesma do `.env`; se a
variável não estiver exportada no shell, use o caminho literal.) Conferindo:

```bash
docker run --rm limpeza-poc --help
```

**No Portainer isso importa duas vezes**, porque ele builda clonando o
repositório: como a pasta do modelo não é versionada, o clone não a traz. Ou
os dois arquivos chegam ao contexto de build no servidor por outro caminho, ou
a imagem é buildada fora e entregue por registry.

### O stack

`docker-compose.yml` sobe três serviços:

| Serviço | O que é |
|---|---|
| `ollama` | o provedor local, sempre de pé; os modelos vivem no volume `ollama-modelos` e sobrevivem a redeploy |
| `arquivos` | `filebrowser` em `:8080`, para subir CSV e baixar artefato pelo navegador. Dispensável se o Portainer da instância navega volumes |
| `poc` | a POC, **job one-shot** (`restart: "no"`): roda `main.py` uma vez e termina |

O serviço `poc` lê os caminhos do ambiente — `SUJO`, `LIMPO` e `SAIDA`, todos
apontando para `/dados`, que é o volume `limpeza-dados` compartilhado com o
`arquivos`. Na linha de comando nada muda: as variáveis são só o **default**
de `--sujo`/`--limpo`/`--saida`.

O mesmo serviço já vem com `PROVEDOR: ollama`, `OLLAMA_URL:
http://ollama:11434` e `MODELO_LLM: qwen2.5-coder:14b`: dentro do stack o
caminho padrão é o local, sem chave de API e sem saída de rede.

```bash
docker compose config          # confere o YAML resolvido
docker compose up -d ollama arquivos
docker compose exec ollama ollama pull qwen2.5-coder:14b
```

Se a porta 8080 já estiver ocupada no host, o `arquivos` não sobe (`Bind for
0.0.0.0:8080 failed: port is already allocated`) — troque o mapeamento em
`docker-compose.yml`. É o único serviço que publica porta; `ollama` e `poc`
conversam pela rede interna do stack.

### O ciclo de trabalho

1. subir os dois CSVs em `http://<servidor>:8080` (ou pelo navegador de
   volumes do Portainer);
2. editar `SUJO` e `LIMPO` no serviço `poc`;
3. **start** no container parado — não é preciso redeployar o stack;
4. acompanhar pelo log do container;
5. baixar `runs/<carimbo>__e2e/` pela mesma UI (um run inteiro dá ~104 KB).

### O susto do primeiro deploy

**Ao subir o stack, o serviço `poc` executa uma vez, imediatamente** — com os
valores de `SUJO`/`LIMPO` que estiverem lá. Se os CSVs ainda não foram
enviados ao volume, esse primeiro run falha com:

```
ERRO: arquivo sujo nao encontrado: /dados/entrada_dirty.csv
```

É `DadosInvalidos`: mensagem curta, sem traceback, nada corrompido e nada
consumido. O container fica parado (`restart: "no"`) e é justamente nesse
estado que você sobe os CSVs, ajusta as variáveis e dá start. O susto é
esperado; a falha, inofensiva.

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

106 testes, nenhum gasta API — os agentes LLM são substituídos por dublês e o
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

### Um número é de uma execução, não uma propriedade do método

**`temperature=0` não garante determinismo.** Duas execuções com entrada
idêntica — mesmo dataset, mesmo modelo, mesmo prompt, e os mesmos
representantes, já que o KMeans usa `random_state=0` — podem produzir regras
diferentes e, com elas, métricas diferentes. Já foi medido nesta POC: uma
coluna oscilou entre 100% e 11,3% de acerto entre duas execuções do mesmo dia
(o histórico está no `CHANGELOG.md`, em `[0.1.0]`, "Variância entre
execuções").

Consequência prática: **nenhum número deste README é propriedade do método** —
todos vêm de uma execução. Colunas cuja regra é estruturalmente simples
repetem; colunas onde generalizar exige um salto oscilam muito. Conclusão
comparativa de verdade exigiria rodar k vezes e reportar a faixa, o que esta
POC **não faz**. O invariante acima escapa disso só porque mede um limpador
**congelado**, sem chamar LLM.

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
não um descarte — ver `CLAUDE.md` seção 8) · fallback por célula via LLM
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
