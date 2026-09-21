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
  → DETECÇÃO intra-coluna: agente escreve detectar(col), refinada por N iterações com oráculo
  → DETECÇÃO por dependência funcional: sobre a máscara intra, marca desvio da moda condicionada
  → máscara 0/1 do dataset (as duas vias combinadas)
  → CORREÇÃO em cascata: regra de código → dependência funcional → flag
       (cada camada com gate de 100% no rotulado; célula intacta escala)
  → limpador autônomo empacotado + métricas + cadeias
```

Nenhuma célula recebe valor inventado: o que nenhuma camada resolve fica
sinalizado com o valor sujo original.

### Duas vias de detecção

A detecção roda em duas etapas, não uma. A **intra-coluna** vê o valor
isolado — `detectar(col)` não sabe o resto da linha. A **dependência
funcional** (`deteccao/dependencia.py`) vê a linha: propõe um determinante
por informação mútua e marca quem se desvia da moda condicionada do grupo —
o tipo de erro que só existe em contexto, como `State='MH'` numa linha cuja
`City='Bangalore'`. A FD roda **depois** da intra-coluna e recebe a máscara
dela como entrada (nunca a combinada), o que evita que a moda seja calculada
a partir do próprio resultado.

Quem lê `cadeias_deteccao.md` de um run passa a ver, por coluna, até duas
justificativas: a cadeia de pensamento da regra intra-coluna e — só quando a
FD passou no gate de 100% sobre o rotulado — o determinante escolhido e a
justificativa do agente para ele. Coluna sem FD aprovada não ganha essa
segunda seção; não é omissão, é o gate reprovando.

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
--modelo gpt-4o               # nos quatro papeis; sem ele vale MODELO_LLM
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
`PROVEDOR=ollama` nenhuma chave é usada e **o dado do usuário não sai da
rede** — é para isso que este caminho existe. A promessa é essa e só essa: a
máquina continua baixando modelo e imagem; o que nunca atravessa a fronteira
são as células da tabela, que é o que os prompts carregam. Pelo mesmo motivo,
`limpeza/config.py` desliga as quatro variáveis de tracing do
LangChain/LangSmith em todo start — ver
`docs/DECISOES.md#tracing-desligado`.

| Variável | Default | Para quê |
|---|---|---|
| `PROVEDOR` | `openai` | `openai` ou `ollama`. Valor desconhecido **falha**: não cai para `openai` em silêncio, porque isso mandaria dado para fora da rede |
| `OLLAMA_URL` | `http://localhost:11434` | onde o Ollama atende |
| `MODELO_LLM` | `gpt-4o-mini` | o modelo geral, dos quatro papéis. Com `PROVEDOR=ollama` é **obrigatório na prática**: o default é um nome da OpenAI e o Ollama responde 404 em todas as chamadas |
| `MODELO_DETECCAO`, `MODELO_ESPECIFICADOR`, `MODELO_CODIGO`, `MODELO_FD` | vazio | sobrepõem o geral **num papel só**; vazio usa o geral |
| `TIMEOUT_LLM` | 180s no `openai`, 900s no `ollama` | segundos de uma chamada. Os 900s não são folga: uma chamada em CPU já levou 808s. Valor não-inteiro ou `<= 0` é ignorado, sem traceback |

A precedência de modelo é **`--modelo` → `MODELO_<PAPEL>` → `MODELO_LLM`**:
`--modelo` na linha de comando vale para os quatro papéis de uma vez; sem ele,
cada papel usa a sua sobreposição, e quem não tiver uma cai no geral. Um run
em que **nenhuma** coluna recebeu regra de detecção — LLM fora do ar, modelo
inexistente, schema nunca preenchido — imprime no log:

```
  AVISO: 0/9 colunas receberam regra de deteccao; o limpador esta vazio.
```

O código de saída continua 0: o run completou, e o artefato que ele escreveu é
um limpador que não corrige nada. O aviso é o que separa esse caso de um run
bem-sucedido.

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

> **Nunca usou Portainer?** [`docs/GUIA_PORTAINER.md`](docs/GUIA_PORTAINER.md)
> é o passo a passo completo, do primeiro clique ao download do resultado,
> sem pressupor conhecimento de Docker. Esta seção aqui é a referência
> técnica; o guia é a receita.

A imagem traz a POC e o MiniLM dentro (732 MB de conteúdo, 255 MB
comprimido); os CSVs **não** entram nela — entram por volume, para que dado
sensível não fique preso num artefato que se copia por engano.

O container roda como usuário **não-root** (`limpeza`, uid 10001): o código
que o LLM escreve executa aqui dentro, e o portão AST de `limpeza/sandbox.py`
reduz superfície mas não é sandbox. Ver `docs/DECISOES.md#container-nao-root`.

### O build funciona direto, inclusive no Portainer

Os dois arquivos do MiniLM que o `Dockerfile` copia
(`all-MiniLM-L6-v2/onnx/model_O4.onnx`, ~45 MB, e
`all-MiniLM-L6-v2/tokenizer.json`) **são versionados neste repositório**. É
uma decisão de entrega: o Portainer builda **clonando o repositório**, e com a
pasta fora do Git o `COPY` falhava no servidor — as saídas eram copiar arquivo
por shell no host ou publicar a imagem num registry, exatamente o que o
usuário-alvo (que não administra o servidor e opera pela interface web) não
tem.

```bash
docker build -t limpeza-poc .
docker run --rm limpeza-poc --help
```

No Portainer: **Stacks → Add stack → Repository**, apontando para este
repositório e para `docker-compose.yml`. Nada precisa ser copiado à mão.

**Procedência e licença do modelo:** `all-MiniLM-L6-v2` é da
[sentence-transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
sob **Apache 2.0** — que permite a redistribuição. Os dois arquivos aqui são a
exportação ONNX (`model_O4.onnx`) e o tokenizer, sem modificação. O dataset
`beers` continua **não** redistribuído: as fixtures de teste são um recorte de
300 linhas, e as condições de uso são as da origem.

### O stack

`docker-compose.yml` sobe dois serviços:

| Serviço | O que é |
|---|---|
| `ollama` | o provedor local, sempre de pé; os modelos vivem no volume `ollama-modelos` e sobrevivem a redeploy |
| `poc` | a POC, **job one-shot** (`restart: "no"`): roda `main.py` uma vez e termina |

Há um terceiro serviço, `arquivos`, **comentado por segurança** — ver "Subindo
e baixando arquivo", abaixo.

O serviço `poc` lê os caminhos do ambiente — `SUJO`, `LIMPO` e `SAIDA`, todos
apontando para `/dados`, que é o volume `limpeza-dados`. Na linha de comando
nada muda: as variáveis são só o **default** de `--sujo`/`--limpo`/`--saida`.

O mesmo serviço já vem com `PROVEDOR: ollama`, `OLLAMA_URL:
http://ollama:11434` e `MODELO_LLM: qwen2.5-coder:14b`: dentro do stack o
caminho padrão é o local, sem chave de API. **O dado do usuário não sai da
rede** — é essa a propriedade, e só ela: o stack ainda puxa imagens do Docker
Hub, o Ollama baixa o modelo e checa versão. O que nunca sai são as células da
tabela, que é o que os prompts carregam.

**`MODELO_LLM` é obrigatório na prática com `PROVEDOR=ollama`.** O default de
`limpeza/config.py` é `gpt-4o-mini`, que é o nome do caminho OpenAI: pedir
esse nome ao Ollama devolve **404 em todas as ~100 chamadas do run**, e o run
termina com um limpador vazio (o log traz `AVISO: 0/N colunas receberam regra
de deteccao`). O `docker-compose.yml` já define `MODELO_LLM`; se você mexer
nele, defina um nome que o Ollama tenha.

### Antes do primeiro start útil: baixe o modelo

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5-coder:14b
```

O `qwen2.5-coder:14b` são **~9 GB** de download, uma vez só — ficam no volume
`ollama-modelos` e sobrevivem a redeploy. No Portainer o mesmo `pull` sai pelo
**console** do container `ollama` (Containers → ollama → Console → `/bin/sh`).
Sem esse passo, todas as chamadas voltam 404.

### Quanto tempo leva um run

**Horas, não minutos, em CPU.** Uma chamada ao `qwen2.5-coder:14b` sem GPU
mediu entre 26 e 84 segundos nesta POC, com um pior caso de **808 segundos** —
é por isso que o timeout default do Ollama é 900s e não 180s. Um run completo
faz da ordem de **100 chamadas** (quatro papéis, uma coluna por vez, mais o
loop de refino). Com GPU a conta muda de patamar; a reserva de GPU está
comentada no `docker-compose.yml` porque depende de configuração no host.

Não é travamento: acompanhe pelo log do container, que imprime uma linha por
coluna e por etapa.

### O ciclo de trabalho

1. subir os dois CSVs no volume `limpeza-dados` pelo **navegador de volumes do
   Portainer** (Volumes → `limpeza-dados` → Browse: dá upload e download de
   arquivo pela mesma tela);
2. editar `SUJO` e `LIMPO` no serviço `poc`;
3. **start** no container parado — não é preciso redeployar o stack;
4. acompanhar pelo log do container;
5. baixar `runs/<carimbo>__e2e/` pela mesma tela (um run inteiro dá ~104 KB).

### Subindo e baixando arquivo sem o Portainer

O `docker-compose.yml` traz um serviço `arquivos` (`filebrowser` em `:8080`)
**comentado**, e o comentário diz por quê:

- a imagem `filebrowser/filebrowser` foi **arquivada em 01/set/2026** — sem
  releases novos, sem correção para advisory aberto;
- estava fixada em `latest`, publicando em `0.0.0.0:8080` sem restrição de
  bind;
- e o que ela serve é o volume `limpeza-dados`: o CSV sujo, o CSV limpo e o
  `runs/<carimbo>/correcoes.csv`, que é **a tabela inteira, célula a célula**.

O caminho primário é o navegador de volumes do Portainer, acima. Se a sua
instância não o tiver, descomente o serviço **sabendo do risco** e, no mínimo,
prenda o bind a uma interface interna (`127.0.0.1:8080:80`) e fixe uma tag em
vez de `latest`.

### Volume de um deploy antigo pode recusar escrita

Se o volume `limpeza-dados` já existia de uma versão anterior da imagem (que
rodava como root), seus arquivos continuam `root:root` e a POC não-root falha
com `Permission denied`. Conserta-se uma vez, com um container avulso — no
Portainer, Containers → Add container, imagem `alpine`, com o volume montado
em `/dados` e o comando:

```
chown -R 10001:10001 /dados
```

Volume **novo** não precisa disso: o Docker o semeia com o dono do diretório
`/dados` da imagem, que já é `limpeza:limpeza`.

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

127 testes, nenhum gasta API — os agentes LLM são substituídos por dublês e o
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

FAISS e RAG de tuplas vizinhas · o contrato geral de detecção cross-column,
`detectar(df)` (a detecção por dependência funcional já cobre o único padrão
de erro cross-column presente nos datasets disponíveis; o contrato geral
continua o próximo projeto, não um descarte — ver `CLAUDE.md` seção 8 e
`docs/DECISOES.md#deteccao-por-fd`) · fallback por célula via LLM (medido e
rejeitado, ver `docs/DECISOES.md#veredito-contra-o-fallback`) · destilação
professor→aluno · detecção de erro a partir do CSV limpo (ele serve para
*medir*, nunca para detectar).

## Onde ler mais

- **`CLAUDE.md`** — arquitetura, tipos, invariantes e políticas. É o ponto de
  partida para editar o código.
- **`docs/DECISOES.md`** — o porquê medido de cada parâmetro e cada gate. O
  código só carrega a mecânica; o raciocínio mora aqui.
- **`CHANGELOG.md`** — o que mudou e quando.

## Licença

[MIT](LICENSE). O código desta POC é original.

O modelo de embeddings **é** redistribuído aqui, em
`all-MiniLM-L6-v2/`: são a exportação ONNX (`onnx/model_O4.onnx`) e o
tokenizer do
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
sem modificação, sob **Apache 2.0** — licença que permite a redistribuição. É
o que faz o build por Git do Portainer funcionar sem shell no servidor.

O dataset `beers` **não** é redistribuído: as fixtures de teste são um recorte
de 300 linhas para o invariante, e as condições de uso são as do repositório
de origem.
