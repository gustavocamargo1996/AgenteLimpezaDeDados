# Provedor local (Ollama) e implantação em container — design

**Data:** 2026-09-19
**Estado:** aguardando revisão do usuário

## 1. Objetivo

Os dados que a POC processa **não podem sair da rede**. Hoje os quatro agentes
falam com a API da OpenAI, e os prompts carregam valores reais das células —
os representantes escolhidos pelo KMeans e, no orçamento, os pares
`sujo → limpo`. É exatamente esse dado que a restrição protege.

Este trabalho faz duas coisas:

1. põe os dois provedores — OpenAI e Ollama — atrás de um seletor, num
   repositório só;
2. empacota a POC num container e a implanta num stack do Portainer, ao lado
   de um serviço Ollama.

Os embeddings **já são locais** (MiniLM ONNX, sem rede). Metade do caminho já
estava pronta antes deste trabalho.

### O que NÃO muda

O caminho OpenAI continua funcionando exatamente como hoje. O seletor é
aditivo: com `PROVEDOR=openai`, o comportamento tem de ser indistinguível do
atual. Ver a seção 7.

## 2. O que a sondagem já respondeu

Antes de desenhar, uma sondagem descartável mediu se `ChatOllama` consegue
preencher os quatro schemas pydantic do projeto. Modelo `qwen2.5:3b`, em CPU,
três repetições por agente, com os **prompts reais** do repositório.

| Agente | Schema preenchido | Código aceito pelo portão AST |
|---|---|---|
| Detecção (`RegraDeteccao`) | 3/3 | **0/3** |
| Especificador (`RegraCorrecao`) | 2/3 | — |
| Tradutor (`CodigoGerado`) | 3/3 | **3/3** |
| Dependência funcional | 3/3 | — |

**Conclusão que autoriza este projeto:** `ChatOllama.with_structured_output`
preenche os schemas do repositório sem alterar uma linha deles. A viabilidade
técnica está demonstrada.

**Conclusão que molda o desenho:** com 3B, a detecção preenche o schema mas o
código gerado é sempre rejeitado pelo portão — sempre por *"o codigo deve
conter exatamente uma definicao de funcao e mais nada"*. O modelo acrescenta
`import re` ou uma constante. Com esse modelo, toda coluna cairia para
`DETECTA_NADA` e o limpador sairia vazio.

Isso não condena o Ollama: é falha de seguir instrução, que modelo maior
tende a resolver. Mas é **hipótese, não medição** — e é por isso que a
escolha de modelo vira uma ferramenta do repositório (seção 4).

**Conclusão de segurança:** o portão AST rejeitou o código ruim em vez de
executá-lo, e o pipeline degradou para "não detecta nada" em vez de quebrar.
A propriedade de segurança se mantém com modelo fraco — experimentar com
modelo local é seguro.

**Tempos medidos em CPU, para calibrar expectativa:** 26 s a 84 s por chamada
depois do modelo carregado; **808 s** no pior caso do especificador. Um run
completo do `beers` são ~100 chamadas.

## 3. O seletor de provedor

### Estado atual

Os quatro construtores de agente são **idênticos byte a byte**, exceto pelo
schema:

```python
def construir_agente(modelo: str | None = None):
    llm = ChatOpenAI(model=modelo or config.MODELO_LLM,
                     temperature=config.TEMPERATURA,
                     timeout=config.TIMEOUT_LLM, max_retries=2)
    return llm.with_structured_output(RegraDeteccao)   # <- só isto muda
```

Vivem em `limpeza/deteccao/regra.py`, `limpeza/correcao/regras.py` (dois) e
`limpeza/correcao/fd.py`.

### Desenho

Um módulo novo, `limpeza/llm.py`, é **o único lugar do repositório que sabe
que existe mais de um provedor**. Os quatro construtores viram uma linha:

```python
# limpeza/llm.py
def construir(schema, papel: str):
    """Devolve um agente que preenche `schema`, no provedor configurado."""

# limpeza/deteccao/regra.py
def construir_agente(modelo: str | None = None):
    return llm.construir(RegraDeteccao, "deteccao")
```

Nenhum outro módulo descobre que Ollama existe. É a mesma disciplina que a
seção 7 do `CLAUDE.md` impõe à costura cross-column: concentrar o
conhecimento num ponto em vez de espalhá-lo.

### Modelo por papel

A sondagem mostrou que os agentes têm exigências diferentes: o tradutor de
código foi 3/3 e a detecção 0/3 no portão. Amarrar todos a um modelo só
deixa o conjunto refém do pior caso.

Por isso a fábrica recebe `papel` — um de `deteccao`, `especificador`,
`codigo`, `fd` — e resolve o modelo nesta ordem:

1. `MODELO_<PAPEL>` (ex.: `MODELO_DETECCAO=qwen2.5-coder:14b`)
2. `MODELO_LLM` (o default de todos)
3. o default do provedor

Com OpenAI, ninguém define as específicas e tudo continua em `MODELO_LLM` —
exatamente o comportamento de hoje.

### Timeout por provedor

`TIMEOUT_LLM = 180` hoje. A sondagem registrou 808 s numa chamada local. Com
180, um run em Ollama falharia por timeout em chamadas que iam completar.

O timeout passa a ter default por provedor (`180` para OpenAI, `900` para
Ollama), sobreponível por `TIMEOUT_LLM`.

### Configuração

| Variável | Default | Para quê |
|---|---|---|
| `PROVEDOR` | `openai` | `openai` ou `ollama` |
| `OLLAMA_URL` | `http://localhost:11434` | endereço do serviço Ollama |
| `MODELO_LLM` | `gpt-4o-mini` | modelo de todos os papéis |
| `MODELO_DETECCAO` / `_ESPECIFICADOR` / `_CODIGO` / `_FD` | vazio | sobrepõe por papel |
| `TIMEOUT_LLM` | por provedor | sobrepõe o default |

`PROVEDOR` com valor desconhecido **falha na construção, com mensagem clara**
— não cai em silêncio para OpenAI, que mandaria dado para fora da rede.
Falhar alto é a única opção segura sob esta restrição.

### Ponto a verificar, não a assumir

`ChatOpenAI` recebe `max_retries=2`. Se `ChatOllama` aceita o mesmo parâmetro
precisa ser **verificado na implementação instalada**, não presumido. Se não
aceitar, a fábrica omite o parâmetro no ramo Ollama.

## 4. `escolher_modelo.py` — a sondagem vira ferramenta

A escolha de modelo é a decisão central deste projeto, e ela só se resolve no
servidor de destino. Nenhuma medição feita numa máquina de desenvolvimento
diz o que um modelo de 32B numa GPU vai fazer.

A sondagem descartável vira utilitário versionado, ao lado de
`avaliar_limpador.py` — mesma natureza: CLI que não faz parte do pipeline.

```bash
python escolher_modelo.py --modelo qwen2.5-coder:14b --repeticoes 3
```

Faz as doze chamadas (quatro agentes × três repetições) com os prompts reais
e reporta, por agente, quantas preencheram o schema e — nos dois que produzem
código — quantas passaram no portão AST. Em minutos, e sem gastar API.

**O critério de aprovação de um modelo é a linha da detecção.** Um modelo que
não gera `detectar(col)` aceito pelo portão produz limpador vazio,
independentemente de como se saia no resto.

## 5. O container da POC

| Camada | Tamanho |
|---|---|
| `python:3.12-slim` + `requirements.txt` | ~400 MB |
| `limpeza/`, `main.py`, `avaliar_limpador.py`, `escolher_modelo.py` | ~300 KB |
| MiniLM: `onnx/model_O4.onnx` + `tokenizer.json` | **43,8 MB** |
| **Total** | **≈ 500 MB** |

O modelo de embeddings vai **assado na imagem**, e não num volume: sob
restrição de privacidade, uma imagem que baixa modelo em runtime faz uma
chamada externa a cada start. Assada, o container nunca sai da rede. São
43,8 MB — a pasta do modelo tem 131 MB, mas o resto é formato PyTorch que a
POC não usa.

Os CSVs **não entram na imagem**. Entram por volume (seção 6), o que mantém
dado sensível fora de qualquer artefato que se possa copiar por engano.

Um run inteiro escreve **104 KB** de artefatos.

## 6. O stack do Portainer

### Premissa de acesso

O usuário **não administra o servidor** e opera pela interface web do
Portainer — o caso comum para quem recebe acesso ao Portainer, que existe
justamente para dar controle de containers sem distribuir shell e root.

O desenho assume esse cenário, o mais restrito. Um desenho que funciona só
com a web também funciona com SSH; o contrário não.

### Os serviços

```yaml
services:
  ollama:                      # serviço, sempre de pé
    image: ollama/ollama
    volumes: [ollama-modelos:/root/.ollama]
    # acesso a GPU: configurado por quem administra o host

  arquivos:                    # UI web para subir CSV e baixar artefato
    image: filebrowser/filebrowser
    volumes: [limpeza-dados:/srv]

  poc:                         # job one-shot
    build: .                   # o Portainer clona o repo; o contexto é a raiz
    environment:
      PROVEDOR: ollama
      OLLAMA_URL: http://ollama:11434
      SUJO: /dados/entrada/x_dirty.csv
      LIMPO: /dados/entrada/x_clean.csv
    volumes: [limpeza-dados:/dados]
    restart: "no"

volumes:
  ollama-modelos:              # 8 GB de modelos, persistem entre deploys
  limpeza-dados:               # CSVs de entrada e artefatos de saída
```

O volume dos modelos é o que mais economiza tempo: atualizar a POC não força
rebaixar gigabytes de modelo.

**O serviço `arquivos` pode ser dispensável.** O Portainer tem navegação de
volumes embutida, mas a disponibilidade varia por edição e versão. Verificar
na instância antes de montá-lo — é um serviço a menos.

### Variáveis de ambiente como default dos argumentos

Na interface do Portainer, editar variável de ambiente é fácil e editar o
comando é desconfortável. Como cada rodada usa um par de CSVs diferente, o
ciclo de trabalho vira "mudar dois valores e apertar start".

A mudança é **aditiva**: os argumentos continuam sendo a interface, e as
variáveis viram o default deles.

```python
ap.add_argument("--sujo", default=os.getenv("SUJO"))
# obrigatório apenas quando nem o argumento nem a variável existem
```

Pela linha de comando nada muda. Pelo Portainer, editar `SUJO`/`LIMPO` e
clicar start dispara o job. Os dois caminhos convivem sem estorvo.

### O ciclo de trabalho resultante

1. subir os dois CSVs pela UI de arquivos (ou pelo navegador de volumes)
2. editar `SUJO` e `LIMPO` no serviço `poc`
3. start
4. acompanhar pelo log do container
5. baixar os artefatos pela mesma UI

**Comportamento de job one-shot num compose, que é contra-intuitivo:** ao
subir o stack, o serviço `poc` executa uma vez e termina — com os valores de
`SUJO`/`LIMPO` que estiverem lá. O `restart: "no"` garante que ele não
reinicie sozinho. A partir daí, cada rodada é editar as duas variáveis e dar
**start** no container parado; não é preciso redeployar o stack.

Consequência a documentar no `README`: no primeiro deploy, ou os CSVs já
estão no volume, ou o primeiro run falha por arquivo inexistente — o que é
inofensivo (`DadosInvalidos`, mensagem curta, sem traceback) mas assusta quem
não espera.

### Entrega da imagem

**Build pelo próprio Portainer, a partir do Git.** O Portainer clona o
repositório e builda no servidor, então não há registry nem transferência de
arquivo, e atualizar é um redeploy depois do push.

**A verificar antes:** se o repositório for privado, o Portainer precisa de
credencial de acesso; e o servidor precisa alcançar o GitHub. Se qualquer uma
das duas não valer, a alternativa é registry.

## 7. Invariante: o caminho OpenAI não pode mudar

Com `PROVEDOR=openai`, o comportamento tem de ser **indistinguível do atual**.

- Os **89 testes** continuam passando.
- `tests/test_invariante.py` continua travando `erros=495, mudancas=121,
  tp=121, precisao=1.0, recall=0.2444, f1=0.3929, flags=71`.
- `tests/test_estatico_congelado.py` continua verde: nada aqui toca
  `_ESTATICO`.

O seletor é aditivo. Se um teste mudar de resultado, o seletor vazou para
onde não devia.

Testes novos, todos sem rede:

- a fábrica devolve o cliente certo para cada valor de `PROVEDOR`;
- `PROVEDOR` desconhecido levanta erro com mensagem clara;
- a resolução de modelo respeita a ordem papel → geral → default;
- o timeout default muda com o provedor;
- `--sujo`/`--limpo` leem de `SUJO`/`LIMPO` quando o argumento falta, e o
  argumento vence a variável quando os dois existem.

## 8. Fora de escopo

- **Melhorar a qualidade das regras.** Este trabalho troca o provedor; não
  persegue recall.
- **Retry por falha de structured output.** A sondagem viu o especificador
  falhar 1 vez em 3 com modelo de 3B. O pipeline já degrada por conta disso
  (`cascata.rodar_coluna` absorve). Se retry é necessário depende do modelo
  escolhido no servidor — decidir com dado, não agora.
- **API HTTP ou UI para a POC.** Avaliado e descartado: uma pessoa, com
  acesso ao Portainer, disparando runs ocasionais. Um endpoint que recebe
  caminhos e dispara execução de código escrito por LLM é superfície que este
  caso não justifica.
- **Configurar GPU no host.** Depende de quem administra o servidor.
- **Detecção cross-column** — é o projeto seguinte, com o mapa na seção 7 do
  `CLAUDE.md`.

## 9. Riscos

| Risco | Mitigação |
|---|---|
| Nenhum modelo local gera detecção que passe no portão | `escolher_modelo.py` mede isso em minutos, antes de qualquer run caro. Se nenhum passar, a decisão volta para o usuário com dado na mão. |
| O seletor altera o caminho OpenAI sem querer | Os 89 testes e o invariante rodam a cada fase (seção 7) |
| `PROVEDOR` errado manda dado para fora da rede | Valor desconhecido falha alto na construção, nunca cai para OpenAI em silêncio |
| `ChatOllama` não aceitar `max_retries` | Verificar na implementação instalada; omitir o parâmetro no ramo Ollama se não aceitar |
| Timeout curto derruba run local que ia completar | Default por provedor: 900 s no Ollama, medido contra os 808 s da sondagem |
| Repositório privado impede o build por Git | Verificar credencial no Portainer; alternativa é registry |
| Portainer sem navegação de volumes | O serviço `arquivos` cobre; verificar antes para não subir serviço desnecessário |

## 10. Entregáveis

1. `limpeza/llm.py` — a fábrica
2. Os quatro construtores reduzidos a uma linha cada
3. `limpeza/config.py` — as variáveis da seção 3
4. `main.py` — `SUJO`/`LIMPO` como default dos argumentos
5. `escolher_modelo.py` — a sondagem versionada
6. `Dockerfile` + `.dockerignore`
7. `docker-compose.yml` — o stack da seção 6
8. Testes novos da seção 7
9. `requirements.txt` — `langchain-ollama`
10. `.env.example` — as variáveis novas
11. `CLAUDE.md` — seção nova sobre provedores; seção 2 atualizada
12. `README.md` e `CHANGELOG.md`
