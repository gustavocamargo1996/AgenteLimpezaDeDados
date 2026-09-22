# Guia passo a passo — rodando a POC no Portainer

Este guia assume que você **nunca usou Docker nem Portainer**. Cada passo diz
onde clicar e o que digitar. Siga na ordem; os passos 1 a 4 você faz **uma vez
só**, e os passos 5 a 9 se repetem a cada dataset que você quiser limpar.

Se algo der errado, vá direto para a seção **Quando der errado** no fim — os
quatro problemas comuns estão lá com o conserto.

---

## Antes de começar: o que você precisa ter

| | |
|---|---|
| Acesso ao Portainer | um usuário com permissão de criar *stack* |
| Os dois CSVs | o **sujo** (com erros) e o **limpo** (a referência) |
| Paciência no primeiro dia | o passo 3 baixa ~9 GB e o passo 7 pode levar horas sem GPU |

**Você NÃO precisa de:** acesso SSH ao servidor, chave da OpenAI, instalar nada
na sua máquina.

### Os dois CSVs, e por que são dois

A POC não adivinha o que é erro. Ela aprende com um punhado de exemplos
corrigidos:

- **sujo** — a tabela com os defeitos, que você quer limpar;
- **limpo** — a mesma tabela com os valores corretos. Ela usa **~10 células
  por coluna** como orçamento de rotulagem (simulando um humano que corrigiu à
  mão) e a tabela inteira só para **medir** o resultado no fim.

Os dois precisam ter **as mesmas colunas, na mesma ordem, e o mesmo número de
linhas** — linha 5 do sujo é a linha 5 do limpo. Se não baterem, o run para
com uma mensagem clara, sem estragar nada.

---

## Passo 1 — Criar o stack

No Portainer, menu da esquerda:

**Stacks** → **+ Add stack**

Preencha:

| Campo | Valor |
|---|---|
| **Name** | `limpeza` |
| **Build method** | clique em **Repository** |
| **Repository URL** | `https://github.com/gustavocamargo1996/AgenteLimpezaDeDados` |
| **Repository reference** | `refs/heads/main` |
| **Compose path** | `docker-compose.yml` |

Se o repositório for **privado**, ligue **Authentication** e informe usuário e
um token do GitHub.

Role até o fim e clique em **Deploy the stack**.

O Portainer vai clonar o repositório e **construir a imagem** no servidor.
Isso leva alguns minutos na primeira vez. Não feche a página.

> **O nome do stack vira prefixo de tudo.** Chamando de `limpeza`, os
> containers nascem `limpeza-poc-1` e `limpeza-ollama-1`, e os volumes
> `limpeza_limpeza-dados` e `limpeza_ollama-modelos`. Se você der outro nome
> ao stack, troque o prefixo nos passos seguintes.

> **Não é preciso copiar nenhum arquivo à mão.** O modelo de embeddings que a
> POC usa está versionado no repositório justamente para que este passo
> funcione sozinho.

---

## Passo 2 — Entender o susto que vem agora

Assim que o stack sobe, o Portainer mostra dois containers:

| Container | Estado esperado |
|---|---|
| `limpeza-ollama-1` | **running** (verde) |
| `limpeza-poc-1` | **exited** (parado) — com erro |

**Isso está certo.** Abra o log do `poc` (**Containers** → clique nele → aba
**Logs**) e você verá:

```
ERRO: arquivo sujo nao encontrado: /dados/entrada_dirty.csv
```

Traduzindo: a POC subiu, procurou os CSVs, não achou (você ainda não os
enviou), avisou e parou. Nada foi corrompido, nada foi cobrado, nada precisa
ser consertado. É exatamente nesse estado — parado — que a gente quer que ele
fique até o passo 6.

---

## Passo 3 — Baixar o modelo de linguagem (~9 GB, uma vez só)

O `ollama` subiu **vazio**. Ele é o motor que vai escrever as regras, mas ainda
não tem nenhum modelo dentro.

**Containers** → clique em `limpeza-ollama-1` → botão **Console** → em
*Command* escolha `/bin/sh` → **Connect**.

Vai abrir um terminal preto dentro do container. Digite:

```sh
ollama pull qwen2.5-coder:14b
```

Aperte Enter e **espere**. São ~9 GB; a barra de progresso aparece sozinha.
Quando terminar, confirme que o modelo está lá:

```sh
ollama list
```

Tem de aparecer `qwen2.5-coder:14b` na lista. Pode fechar o console.

> **Esse é o passo que mais gente pula, e é o mais caro de pular.** Sem
> modelo, todas as ~100 chamadas do run voltam erro 404, o run termina, e você
> recebe um limpador **vazio** — com o container aparecendo **verde** no
> painel. O log avisa (`AVISO: 0/N colunas receberam regra de deteccao`), mas
> é fácil não ver.

O modelo fica guardado num volume separado (`ollama-modelos`). Atualizar a POC
depois **não** apaga o download.

---

## Passo 4 — Conferir se o modelo serve (10 minutos bem gastos)

Antes de torrar horas num run completo, vale medir se esse modelo consegue
fazer o trabalho. A POC tem uma ferramenta para isso.

**Containers** → `limpeza-poc-1` → **Console** → `/bin/sh` → **Connect**.

> Se o botão Console estiver desabilitado porque o container está parado,
> use **Duplicate/Edit** → **Deploy the container** só para ter o console, ou
> pule este passo e vá ao passo 5 — ele é recomendado, não obrigatório.

No terminal:

```sh
python escolher_modelo.py --modelo qwen2.5-coder:14b
```

Ele faz 12 chamadas e imprime uma tabela assim:

```
  papel               schema    portao AST
  deteccao             3/3           3/3
  especificador        3/3             -
  codigo               3/3           3/3
  fd                   3/3             -

  Serve: a deteccao gera codigo que o portao aceita.
```

**Olhe a linha `deteccao`, coluna `portao AST`.** Se ela vier `0/3`, esse
modelo **não serve**: ele vai produzir um limpador sem detecção
intra-coluna, por melhor que seja no resto. Volte ao passo 3 e baixe um
modelo maior ou diferente.

Se disser **"Serve"**, siga em frente.

---

## Passo 5 — Enviar os CSVs

**Volumes** (menu da esquerda) → clique em `limpeza_limpeza-dados` → botão
**Browse**.

Vai abrir um navegador de arquivos. Use **Upload** e envie os seus dois CSVs.

Anote os nomes exatos que aparecerem na lista — você vai precisar deles no
próximo passo. Por exemplo: `vendas_dirty.csv` e `vendas_clean.csv`.

> **Não renomeie seus arquivos** para casar com o exemplo. É mais seguro
> ajustar a configuração no passo 6 do que mexer nos nomes dos dados.

---

## Passo 6 — Apontar a POC para os seus arquivos

**Containers** → `limpeza-poc-1` → **Duplicate/Edit**.

Role até **Environment variables**. Você vai ver uma lista. Ajuste:

| Nome | Valor |
|---|---|
| `SUJO` | `/dados/vendas_dirty.csv` ← *o nome do SEU arquivo* |
| `LIMPO` | `/dados/vendas_clean.csv` ← *o nome do SEU arquivo* |
| `SAIDA` | `/dados/runs` ← **não mexa** |
| `MODELO_LLM` | `qwen2.5-coder:14b` ← o que você baixou no passo 3 |
| `PROVEDOR` | `ollama` ← **não mexa** |
| `OLLAMA_URL` | `http://ollama:11434` ← **não mexa** |

> O `/dados/` na frente é obrigatório: é o nome que o volume tem **dentro** do
> container. Na tela do passo 5 você vê `vendas_dirty.csv`; aqui você escreve
> `/dados/vendas_dirty.csv`. É o mesmo arquivo.

**Cuidado com o `PROVEDOR`:** se você digitar errado (`olama`, `Ollama `, vazio),
a POC **para com erro** em vez de tentar outra coisa. Isso é proposital — o
projeto existe para que seus dados não saiam da rede, e falhar alto é melhor
que mandar a tabela para uma API externa por causa de um typo.

Clique em **Deploy the container**.

---

## Passo 7 — Rodar e acompanhar

O container inicia sozinho depois do passo 6. Se estiver parado, use o botão
**Start**.

**Containers** → `limpeza-poc-1` → aba **Logs** → marque **Auto-refresh**.

Você vai ver, linha a linha:

```
[e2e] dataset=vendas | 2410 linhas | colunas=['preco', 'cidade', ...]
[e2e] modelo=qwen2.5-coder:14b | iteracoes_deteccao=5 amostras_iter=2

  preco: deteccao a partir de 12 representantes sujos...
  preco: refinando por 5 iteracoes com oraculo...
  preco: oraculo rotulou 10 valores / 125 celulas
  cidade: deteccao a partir de 12 representantes sujos...
```

### Quanto tempo isso leva

**Horas, se o servidor não tiver GPU.** Uma chamada mediu entre 26 e 808
segundos em CPU nesta POC, e um run faz da ordem de 100 chamadas. Com GPU a
conta muda de patamar.

**Silêncio no log não é travamento** — é uma chamada em andamento. Se ficar
inseguro, o container continua `running` no painel; enquanto estiver assim,
está trabalhando.

Você pode fechar o navegador e voltar depois. O container não depende da sua
sessão.

### Como saber que terminou

O container muda para **exited** e as últimas linhas do log trazem a tabela de
resultado:

```
  [e2e] artefatos em runs/2026-09-20_1430__e2e
  [e2e] limpador autonomo: runs/2026-09-20_1430__e2e/limpador_vendas_....py

  preco      | deteccao P=1.00 R=1.00 F1=1.00 | correcao acerto=87.5% dano=0.0%
  cidade     | deteccao P=1.00 R=0.11 F1=0.20 | correcao acerto=11.1% dano=0.0%
```

⚠️ **Se aparecer `AVISO: 0/N colunas receberam regra de deteccao`**, o limpador
saiu vazio. Quase sempre é o passo 3 (modelo não baixado) ou o `MODELO_LLM`
com nome errado no passo 6.

---

## Passo 8 — Baixar os resultados

**Volumes** → `limpeza_limpeza-dados` → **Browse** → entre na pasta `runs/` e
depois na pasta com o carimbo de data do seu run.

Baixe o que interessa:

| Arquivo | O que é |
|---|---|
| `limpador_<nome>_<data>.py` | **o produto**: um programa Python que roda sozinho, sem este projeto, em qualquer tabela igual à sua |
| `cadeias_deteccao.md` | o raciocínio do modelo para cada regra, em texto legível |
| `cascata.md` | o que cada camada de correção fez, coluna a coluna |
| `correcao_metricas.json` | quanto acertou e quanto estragou, por coluna |
| `deteccao_metricas.json` | precisão/recall/F1 da detecção |
| `correcoes.csv` | a tabela corrigida |
| `mascara.csv` | o mapa de quais células foram marcadas como erradas |

Um run inteiro dá cerca de **104 KB** — é tudo texto.

> **O produto é o `limpador_*.py`, não o `correcoes.csv`.** A tabela corrigida
> serve para você conferir a qualidade; o que você leva para usar de novo é o
> programa.

---

## Passo 9 — Rodar de novo, com outro dataset

Não precisa refazer nada dos passos 1 a 4. Só:

1. passo 5 — enviar os novos CSVs;
2. passo 6 — trocar `SUJO` e `LIMPO`;
3. passo 7 — start.

Os runs antigos continuam em `runs/`, cada um na sua pasta com carimbo de
data.

---

## Usando o limpador que você gerou

O `limpador_*.py` é autônomo: roda em qualquer máquina com Python e pandas, e
**não chama modelo nenhum** — as regras já estão escritas dentro dele.

```bash
python -c "
import pandas as pd, importlib.util
spec = importlib.util.spec_from_file_location('lm', 'limpador_vendas_2026-09-20_1430.py')
lm = importlib.util.module_from_spec(spec); spec.loader.exec_module(lm)
sujo = pd.read_csv('outra_tabela.csv', dtype=str, keep_default_na=False)
corrigido, flags = lm.aplicar(sujo)
corrigido.to_csv('resultado.csv', index=False)
print('celulas sinalizadas e nao corrigidas:', int(flags.sum().sum()))
"
```

`flags` é o mapa de células que o limpador **detectou como erradas mas não
soube corrigir**. Ele prefere sinalizar a chutar — e é por isso que o `dano`
fica em ~0%.

> **Leia o limpador antes de usar em produção.** Ele foi escrito por um modelo
> de linguagem e roda **sem** o portão de segurança que existe durante a
> geração. São poucas dezenas de linhas de `re` e `pandas`, legíveis.

---

## Quando der errado

### `ERRO: arquivo sujo nao encontrado: /dados/...`

O caminho em `SUJO`/`LIMPO` não bate com o nome do arquivo no volume.
Volte ao passo 5, confira o nome **exato** (maiúsculas e minúsculas contam), e
corrija no passo 6 lembrando do prefixo `/dados/`.

### `AVISO: 0/N colunas receberam regra de deteccao`

O limpador saiu vazio. Nesta ordem:

1. o modelo foi baixado? Console do `ollama` → `ollama list`;
2. o nome em `MODELO_LLM` bate **exatamente** com o da lista?
3. se os dois estiverem certos, rode o passo 4 — provavelmente o modelo não
   dá conta, e você precisa de um maior.

### `Permission denied` ao escrever em `/dados`

Acontece quando o volume foi criado por uma versão antiga da imagem, que
rodava como root. Conserto de uma vez:

**Containers** → **+ Add container**:

| Campo | Valor |
|---|---|
| Name | `conserta-permissao` |
| Image | `alpine` |
| Command (aba *Command & logging*) | `chown -R 10001:10001 /dados` |
| Volumes (aba *Volumes*) | mapear `limpeza_limpeza-dados` em `/dados` |

**Deploy**, espere terminar, e apague o container. Volume **novo** nunca
precisa disso.

### O container `poc` reinicia sozinho, sem parar

Não deveria — ele é um job de uma execução só (`restart: "no"`). Se estiver
reiniciando, alguém mudou a política de restart. Volte para `no`.

### Não encontro o botão *Browse* nos volumes

Algumas edições ou versões do Portainer não trazem o navegador de volumes.
Nesse caso o `docker-compose.yml` tem um serviço `arquivos` **comentado** que
resolve, mas leia o comentário antes de descomentar: a imagem está
**arquivada desde setembro de 2026**, sem correções de segurança, e serviria
seus CSVs numa porta aberta. Se for usar, prenda a porta a `127.0.0.1` e fixe
uma versão em vez de `latest`.

---

## Perguntas que todo mundo faz

**Meus dados saem da rede?**
Não. Os prompts carregam valores reais das suas células, e é por isso que o
modelo roda **dentro do servidor**. O stack baixa imagem e modelo da internet;
o que nunca sai são as células da sua tabela.

**Preciso de chave da OpenAI?**
Não, no stack. O `PROVEDOR` já vem como `ollama`. A chave só entra se você
mudar para `openai`, o que manda os dados para fora — e aí é decisão sua.

**Posso usar outro modelo?**
Pode. Baixe no passo 3, troque `MODELO_LLM` no passo 6, e rode o passo 4 antes
para saber se ele serve.

**Posso dar um modelo diferente para cada etapa?**
Pode: as variáveis `MODELO_DETECCAO`, `MODELO_ESPECIFICADOR`, `MODELO_CODIGO`
e `MODELO_FD` sobrepõem o geral naquele papel. Útil quando só a detecção
precisa de um modelo grande — ela é a etapa mais exigente.

**O run falhou no meio. Perdi tudo?**
As colunas que já tinham sido processadas entram no limpador assim mesmo; uma
coluna que falha vira "não marcada" e o run segue. O log diz qual falhou e por
quê.

**Por que o `recall` é baixo?**
Porque a POC só conserta o que consegue **provar** que está certo: uma regra
só é aplicada se acertar 100% das células de referência. O preço dessa
garantia é não consertar tudo — em troca, o `dano` fica em ~0%. Ela prefere
sinalizar a estragar.
