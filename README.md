# POC — Agente gerador de cadeias de pensamento para regras de correção

Inspirada no [ZeroDC](https://github.com/YangChen32768/ZeroDC), clonado em `../ZeroDC`.
O produto **é a cadeia de pensamento**, não a tabela limpa: o objetivo é ler como um
LLM raciocina sobre dados sujos sob dois regimes de informação diferentes.

## Como funciona

```
CSV → embeddings MiniLM (local) → KMeans agrupa valores distintos
   → representantes (os mais atípicos + os mais típicos de cada grupo)
   → AGENTE 1 (especificador): cadeia de pensamento + regra em JSON
   → AGENTE 2 (tradutor):      JSON → função Python
   → portão AST → avaliação no holdout → artefatos
```

**Dois agentes em série, de propósito.** O especificador infere; o tradutor traduz.
Quando a correção sai errada, você sabe se falhou a *inferência* ou a *tradução* —
diagnóstico que um agente único não te dá.

**Dois modos, para comparar:**

| Modo | O agente vê | Pergunta que responde |
|---|---|---|
| `blind` | só os valores sujos | o LLM consegue *descobrir* o defeito sozinho? |
| `budget` | os pares `sujo → correto` das representantes | dado o gabarito, ele reconstrói o raciocínio? |

O `budget` é o orçamento de rotulagem do ZeroDC: o custo escala com o número de
rótulos, não com o tamanho da tabela.

## Rodando

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env    # e cole sua OPENAI_API_KEY
```

Antes de gastar API, valide o ambiente — roda tudo que não precisa de LLM e mostra
exatamente o que o KMeans vai entregar ao agente 1:

```bash
.venv/Scripts/python verificar_ambiente.py
```

Depois:

```bash
.venv/Scripts/python main.py                          # os dois modos + comparativo
.venv/Scripts/python main.py --modo budget            # um modo só
.venv/Scripts/python main.py --colunas ounces,state   # subconjunto
.venv/Scripts/python main.py --colunas todas          # sem viés de seleção de coluna
.venv/Scripts/python main.py --dataset hospital       # outro dataset do clone ZeroDC
.venv/Scripts/python main.py --modelo gpt-4o          # outro modelo OpenAI
```

`--dataset` aceita qualquer pasta de `../ZeroDC/datasets/` que tenha o par
`<nome>_dirty.csv` + `<nome>_clean.csv`. Note que `COLUNAS_PADRAO` em `poc/config.py`
é específico do `beers` — para outro dataset, use `--colunas todas` ou passe a lista.

Cada execução escreve `runs/<data>__<modo>/` com `cadeias.md` (para ler),
`regras.json`, `codigo.json`, `corretores.py` e `metricas.json` (para diffar),
mais um `comparativo.md` quando os dois modos rodam.

## Módulos

| Arquivo | Papel |
|---|---|
| `poc/dados.py` | carga literal do CSV + partição amostra/holdout |
| `poc/embeddings.py` | MiniLM ONNX local — sem torch, sem download, sem rede |
| `poc/amostragem.py` | KMeans → representantes por coluna |
| `poc/identificador_regras.py` | **Agente 1** — cadeia de pensamento + JSON |
| `poc/gerador_codigo.py` | **Agente 2** — JSON → Python, com retry sobre rejeição |
| `poc/sandbox.py` | portão AST + namespace restrito |
| `poc/esquemas.py` | contrato pydantic entre os dois agentes |
| `poc/avaliacao.py` | acerto + dano no holdout |
| `poc/relatorio.py` | artefatos por execução |
| `poc/config.py` | caminhos, modelo, clusters e colunas processadas por default |

## As métricas: dois eixos

**Acerto e dano** — em qualquer escopo:

- **acerto** — entre as células erradas, quantas a regra consertou
- **dano** — entre as células que *já estavam certas*, quantas a regra estragou

A segunda existe porque sem ela a POC mente. Em `ounces` 100% das células estão
erradas: qualquer regra agressiva marca 100% de acerto e parece perfeita. Em `city`
só 5% estão erradas — a mesma regra acerta os 5% e destrói os 95%, com o acerto
continuando lindo.

**Duas escalas de holdout**, porque uma só não mede tudo:

| Escopo | O que exclui | O que responde |
|---|---|---|
| `generalizacao` | todas as linhas cujo **valor** foi mostrado | a regra vale para formas que não estavam na amostra? |
| `cobertura` | só as ~12 **linhas** exibidas | a regra, uma vez formulada, se aplica ao volume da coluna? |

A generalização é a medida honesta — testar sobre valor já mostrado mede memorização,
que é o furo que a auditoria encontrou no ZeroDC. Mas ela **degenera**: quando uma
classe de erro inteira é um único valor (`ibu` = `"N/A"` em 1.005 células, `state` = `""`
em 127), mostrar esse valor remove a classe toda e não sobra nada para medir. Nessas
colunas o relatório emite `NÃO MENSURÁVEL` e você julga pela cobertura.

A cobertura **não prova generalização** — o valor pode ser o mesmo que o agente viu.
Ela prova que a regra foi bem formulada, bem traduzida e não quebra no volume. Onde as
duas existem elas concordam (`abv`: 80,7% e 79,7%), o que valida o instrumento.

## Reavaliar sem gastar API

Mudou a métrica e quer os números novos das execuções antigas? A seleção do KMeans é
determinística (`random_state=0`) e independente do modo, então uma reavaliação
reproduz exatamente os mesmos representantes. As regras e o código ficam salvos em
`regras.json` e `codigo.json`:

```bash
.venv/Scripts/python main.py --reavaliar runs/2026-07-23_1133__budget
```

Recalcula tudo e reescreve os artefatos, sem uma única chamada de LLM.

## Variância entre execuções — leia antes de citar qualquer número

**Temperatura 0 não dá determinismo.** Duas execuções de 23/07 (11h33 e 17h11) tinham
entrada idêntica — mesmo dataset, mesmo modelo, mesmo prompt, e os mesmos
representantes, já que o KMeans usa `random_state=0`. Ainda assim:

| Coluna (budget) | 11h33 | 17h11 | |
|---|---|---|---|
| `abv` | 80,7% | 80,7% | estável |
| `ibu` | 100% | 100% | estável |
| `ounces` | 0,4% | 0,4% | estável |
| `city` | **100%** | **11,3%** | **volátil** |
| `state` (blind) | `erro=sim` | `erro=NAO` | **volátil** |

A causa do caso `city` é legível no código gerado. Numa execução o agente generalizou
o sufixo, na outra decorou os dois exemplos que viu:

```python
re.sub(r'\s+[A-Z]{2}|\s+PA$', '', valor)          # generaliza  → 127/127
re.sub(r'^(.*?)(?:\s+(?:PA|IN))?$', ..., valor)   # decora      →  17/127
```

As 127 células erradas de `city` têm **35 sufixos distintos** (`CO` 17×, `CA` 13×,
`PA` 9×, `IN` 8×…). `PA` e `IN` foram os únicos que caíram na amostra do KMeans — o
segundo regex cobre exatamente eles, e 17/127 = 13,4% é o 11,3% medido no holdout.

Suspeita anotada, não comprovada: a instrução do system prompt *"não use conhecimento
de mundo que a amostra não mostra"* foi escrita contra alucinação, mas aqui pode estar
empurrando para hardcodar em vez de reconhecer "sigla de estado" como classe.

**Consequência prática:** nenhum número deste README é uma propriedade do método —
todos são de **uma execução**. Colunas com regra estruturalmente simples (`abv`,
`ibu`) repetem; colunas onde generalizar exige um salto (`city`) oscilam muito. Para
conclusão comparativa de verdade seria preciso rodar k vezes e reportar faixa, o que
esta POC **não faz**.

## O que esperar (previsões antes de rodar)

- **`ounces` no modo `blind` deve falhar.** 100% da coluna está corrompida, então a
  forma corrompida *é* a norma e não há atípico para o KMeans achar. Detecção por
  anomalia é cega para corrupção universal. No `budget` resolve de primeira.
- **`city` e `state` são insolúveis por construção.** Erram por valor ausente, e
  nenhuma regra local inventa a informação que não está lá — só uma dependência
  funcional (`brewery_name → city, state`) resolveria, e FD está fora de escopo.
  O acerto esperado é 0%; o resultado bom aqui é o agente **declarar** a
  impossibilidade em vez de inventar regra. Repare no `dano`.
  > **Refutada para `city`** (execução de 23/07). A corrupção acopla as duas colunas:
  > `state` fica vazia e a sigla vai colada em `city` (`"Export PA"` → `"Export"`).
  > Isso é sufixo, não ausência — solúvel por regra local, e existe regex que pega
  > 127/127. Se o agente *encontra* essa regra é outra história: numa execução fez
  > 100%, em outra 11,3% (ver "Variância entre execuções"). A previsão continua
  > valendo para `state`, que ficou em 0% nos dois modos e foi corretamente declarado
  > `nenhuma` pelo agente.
- **`abv` e `ibu` são o caso justo**, com 29% e 42% de erro — os dois modos têm chance.
- `abv` tem dois defeitos sobrepostos: sufixo `%` e artefato de float
  (`0.040999999999999995%` → `0.041`).

## Três desvios deliberados do ZeroDC

1. **KMeans sobre valores distintos, não sobre células.** 2.410 linhas de `ounces`
   colapsam em 25 valores; embedar 25 é o que revela o padrão, e é ~100× mais barato.
2. **Carga literal do CSV** (`keep_default_na=False`). O ZeroDC usa
   `read_csv(dtype=str).fillna('null')`, e o pandas converte `"N/A"` em NaN por
   default — o `"N/A"` do dirty e o vazio do clean viram o mesmo token e a diferença
   some. Na coluna `ibu` isso apaga **1.005 erros reais** antes de qualquer algoritmo
   rodar.
3. **Portão AST + namespace restrito** antes de executar código gerado. O
   `correction.py:790` do ZeroDC faz `exec(code, globals())` sem validação nenhuma,
   permitindo ao código gerado redefinir qualquer nome do módulo.

## Fora de escopo

FAISS e RAG de tuplas vizinhas · dependências funcionais · loop de refinamento
iterativo · destilação professor→aluno para modelo barato · detecção de erro (a POC
usa `clean.csv` para *medir*, nunca para detectar).

## Segurança

`exec` de código gerado por LLM. O portão AST reduz superfície mas **não é sandbox** —
builtins seguem alcançáveis em CPython por código determinado. Rode sobre `beers`
(dados públicos) e nada mais. A chave de API vive em `.env`, que está no `.gitignore`;
o repositório original tem a mesma chave literal em 10 pontos de 3 arquivos e aponta
para um relay de terceiro — não reuse nada de lá.
