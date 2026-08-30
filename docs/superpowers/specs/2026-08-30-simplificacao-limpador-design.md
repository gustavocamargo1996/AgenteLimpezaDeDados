# Simplificação do gerador de limpadores — design

**Data:** 2026-08-30
**Estado:** aprovado para plano de implementação

## 1. Objetivo

O projeto cresceu em duas direções e hoje carrega dois produtos ao mesmo tempo:
um instrumento de pesquisa (comparação `blind` vs `budget`, jul/2026) e um
gerador de limpadores (caminho `--e2e`, ago/2026). São 4.969 linhas com 23,7%
de comentário, oito dicionários paralelos na função de orquestração, 22 blocos
`except` e 68 loops.

Este trabalho reduz o projeto a **um produto só**, com a ordem das etapas
legível na estrutura.

### O produto

```
entrada:  dataset sujo + amostra de dados limpos
saída:    1. limpador_<nome>.py autônomo, aplicável a tabelas de mesma origem
          2. a cadeia de pensamento que gerou cada regra
          3. a qualidade do limpador no dataset usado (P/R/F1 de reparo)
```

### Métricas de sucesso

| | hoje | alvo |
|---|---|---|
| Linhas | 4.969 | 2.100–2.400 |
| Comentário + docstring | 23,7% | ~9% |
| Blocos `except` | 22 (17 amplos) | ~7 |
| Loops | 68 | ~35 |
| Maior arquivo | 1.118 linhas | < 300 |

## 2. Invariante de aceitação

**Os números publicados têm de ser reproduzidos, dígito a dígito.** O artefato
"Gerador de Limpadores" publica três linhas de resultado. A referência
verificada nesta sessão, sobre `beers` (sufixo 300):

```
TOTAL | erros 495 | mudou 121 | TP 121 | precisão 100,0% | recall 24,4% | F1 39,3% | flags 71
```

Isso é o teste de regressão do refactor inteiro. Qualquer mudança que altere
esses números é um bug, não uma simplificação — exceto onde este documento
declara explicitamente o contrário (nada declara).

Os runs que produziram o artefato usaram `--e2e --sufixo 300` **com o loop de
refinamento ligado** (5–6 iterações, orçamento de ~10 valores por coluna) e o
**fallback desligado** (0 chamadas em todas as colunas). Confirmado nos
artefatos de `runs/2026-08-20_0927__e2e`, `..._0939__e2e` e `..._1033__e2e`.

## 3. Decisões fechadas

| # | Decisão | Consequência |
|---|---|---|
| 1 | O modo `blind` e toda a comparação saem | Sai o parâmetro `modo`, o `comparativo.md` e o `--reavaliar` |
| 2 | O loop de refinamento da detecção **fica e vira default** | `--iteracoes-deteccao` deixa de ser opt-in; era o que gerava os números publicados |
| 3 | O fallback por célula (camada 3) sai | Sai `fallback_celula.py`, a camada 3 da cascata e `--limite-fallback` |
| 4 | Entrada por caminho de arquivo | `--sujo <path> --limpo <path>`; sai o acoplamento a `ZERODC_DIR` |
| 5 | O raciocínio migra para `docs/DECISOES.md` | Código fica com comentários curtos sobre mecânica |
| 6 | Os testes viram suíte pytest | `verificar_e2e.py` + `verificar_ambiente.py` → `tests/` |
| 7 | Estrutura: `pipeline.py` linear + módulos por responsabilidade | A ordem vive num arquivo; nomes não carregam numeração |

### Decisão 3 — justificativa

O fallback não participou de nenhum número publicado, o artefato não o desenha
(a cascata dele termina em "Sinaliza / nunca chuta um valor"), e existe medição
de 7 runs contra ele: resolve ~100% das correções de `ounces`/`abv` com acerto
de ~0–0,5%, ao custo de milhares de chamadas, e é a única camada sem gate.

### Decisão 4 — o duplo papel do `--limpo`

O arquivo limpo é usado em três lugares hoje, e a distinção precisa sobreviver
à simplificação:

- **orçamento de rotulagem** (~10 valores por coluna) — legítimo, simula o
  humano rotulando;
- **gates da cascata** — uma regra só é aplicada se acertar 100% dos rótulos;
- **gabarito da métrica** — só para medir, nunca para detectar.

Quando `--limpo` traz a tabela completa, os três funcionam. Quando traz só uma
amostra parcial, o orçamento e os gates funcionam e a métrica reporta apenas
onde há verdade disponível.

## 4. Arquitetura alvo

```
main.py                  CLI: argumentos → pipeline → impressão
avaliar_limpador.py      mede o F1 de reparo de um limpador empacotado
limpeza/
  pipeline.py      (—)   A ESPINHA: as etapas em ordem, uma tela
  config.py        (—)   parâmetros (modelo, clusters, limiares)
  esquemas.py      (—)   contratos pydantic entre os agentes
  dados.py         (1)   carrega os CSVs, monta a coluna, separa o orçamento
  amostragem.py    (2)   embeddings MiniLM + KMeans → representantes
  deteccao/        (3)   onde estão os erros
    __init__.py          só re-exporta a API do pacote
    regra.py             agente detector (1 passe)
    refino.py            o loop com oráculo (hoje 562 das 776 linhas do módulo)
    mascara.py           aplica os detectores à tabela
  correcao/        (4)   como consertar
    __init__.py          só re-exporta a API do pacote
    cascata.py           orquestra: camada 1 → camada 2 → flag
    regras.py            agente 1 (spec) + agente 2 (código) — insumo da camada 1
    fd.py                dependência funcional + MI — a camada 2
  sandbox.py       (—)   portão AST (transversal)
  empacotar.py     (5)   escreve o limpador_<nome>.py autônomo
  metricas.py      (6)   P/R/F1 de detecção e de correção
  relatorio.py     (6)   artefatos .md/.json/.csv
tests/                   suíte pytest
docs/DECISOES.md         o porquê medido, fora do código
CLAUDE.md                orientação para sessões futuras
```

Os números na margem são posição na sequência e **não entram nos nomes de
arquivo** — inserir uma etapa não força renomear as seguintes.

### Destino de cada módulo atual

| Hoje | Vira | Nota |
|---|---|---|
| `main.py` (515) | `main.py` (CLI) + `limpeza/pipeline.py` | `rodar_e2e` tem 218 linhas |
| `poc/dados.py` | `limpeza/dados.py` | perde `particionar` (holdout do modo antigo) |
| `poc/amostragem.py` + `poc/embeddings.py` | `limpeza/amostragem.py` | consumidor único; uma etapa mental |
| `poc/deteccao.py` (776) | `limpeza/deteccao/` (3 arquivos) | 545 linhas de código puro; comprimir comentário não basta |
| `poc/identificador_regras.py` + `poc/gerador_codigo.py` | `limpeza/correcao/regras.py` | os dois agentes em série |
| `poc/cascata.py` | `limpeza/correcao/cascata.py` | perde a camada 3 |
| `poc/fd.py` + `poc/contexto.py` | `limpeza/correcao/fd.py` | a MI passa a ter um consumidor só |
| `poc/avaliacao.py` (262) | `limpeza/metricas.py` | perde `avaliar`/`Medida`/`Resultado` |
| `poc/relatorio.py` (304) | `limpeza/relatorio.py` | perde `escrever` e `comparativo` |
| `poc/esquemas.py` | `limpeza/esquemas.py` | perde `CorrecaoCelula` |
| `poc/sandbox.py`, `poc/empacotar.py`, `poc/config.py` | iguais | só compressão de comentário |
| `poc/fallback_celula.py` | **sai** | decisão 3 |
| `verificar_ambiente.py`, `verificar_e2e.py` | `tests/` | decisão 6 |

`deteccao/` e `correcao/` são subpacotes, e não arquivos únicos de ~380 e ~500
linhas, porque arquivo grande esconde a ordem interna — o problema que este
trabalho existe para resolver. Em ambos, `__init__.py` **só re-exporta**: não
carrega lógica, para que "um arquivo, um trabalho" valha sem exceção.

## 5. Fluxo de dados

### Causa raiz da verbosidade

Não existem tipos com nome. Os dados viajam como dicionários anônimos
aninhados (`saida_det["funcao"]`, `contexto_por_coluna[nome]["rotulados"]["rows"]`),
então cada consumidor re-checa chave e cada produtor monta o dict do seu jeito.
`rodar_e2e` mantém **oito dicionários paralelos** indexados por coluna:
`funcoes_deteccao`, `regras_deteccao`, `orcamento_deteccao`,
`historico_deteccao`, `contexto_por_coluna`, `deteccao_metricas`,
`correcao_metricas`, `trilhas`. Mantê-los em sincronia exige percorrer as
colunas várias vezes.

### Os tipos

```python
@dataclass
class Coluna:      nome; sujo; limpo; valores_distintos
@dataclass
class Amostra:     representantes; linhas; rotulados
@dataclass
class Detector:    codigo; funcao; cadeia; orcamento; historico
@dataclass
class Correcao:    passos; trilha; cadeia
@dataclass
class Medida:      deteccao; correcao
@dataclass
class Trabalho:    coluna; amostra; detector; correcao=None; medida=None
```

Os oito dicts paralelos viram uma lista de `Trabalho`, um por coluna. Os três
primeiros campos são preenchidos na passagem por coluna; `correcao` e `medida`
chegam nas etapas 4 e 6, que operam sobre a tabela inteira.

### A espinha

```python
def gerar_limpador(caminho_sujo, caminho_limpo, colunas, saida):
    tabela = dados.carregar(caminho_sujo, caminho_limpo, colunas)

    trabalho = []
    for coluna in tabela.colunas:                                  # ÚNICA passagem
        amostra  = amostragem.representantes(coluna)               # (2) sem LLM
        detector = deteccao.gerar(coluna, amostra)                 # (3) 1 chamada
        detector = deteccao.refinar(detector, coluna, amostra)     # (3) oráculo
        trabalho.append(Trabalho(coluna, amostra, detector))

    mascara   = deteccao.aplicar(trabalho, tabela)                 # (3)
    correcoes = correcao.cascata(trabalho, tabela, mascara)        # (4)
    limpador  = empacotar.escrever(trabalho, correcoes, saida)     # (5)
    medida    = metricas.avaliar(trabalho, tabela, mascara, correcoes)  # (6)
    relatorio.escrever(saida, trabalho, medida)
    return limpador, medida
```

## 6. Políticas

### Loops

1. Colapsar passagens paralelas — uma passagem por coluna em vez de quatro.
2. Vetorizar onde a semântica permite (máscara, aplicação de detectores).
3. **Manter onde o laço É a semântica.** A cascata escala célula a célula por
   definição: célula que a camada deixou intacta escala para a próxima.

O alvo não é "zero loops"; é "todo loop que sobrar tem um motivo legível".

### Exceções

Uma única fronteira de resiliência: **por coluna, em `pipeline.py`**. Coluna que
falhar cai para "não detecta nada", é registrada, e o run continua — o
comportamento de hoje, num lugar só.

**Ficam, porque definem comportamento e não são ruído defensivo:**

- o guarda por célula na aplicação do corretor (`cascata.py:94`): se um regex
  gerado estoura num valor, a célula fica intacta e as outras seguem. Removê-lo
  derrubaria a coluna e mudaria os números do artefato;
- os três `except` de `empacotar.py` e os três de `sandbox.py`: são portões de
  segurança.

### Comentários

1. Docstring de módulo: uma linha, qual etapa do pipeline ele é.
2. Docstring de função: uma linha, o que a função faz. Parâmetros só quando
   nome e tipo não bastam.
3. Comentário inline: só quando o código sozinho engana. Teto de 2 linhas.
4. Onde o porquê protege o código: uma linha com o número + link para o doc.

**Sai do código, sem exceção:** histórico de decisão ("DESVIO DELIBERADO DO
ZERODC"); medições de sessões passadas ("medido em 7 runs..."); comparações com
o ZeroDC (`correction.py:790`, `detection.py:392-459` e afins); referências a
planos e invariantes de sessões de IA ("invariante 6", "plano seção 8");
argumentação sobre alternativas descartadas.

Exemplo — `dados.py::carregar`, 12 linhas de docstring para 7 de código:

```python
# ANTES
"""Le os dois CSVs como texto literal, byte a byte.

DESVIO DELIBERADO DO ZERODC. O original faz `pd.read_csv(dtype=str).fillna('null')`,
e o pandas converte "N/A" em NaN por default -- entao o "N/A" do dirty e o vazio
do clean viram o MESMO token e a diferenca some. Na coluna `ibu` do beers isso
apaga 1005 erros reais (42% da coluna) antes de qualquer algoritmo rodar.
[...]
"""

# DEPOIS
"""Le os CSVs como texto literal e confere que estao alinhados por linha."""
# keep_default_na=False: sem isso o pandas converte "N/A" em NaN e apaga
# 1.005 erros reais de `ibu`. Ver docs/DECISOES.md#carga-literal.
```

Nenhum bloco acima de 4 linhas fora do `DECISOES.md`.

## 7. Entregáveis

1. `limpeza/` — o pacote reestruturado
2. `main.py` reduzido a CLI
3. `tests/` — suíte pytest
4. `docs/DECISOES.md` — 12–15 entradas: carga literal, portão AST, orçamento vs.
   gabarito, gates de 100%, veredito contra o fallback, ponto cego da corrupção
   universal, KMeans sobre valores distintos
5. `CLAUDE.md` — o que é o projeto, a arquitetura, as políticas acima, e o
   invariante de aceitação da seção 2. Escrito na **última** fase, para
   descrever a estrutura que existe e não a pretendida
6. `README.md` e `CHANGELOG.md` atualizados

### Formato do `DECISOES.md`

```markdown
## Carga literal do CSV
**Decisão:** `keep_default_na=False, na_values=[]` na leitura dos dois CSVs.
**Por quê:** o pandas converte "N/A" em NaN por default. Em `ibu` do beers isso
funde a sentinela do dirty com o vazio do clean e apaga 1.005 erros (42% da
coluna) antes de qualquer algoritmo rodar.
**Onde:** `limpeza/dados.py::carregar`
```

## 8. Fora de escopo

- Melhorar o recall. Este trabalho preserva comportamento; não persegue número.
- Atacar o ponto cego do `ounces` (detecção F1=0 na corrupção universal).
- Trocar de modelo ou de provedor.
- Detecção cross-column (hoje a regra é `detectar(col) -> Series`, intra-coluna).
- Publicar pacote, versionar release, empacotar para distribuição.

## 9. Riscos

| Risco | Mitigação |
|---|---|
| O refactor muda os números do artefato | A seção 2 é teste de regressão; roda a cada fase |
| Remover um `except` derruba coluna que hoje sobrevive | Os que definem comportamento estão listados na seção 6 e ficam |
| Perder conhecimento ao podar comentário | `DECISOES.md` é escrito **antes** da poda, não depois |
| `pandas 3.0.5` diverge do 2.x das execuções antigas | Já verificado: o `budget` de 30/08 reproduziu jul/2026 dígito a dígito |
| Renomear `poc/` → `limpeza/` perde histórico | `git mv` preserva; nenhum arquivo é recriado do zero |
