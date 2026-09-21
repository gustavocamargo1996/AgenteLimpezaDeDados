# A detecção por FD viaja para o limpador gerado

**Data:** 2026-09-21
**Branch:** `fd-no-limpador`, criado a partir de `deteccao-por-fd` (ainda não mergeado)
**Antecessor:** `docs/superpowers/specs/2026-09-20-deteccao-por-dependencia-funcional-design.md`

---

## 1. O problema

O branch `deteccao-por-fd` acrescentou uma segunda via de detecção: a
dependência funcional, que vê a linha e não só o valor. Ela roda **só na POC**.
O limpador gerado monta a máscara em `_ESTATICO::_mascara` iterando apenas
`_DETECTORES`, que são os `detectar(col)` intra-coluna. O dicionário `FDS`
empacotado só **corrige**, e só age sobre células que alguém marcou.

Consequência, medida pela revisão final do branch anterior no `environment`,
coluna `State`:

| | Run da POC | Limpador entregue |
|---|---|---|
| detecção | 51/51, F1 = 1,00 | **0 marcadas** |
| correção | 51 acertos, 0 dano | **0 alteradas** |

O CLAUDE.md §1 diz que o produto é o **programa**. Hoje o run publica um número
que o programa entregue não reproduz. Este projeto fecha essa distância.

Há um segundo defeito com a mesma raiz (Important 5 da revisão final). A
cascata da POC calcula a moda da correção com a máscara **combinada**
(intra + FD), e o limpador com a máscara **intra**. No caso construído
`City=[Bangalore, Bangalore]`, `State=[KA, AA]`, com `AA` marcado só pela FD, a
POC corrige para `KA` e o limpador manteria `AA`.

## 2. Decisões tomadas com o usuário

| Decisão | Escolha | Por quê |
|---|---|---|
| Com qual FD o limpador detecta | **dois dicionários**: `DEPENDENCIAS` detecta, `FDS` corrige | é o que a POC faz; com um só, o limpador detectaria com FDs que a cascata criou só para corrigir, que o run nunca usou para marcar |
| Como impedir que POC e limpador divirjam | **espelho + teste de equivalência** | a lógica existe duas vezes por construção (o limpador é autônomo); um teste compara as duas máscaras célula por célula |
| Escopo do run pago do `environment` | **todas as colunas** das fixtures de 300 linhas | fixture realista, com a detecção intra e a FD convivendo no dataset inteiro |
| Se o LLM não aprovar FD para `State`/`Climate_Zone` | **congelar assim mesmo** e registrar | rodar de novo até sair resultado bonito seria escolher o número publicado |

Rejeitado: fazer a POC executar a função de `_ESTATICO` (fonte única). A POC
usa máscara `int` e o limpador `bool`, e a POC passaria a rodar código de
string fora do sandbox.

## 3. O desenho

### 3.1 O limpador gerado

```python
# metade de cima -- gerada por empacotar.gerar_limpador, FORA de _ESTATICO
_DETECTORES  = {...}                          # inalterado
_CORRETORES  = {...}                          # inalterado
DEPENDENCIAS = {'State': ('City', 'State')}   # NOVO
FDS          = {...}                          # inalterado: so' corrige

# _ESTATICO
def _mascara(df):
    # m_intra: exatamente o laco de hoje sobre _DETECTORES
    # m = m_intra.copy()
    # para cada col, (det, dep) em DEPENDENCIAS:
    #     m[col] |= _desvios_da_moda(df, m_intra, det, dep)
    # return m

def _desvios_da_moda(df, mascara, det, dep):   # NOVO: espelho de deteccao/dependencia.py
def _aplicar_fd(...)                          # inalterado: ja recebe m
def aplicar(df)                               # inalterado
```

### 3.2 Três propriedades

1. **Anticircularidade, igual à POC.** Toda FD lê `m_intra`, nunca a máscara
   que está sendo acumulada. É a mesma regra de `pipeline.gerar_limpador`, que
   passa `mascara_intra` a `detectar_dependencia`.
2. **O Important 5 se resolve sem código a mais.** `aplicar` já passa `m` a
   `_aplicar_fd`. Com `m` virando a máscara combinada, a moda da correção no
   limpador usa a mesma máscara que `cascata.rodar_coluna` usa
   (`mascara_completa`).
3. **`DEPENDENCIAS` só leva FD aprovada no gate da detecção.** Ela sai de
   `trabalho.dependencia`, que `deteccao/dependencia.py::_marcar_coluna` só
   preenche depois do gate e depois de calcular a marca. Coluna sem
   `dependencia` fica fora de `DEPENDENCIAS`, mesmo que tenha entrada em `FDS`.

### 3.3 Onde cada coisa muda

| Arquivo | Mudança |
|---|---|
| `limpeza/empacotar.py::_ESTATICO` | `_mascara` combina intra + FD; `_desvios_da_moda` novo |
| `limpeza/empacotar.py::gerar_limpador` | coleta `trabalho.dependencia` e emite `DEPENDENCIAS` |
| `limpeza/empacotar.py::_montar_cabecalho` | **sai** o aviso "LIMITACAO CONHECIDA" |
| `tests/fixtures/limpador_beers_congelado.py` | `DEPENDENCIAS = {}` em cima; `_ESTATICO` novo embaixo |
| `tests/fixtures/limpador_environment_congelado.py` | **novo**, saído do run pago |
| `CLAUDE.md`, `README.md`, `docs/DECISOES.md`, `CHANGELOG.md` | sai a limitação; entram as duas decisões de invariante |

A costura cross-column (CLAUDE.md §8) continua em **10 pontos, 7 arquivos**.
`empacotar.py` já é o ponto 10, e a FD não conhece `detectar(col)`.

A mesma moda existe hoje em três lugares: `fd.moda_condicionada` na POC,
`_aplicar_fd` no limpador e, depois deste projeto, `_desvios_da_moda` no
limpador. Todos desempatam do mesmo jeito, com `valores.mode().iloc[0]`, que
devolve o menor valor em ordem. `_desvios_da_moda` **tem de** usar a mesma
expressão.

## 4. Testes

Do mais barato ao mais caro. Nenhum dos quatro primeiros gasta API.

1. **Equivalência POC × limpador.** Mesma tabela (`environment_300`), mesmas
   FDs injetadas por dublê de agente e mesmos detectores intra. De um lado, a
   máscara combinada do pipeline (`construir_mascara` + `detectar_dependencia`,
   combinadas como em `pipeline.gerar_limpador`). Do outro, a `_mascara` de um
   limpador **gerado de verdade** por `empacotar.gerar_limpador` e carregado do
   disco. As duas têm de ser idênticas célula por célula, em quatro cenários:
   - máscara intra vazia;
   - máscara intra realista, em que o duplo filtro muda o resultado;
   - o empate `Bangalore` com `KA`/`AA`;
   - **FDs encadeadas**, em que a dependente de uma é a determinante de outra
     (`City → State` e `State → Country`). É o único cenário em que ler `m` em
     vez de `m_intra` muda o resultado: as marcas de `State` feitas pela
     primeira FD passariam a tirar linhas da moda da segunda. Sem ele, a
     mutação de circularidade abaixo não morde.
2. **Ponta a ponta com FD injetada.** Um limpador gerado com
   `DEPENDENCIAS = {'State': ('City', 'State'), 'Climate_Zone': ('City', 'Climate_Zone')}`,
   aplicado ao `environment_dirty_300.csv`. O lado da POC já trava 54 marcas em
   `State` e 26 em `Climate_Zone` (`tests/test_dependencia.py`). O teste trava
   quanto o **limpador** marca e corrige, com os números medidos na
   implementação. Hoje a resposta é zero, e é esse o teste que prova que o
   problema da seção 1 acabou.
3. **Invariante do `beers` inalterado.** 495 / 121 / 121 / 1,0 / 0,2444 /
   0,3929 / 71. O run de 20/ago é anterior à detecção por FD e não tem nenhuma
   FD de detecção, então `DEPENDENCIAS = {}` e o maquinário novo não pode mudar
   nada. Números inalterados são a prova de que ele não interfere quando não há
   FD de detecção. As duas FDs de `FDS` (`city`, `state`) continuam só
   corrigindo.
4. **`test_estatico_congelado` sobre as duas fixtures.** Ambas têm de terminar
   com o `_ESTATICO` atual, byte a byte.
5. **Invariante do `environment`.** O limpador do run pago, congelado, aplicado
   às 300 linhas. Os números são **medidos, não previstos**, e ficam travados
   com uma entrada em `DECISOES.md`.

Cada teste novo precisa de prova por mutação no relatório da tarefa. Os
alvos: para o 1, voltar `_mascara` a ignorar `DEPENDENCIAS`, e trocar `m_intra`
por `m` dentro do laço (circularidade); para o 2, esvaziar `DEPENDENCIAS` em
`gerar_limpador`.

## 5. Ordem

1. Branch `fd-no-limpador` a partir de `deteccao-por-fd`.
2. `_desvios_da_moda` e a `_mascara` nova em `_ESTATICO`, guiadas pelo teste 1.
3. `gerar_limpador` passa a emitir `DEPENDENCIAS`; teste 2.
4. Fixture do `beers` com a metade de baixo nova e `DEPENDENCIAS = {}`;
   testes 3 e 4.
5. Sai o aviso "LIMITACAO CONHECIDA" do código e dos documentos.
6. **Run pago**, rodado pelo controlador, nunca por subagente, e anunciado ao
   usuário antes do disparo:
   ```bash
   .venv/Scripts/python main.py \
     --sujo tests/fixtures/environment_dirty_300.csv \
     --limpo tests/fixtures/environment_clean_300.csv
   ```
   Tem de vir depois do passo 5, porque o limpador gerado pelo run precisa já
   sair com `DEPENDENCIAS`.
7. Congelar o limpador do run em `tests/fixtures/limpador_environment_congelado.py`;
   teste 5; documentação.

## 6. Critério de aceitação

- Os testes 1 a 4 passam, e cada um morde sob a mutação indicada.
- O invariante do `beers` não se move.
- O teste 2 mostra o limpador marcando e corrigindo células de `State` que hoje
  ele não toca.
- A costura continua em 10 pontos / 7 arquivos; densidade ≤ 14%.
- Nenhum documento afirma mais que o código faz. Em particular, some toda
  menção à FD como "não empacotada".

### Sobre o run pago: o que se relata, o que se trava

Depois do run, comparar a métrica que o **run publicou**
(`deteccao_metricas.json`, `correcao_metricas.json`) com a que o **limpador
congelado** obtém via `avaliar_limpador.py`. Essa comparação é o teste real de
que o limpador reproduz o run. Os números **não precisam coincidir**: o run
mede fora das ~10 linhas rotuladas (o holdout, CLAUDE.md §4) e o
`avaliar_limpador` mede a tabela inteira. A diferença, se houver, é
**documentada e explicada**, não travada como igualdade.

Se o LLM não aprovar FD para `State` nem para `Climate_Zone`, a fixture é
congelada assim mesmo e `DECISOES.md` registra isso. Nesse caso o invariante do
`environment` não prova a detecção nova, e a prova fica com os testes 1 e 2, que
existem em qualquer cenário justamente por isso.

## 7. Fora de escopo

- Qualquer mudança na detecção da POC: a lógica já existe e está testada.
- Regerar a fixture do `beers` com um run novo. Isso misturaria dois efeitos
  (LLM diferente e maquinário diferente) e ninguém conseguiria separá-los.
- O contrato geral `detectar(df)`, já descartado em
  `docs/DECISOES.md#deteccao-por-fd`.
- Push e merge: decisão do usuário, como sempre.

## 8. Riscos

| Risco | Mitigação |
|---|---|
| O desempate da moda diverge entre as cópias | seção 3.3 fixa `mode().iloc[0]`; o cenário `KA`/`AA` do teste 1 existe para pegar isso |
| O limpador passa a marcar célula correta que a POC também marca (FP da FD, como `Blackrocks Brewery` no `beers`) | é fidelidade, não bug: o limpador reproduz o run, inclusive os erros dele; o gate de 100% continua sendo o freio |
| O run pago não aprova nenhuma FD | decisão da seção 2: congelar e registrar; testes 1 e 2 cobrem |
| `mascara` `bool` no limpador × `int` na POC | o teste 1 compara depois de converter a da POC com `astype(bool)`; o que importa é a mesma célula marcada |
