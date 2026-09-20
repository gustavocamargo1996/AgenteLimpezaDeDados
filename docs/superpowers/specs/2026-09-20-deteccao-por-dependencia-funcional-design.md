# Detecção por dependência funcional — design

**Data:** 2026-09-20
**Estado:** aguardando revisão do usuário

## 1. O problema, medido

A detecção da POC é **intra-coluna**: o contrato é
`detectar(col) -> pd.Series[bool]`, e a regra só enxerga o próprio valor.
Erros que só existem em relação ao resto da linha passam batido.

A lacuna é real e mensurável. Um erro é **inalcançável intra-coluna** quando o
mesmo valor sujo aparece às vezes errado e às vezes certo — nesse caso nenhuma
função do valor sozinho consegue separá-los.

| dataset | linhas | erros | inalcançáveis | teto de recall intra-coluna |
|---|---|---|---|---|
| beers | 2.410 | 4.362 | **0** | 100% |
| hospital | 1.000 | 509 | 0 | 100% |
| rayyan | 1.000 | 948 | 0 | 100% |
| flights | 2.376 | 4.920 | 314 (6,4%) | 93,6% |
| **environment** | 1.000 | 1.147 | **247 (21,5%)** | 78,5% |

Duas leituras que orientam tudo o que vem depois.

**O `beers` não serve de campo de prova.** Zero por cento dos erros dele são
inalcançáveis. O dataset sobre o qual o invariante do projeto está construído
é justamente cego para esta lacuna.

**No `environment` a concentração é total:** `State` **165 de 165** e
`Climate_Zone` **82 de 82**. Não é que alguns erros escapem — é que *todo*
erro dessas duas colunas é estruturalmente invisível para qualquer regra
intra-coluna.

## 2. A forma do erro, e a assimetria que ela revela

Os erros inalcançáveis têm uma forma só:

```
linha 1:  State='MH' -> deveria ser 'KA'   | City='Bangalore', Country='India'
linha 9:  State='BE' -> deveria ser 'HH'   | City='Hamburg',   Country='Germany'
linha 5:  Climate_Zone='Tropical' -> 'Temperate'  | City='Berlin'
```

São **violações de dependência funcional**. O valor sujo é um código válido do
mesmo domínio — só que o do lugar errado.

O mesmo vale para o `flights`, onde várias fontes (`helloflight`, `boston`,
`weather`) reportam o mesmo voo: `flight -> act_dep_time` vale sem exceção no
gabarito, em 100 grupos.

**Nos cinco datasets, todo erro cross-column é violação de FD.**

Daí a assimetria que este trabalho existe para corrigir:

> O projeto **já sabe usar outra coluna para consertar** uma célula — é a
> camada 2 da cascata, `limpeza/correcao/fd.py`. Ele nunca usa outra coluna
> para **encontrar** uma. E como a FD só age sobre células que a detecção
> marcou, e a detecção marca zero em `State`, a camada que saberia consertar
> nunca recebe a chance.

### Viabilidade, medida antes de desenhar

Detecção por desvio da moda condicionada, no `environment` completo:

| FD | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| `City -> State` | 1,00 | 1,00 | **1,00** | 165 | 0 | 0 |
| `City -> Climate_Zone` | 1,00 | 1,00 | **1,00** | 82 | 0 | 0 |

Fecha 100% da lacuna medida, com zero falso positivo.

## 3. Escopo: por que NÃO o contrato geral `detectar(df)`

A seção 8 do `CLAUDE.md` antecipa um projeto maior — trocar o contrato de
detecção por uma forma cross-column. **Este trabalho não é aquele**, e a
decisão foi tomada com dado, não por conveniência.

**A allowlist do portão AST não é o obstáculo que parecia.** Verificado por
execução: uma regra que compara duas colunas da mesma linha **já passa hoje**.
O que `_ATRIBUTOS_SERIE` bloqueia é outra coisa:

| Bloqueado | Motivo |
|---|---|
| `to_csv`, `read_csv`, `open` | segurança — código gerado não escreve em disco |
| `groupby`, `transform`, `duplicated`, `value_counts`, `shift` | **científico** — regra que agrega está memorizando o dataset, não descrevendo o defeito |

O que impede cross-column hoje é o **prompt** e a **convenção de chamada**
(`detectar(col)` recebe uma Series), não o gate de segurança.

**E a rota geral não teria o que provar.** Ela pegaria, em tese, consistência
aritmética, ordem temporal, faixa condicionada a outra coluna — nenhuma das
quais existe nos cinco datasets disponíveis. Construir a capacidade agora
seria construir sem dado que a valide.

**O custo é alto, e um item decide.** A rota geral mudaria
`limpeza/empacotar.py::_ESTATICO`, que é copiado byte a byte para o limpador
gerado e termina a fixture congelada do invariante. Mudar o contrato de
detecção **obriga a regerar a fixture e o invariante** — ou seja, pede que se
abra mão da rede de segurança justamente enquanto se faz a mudança mais
arriscada do projeto.

A rota geral fica documentada como opção futura, para quando aparecer um
dataset com erro cross-column que não seja FD.

## 4. Onde a etapa entra

```
1. por coluna:  amostragem → detecção intra-coluna → refino
2.              construir_mascara(trabalhos)                  →  mascara_intra
3.  NOVO        detectar_dependencia(..., mascara_intra)      →  mascara_fd
4.              mascara = mascara_intra | mascara_fd
5.              rodar_cascata(trabalhos, tabela, mascara, agentes)
```

### Por que não é circular

`fd._moda_condicionada` usa **duplo filtro**: só entram na moda as linhas em
que *as duas* colunas — determinante e dependente — estão com `mascara == 0`.

A etapa nova recebe `mascara_intra`, a máscara **anterior**, e não a que ela
própria produz. No `environment`, `mascara_intra` é toda zero em `State`, e a
moda é calculada sobre todas as linhas de cada cidade — foi assim que o
F1=1,00 da seção 2 foi medido.

### Um efeito colateral que é bom

A cascata recebe a máscara **combinada**. Quando ela corrigir, o duplo filtro
vai excluir as células que a FD-detecção acabou de marcar: **a moda que
corrige é mais limpa que a moda que detectou**. Não é sorte — é o duplo filtro
fazendo o que foi desenhado para fazer, com informação que antes não existia.

### O ponto cego, herdado

A moda só acha o erro enquanto o valor errado for **minoria** no grupo. Com
corrupção majoritária, a moda *seria* o valor errado. É o mesmo ponto cego da
corrupção universal já documentado para `ounces`
(`docs/DECISOES.md#ponto-cego-da-corrupcao-universal`), e a mitigação é a
mesma: o **gate de 100% no orçamento rotulado**, que rejeita a FD antes que
ela marque qualquer coisa.

## 5. O contrato do módulo

`limpeza/deteccao/dependencia.py` — é etapa 3, e a árvore agrupa por etapa.

```python
def detectar_dependencia(trabalhos, tabela, mascara_intra, agente) -> pd.DataFrame:
    """Propoe uma FD por coluna e marca quem desvia da moda condicionada."""
```

Por coluna, na ordem:

1. `fd.candidatos_determinantes(df, coluna, limiar=config.MI_THRESHOLD)` — sem
   candidato, pula **sem gastar chamada**;
2. `fd.propor_fd(...)` — o agente escolhe o determinante e **justifica**;
3. `fd.validar_fd(fd, rotulados, df, mascara_intra, coluna)` — gate de 100%;
4. reprovou → **não marca nada** nessa coluna, e registra o motivo;
5. passou → marca as linhas cujo valor difere da moda condicionada.

**Marca só a coluna dependente.** Uma violação de `City -> State` torna a
célula de `State` suspeita, não a de `City` — mesma semântica da correção.

**Grupo sem moda não marca ninguém.** `fd._moda_condicionada` devolve `None`
quando o duplo filtro não deixa nenhuma linha no grupo. Nesse caso não há
referência para comparar, e as linhas daquele grupo ficam **sem marca** — a
ausência de evidência não é evidência de erro.

### O acoplamento, declarado

O módulo importa `from ..correcao import fd`: detecção importando de correção.
A alternativa seria duplicar a proposta de FD e o gate, que é pior. A causa é
histórica — `fd.py` guarda duas coisas, a primitiva de informação mútua (que
era `contexto.py` antes do refactor de 2026-08-30) e o agente de FD. Mover a
primitiva para um lugar neutro seria mais limpo, e é churn que não serve a
este objetivo.

### Como a FD chega à cascata

Campo novo em `limpeza/tipos.py::Trabalho`:

```python
dependencia: DependenciaFuncional | None = None
```

**O campo só é preenchido quando a FD PASSOU no gate.** Uma FD reprovada não
pode ser guardada: a cascata a reaproveitaria como se fosse válida, e o gate
de 100% — que é o que impede uma regra ruim de agir — teria sido contornado
pela porta dos fundos. Reprovou, o campo fica `None` e a cascata segue o
caminho de hoje.

E a camada 2 da cascata prefere o que já existe:

```python
fd = trabalho.dependencia            # proposta e validada na deteccao
if fd is None and pendentes and candidatos:
    fd = fd_mod.propor_fd(...)       # caminho de hoje, para quem nao tem
```

**A mesma FD que marcou é a que corrige.** Sem isso, detecção e correção
poderiam propor FDs diferentes e discordar — incoerência observável no
relatório.

### O custo em chamadas de LLM

Sobe, modestamente. Hoje a cascata propõe FD **só quando há células
pendentes**; a detecção vai propor para toda coluna com candidato de MI:

| dataset | colunas | com candidato de MI |
|---|---|---|
| beers | 10 | 9 |
| environment | 10 | 4 |
| flights | 6 | 5 |

Contra as ~100 chamadas de um run, é algo entre **5% e 10% a mais**.

### Falha e degradação

Segue a política da seção 7 do `CLAUDE.md`: exceção numa coluna não derruba o
run. A coluna não ganha marca de FD, entra no log com tipo e mensagem, e as
demais seguem. Não é fronteira nova de resiliência — é o mesmo padrão de
`aplicar_detectores`.

### O relatório

A justificativa do agente entra no `cadeias_deteccao.md` das colunas que hoje
saem vazias ali. Quando o gate reprova, o motivo também: "a FD foi proposta e
rejeitada" é informação diferente de "não havia candidato", e as duas hoje se
pareceriam com silêncio.

## 6. Testes e critério de aceitação

### Por que o critério tem dois níveis

O invariante do `beers` funciona porque mede um limpador **já gerado**. Aqui
não dá para copiar o truque: a FD é **escolhida pelo LLM**, que pode propor
`City -> State` hoje e `Monitoring_Station_ID -> State` amanhã — as duas
válidas.

**Nível 1 — o gate determinístico, sem API.** Com uma FD *fixada pelo teste*
(`City -> State`), o detector marca exatamente **54 células** em `State` e
**26** em `Climate_Zone` nas fixtures de 300 linhas, com zero falso positivo.
Agente dublê, FD pronta; o que se testa é a mecânica. **É este o gate que
trava a implementação.**

**Nível 2 — validação com LLM real, uma vez.** Rodar o `environment` de ponta
a ponta com OpenAI e relatar que `State` e `Climate_Zone` saem de recall 0
para alto. Número relatado, não travado — como os runs do `beers` foram.

### As fixtures novas

`tests/fixtures/environment_dirty_300.csv` e `environment_clean_300.csv`,
**94 KB os dois**, copiadas dos arquivos `_300` oficiais do benchmark. Mesma
convenção das fixtures do `beers`: versionadas, independentes do clone
externo.

Verificado nelas: a FD `City -> State` vale sem exceção no gabarito, e a moda
condicionada dá **54/54** e **26/26** com zero falso positivo.

### O teste mais importante é o mais barato

> **A FD-detecção sobre o `beers` tem de marcar ZERO.**

O `beers` não tem erro cross-column. Se a máscara dele ganhar uma marca
sequer, a FD-detecção vazou — e o invariante
`495 / 121 / 121 / 100% / 24,4% / 39,3% / 71` vai denunciar.

### A lista

| Teste | O que trava |
|---|---|
| marca 54 e 26 com FD fixa | a mecânica da moda condicionada |
| gate reprova → não marca nada | FD ruim não suja a máscara |
| `beers` ganha zero marca de FD | não vazou para dataset sem cross-column |
| a FD proposta chega ao `Trabalho` | a cascata reaproveita em vez de propor de novo |
| coluna sem candidato de MI é pulada | não gasta chamada à toa |
| exceção numa coluna não derruba as outras | a degradação por coluna |
| a justificativa sai no `cadeias_deteccao.md` | a cadeia de pensamento, que é produto |

Todos com dublê. **Nenhum gasta API.**

## 7. Tetos, verificáveis a cada tarefa

- densidade de comentário ≤ **13%** (`tests/medir_verbosidade.py`; hoje 12,8%)
- nenhum arquivo novo acima de **150 linhas**
- a suíte sai de 114 para no máximo **~130** testes
- a costura cross-column continua **10 pontos em 7 arquivos** — zero ponto novo
- o invariante do `beers` e `tests/test_estatico_congelado.py` intactos

## 8. Fora de escopo

- **O contrato geral `detectar(df)`** — seção 3.
- **Iteração da máscara** (marcar, recalcular a moda, remarcar até estabilizar).
  Uma passada já dá 54/54 e 26/26; iterar seria complexidade sem ganho
  demonstrável. Se aparecer dataset com corrupção majoritária num grupo,
  mede-se primeiro.
- **Melhorar a correção.** Este trabalho faz a FD *encontrar*; consertar ela já
  sabia.
- **FD com determinante composto** (duas colunas determinando uma terceira).
  `fd.propor_fd` propõe um determinante único hoje, e os dados não pedem mais.

## 9. Riscos

| Risco | Mitigação |
|---|---|
| A FD-detecção marca célula boa no `beers` | Teste dedicado exigindo zero marca, mais o invariante |
| O LLM propõe uma FD ruim | Gate de 100% no orçamento rotulado, que já existe e é testado |
| Corrupção majoritária num grupo inverte a moda | O gate rejeita antes de marcar; risco documentado, não eliminado |
| Detecção e correção usarem FDs diferentes | A FD viaja no `Trabalho`; a cascata reaproveita em vez de propor |
| Custo de API sobe sem aviso | Medido e declarado: 5–10% a mais, e a spec diz por quê |
| Verbosidade volta | Tetos da seção 7, verificados a cada tarefa |
