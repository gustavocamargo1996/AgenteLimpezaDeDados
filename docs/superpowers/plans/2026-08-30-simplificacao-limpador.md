# Simplificação do Gerador de Limpadores — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduzir a POC a um produto só — dataset sujo + amostra limpa produzem um `limpador.py` autônomo, a cadeia de pensamento que o gerou e o F1 de reparo — com a ordem das etapas legível na estrutura dos módulos.

**Architecture:** Um `limpeza/pipeline.py` linear é a espinha: seis chamadas em ordem, cada uma delegando a um módulo nomeado pela responsabilidade. Oito dicionários paralelos indexados por coluna viram uma lista de um dataclass `Trabalho`. O modo `blind` e a camada 3 da cascata saem; o loop de refinamento da detecção vira default.

**Tech Stack:** Python 3.12, pandas 3.0.5, scikit-learn, onnxruntime + tokenizers (MiniLM local), langchain-openai (gpt-4o-mini), pydantic 2, pytest (a adicionar), uv.

**Spec:** `docs/superpowers/specs/2026-08-30-simplificacao-limpador-design.md`

## Global Constraints

- **Invariante de aceitação (bloqueante em toda tarefa):** `beers` sufixo 300 produz `erros 495 | mudou 121 | TP 121 | precisão 100,0% | recall 24,4% | F1 39,3% | flags 71`. Qualquer desvio é bug, não simplificação.
- **Idioma:** todo código, comentário, docstring e documento em **português**, sem acento em identificadores. Mensagens ao usuário em pt-BR.
- **Comentários:** docstring de módulo e de função com **1 linha**; comentário inline com teto de **2 linhas**; nenhum bloco acima de 4 linhas fora de `docs/DECISOES.md`.
- **Proibido em comentário:** histórico de decisão, medições de sessões passadas, comparações com o ZeroDC (`correction.py:790` e afins), referências a planos/invariantes de sessões de IA, argumentação sobre alternativas descartadas.
- **Costura cross-column:** os 7 pontos listados na seção 9 da spec continuam sendo os únicos que sabem que a detecção é intra-coluna. Nenhuma etapa nova pode assumir isso.
- **Preservar histórico:** movimentação de arquivo sempre com `git mv`, nunca criar-e-deletar.
- **Métricas alvo ao final:** 2.100–2.400 linhas; ~9% de comentário; ~7 blocos `except`; ~35 loops; nenhum arquivo acima de 300 linhas.

---

## File Structure

**Criados:**

| Arquivo | Responsabilidade |
|---|---|
| `tests/fixtures/limpador_beers_congelado.py` | O limpador de 20/ago, congelado, base do invariante |
| `tests/fixtures/beers_dirty_300.csv` | Entrada suja do invariante (32 KB) |
| `tests/fixtures/beers_clean_300.csv` | Gabarito do invariante (29 KB) |
| `tests/test_invariante.py` | Trava o F1 de reparo publicado |
| `tests/test_sandbox.py` | Portão AST: escalar e modo série |
| `tests/test_mascara.py` | Construção da máscara, por coluna, posicional |
| `tests/test_fd.py` | Gate e aplicação da dependência funcional |
| `tests/test_metricas.py` | P/R/F1, incluindo o caso degenerado |
| `tests/test_refino.py` | Loop com oráculo: amostragem, feedback, update |
| `tests/conftest.py` | Fixtures compartilhadas |
| `limpeza/pipeline.py` | A espinha: as etapas em ordem |
| `limpeza/tipos.py` | Os dataclasses que viajam entre etapas |
| `limpeza/deteccao/__init__.py` | Só re-exporta |
| `limpeza/deteccao/regra.py` | Agente detector, 1 passe |
| `limpeza/deteccao/refino.py` | O loop: consome a resposta do oráculo e atualiza a regra |
| `limpeza/deteccao/oraculo.py` | Escolhe o que perguntar ao oráculo (amostragem e scorer) |
| `limpeza/deteccao/mascara.py` | Aplica os detectores à tabela |
| `limpeza/correcao/__init__.py` | Só re-exporta |
| `limpeza/correcao/cascata.py` | Camada 1 → camada 2 → flag |
| `limpeza/correcao/regras.py` | Agente 1 (spec) + agente 2 (código) |
| `limpeza/correcao/fd.py` | Dependência funcional + MI |
| `docs/DECISOES.md` | O porquê medido, fora do código |
| `CLAUDE.md` | Orientação para sessões futuras |

**Movidos (`git mv`):** `poc/` → `limpeza/`; `poc/embeddings.py` funde em `limpeza/amostragem.py`; `poc/avaliacao.py` → `limpeza/metricas.py`; `poc/contexto.py` funde em `limpeza/correcao/fd.py`.

**Deletados:** `poc/fallback_celula.py`, `verificar_e2e.py`, `verificar_ambiente.py`.

---

### Task 1: Rede de segurança — o invariante virar teste automático

Hoje o invariante só existe como um comando manual cuja entrada mora em `runs/` (gitignored) e no `ZERODC_DIR` (externo, já quebrou nesta máquina). Antes de tocar em qualquer código, ele precisa virar um teste que roda offline e não depende de nada fora do repositório.

**Files:**
- Create: `tests/fixtures/limpador_beers_congelado.py`, `tests/fixtures/beers_dirty_300.csv`, `tests/fixtures/beers_clean_300.csv`
- Create: `tests/test_invariante.py`, `tests/conftest.py`, `pytest.ini`
- Modify: `avaliar_limpador.py` (extrair função pura de `main`)
- Modify: `requirements.txt`

- [ ] **Step 1: Copiar as fixtures para dentro do repositório**

```bash
mkdir -p tests/fixtures
cp runs/2026-08-20_0927__e2e/limpador_beers_2026-08-20_0927.py tests/fixtures/limpador_beers_congelado.py
cp "../ZeroDC/datasets/beers/beers_dirty_300.csv" tests/fixtures/
cp "../ZeroDC/datasets/beers/beers_clean_300.csv" tests/fixtures/
```

- [ ] **Step 2: Adicionar pytest às dependências e instalar**

Acrescentar ao final de `requirements.txt`:

```
pytest>=8.0
```

Rodar: `uv pip install -r requirements.txt`

Criar `pytest.ini`:

```ini
[pytest]
testpaths = tests
filterwarnings =
    ignore::SyntaxWarning
    ignore::UserWarning
```

- [ ] **Step 3: Escrever o teste que falha**

Criar `tests/test_invariante.py`:

```python
"""Trava os numeros publicados do limpador de beers."""
from pathlib import Path

from avaliar_limpador import avaliar

FIXTURES = Path(__file__).parent / "fixtures"


def test_f1_reparo_do_limpador_congelado():
    total = avaliar(
        caminho_limpador=FIXTURES / "limpador_beers_congelado.py",
        caminho_sujo=FIXTURES / "beers_dirty_300.csv",
        caminho_limpo=FIXTURES / "beers_clean_300.csv",
    )["total"]
    assert total["erros"] == 495
    assert total["mudancas"] == 121
    assert total["tp"] == 121
    assert total["precisao"] == 1.0
    assert total["recall"] == 0.2444
    assert total["f1"] == 0.3929
    assert total["flags"] == 71
```

- [ ] **Step 4: Rodar o teste e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_invariante.py -v`
Expected: FAIL com `ImportError: cannot import name 'avaliar'`

- [ ] **Step 5: Extrair a função pura de `main()`**

Em `avaliar_limpador.py`, acrescentar antes de `main()`:

```python
def avaliar(caminho_limpador, caminho_sujo, caminho_limpo) -> dict:
    """Aplica o limpador ao sujo e mede o F1 de reparo por coluna e no total."""
    sujo = pd.read_csv(caminho_sujo, **LER)
    limpo = pd.read_csv(caminho_limpo, **LER)
    corrigido, flags = carregar_limpador(str(caminho_limpador)).aplicar(sujo)

    colunas, tp = {}, 0
    n_erro = n_mud = n_flag = 0
    for col in sujo.columns:
        if col not in corrigido.columns:
            continue
        m = f1_reparo(sujo[col], corrigido[col], limpo[col])
        m["flags"] = int(flags[col].sum()) if col in flags.columns else 0
        n_flag += m["flags"]
        colunas[col] = m
        if m["mensuravel"]:
            tp += m["tp"]
            n_erro += m["erros"]
            n_mud += m["mudancas"]

    p = tp / n_mud if n_mud else None
    r = tp / n_erro if n_erro else 0.0
    f1 = 2 * p * r / (p + r) if (p and (p + r)) else 0.0
    return {
        "colunas": colunas,
        "total": {
            "erros": n_erro, "mudancas": n_mud, "tp": tp, "flags": n_flag,
            "precisao": round(p, 4) if p is not None else None,
            "recall": round(r, 4), "f1": round(f1, 4),
        },
    }
```

Reescrever `main()` para consumir `avaliar()` em vez de recalcular: ele resolve os caminhos a partir de `--dataset`/`--sufixo`, chama `avaliar()`, e só imprime o dicionário devolvido. Nenhuma aritmética fica em `main()`.

- [ ] **Step 6: Rodar o teste e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_invariante.py -v`
Expected: PASS

- [ ] **Step 7: Confirmar que a CLI antiga ainda dá o mesmo resultado**

Run: `.venv/Scripts/python avaliar_limpador.py --limpador tests/fixtures/limpador_beers_congelado.py --dataset beers --sufixo 300`
Expected: a linha `TOTAL` mostra `495 | 121 | 121 | 100.0% | 24.4% | 39.3% | 71`

- [ ] **Step 8: Commit**

```bash
git add tests/ pytest.ini requirements.txt avaliar_limpador.py
git commit -m "Rede de seguranca: o invariante publicado vira teste pytest"
```

---

### Task 2: `docs/DECISOES.md` — escrever o porquê ANTES de podar

A spec lista isto como mitigação de risco: o documento precisa existir **antes** da poda de comentários, senão o conhecimento se perde entre uma coisa e outra.

**Files:**
- Create: `docs/DECISOES.md`

**Interfaces:**
- Produces: âncoras de link (`#carga-literal`, `#portao-ast`, ...) que as Tasks 9 e 12 citam a partir do código.

- [ ] **Step 1: Varrer o código e extrair o raciocínio**

Run: `git grep -n "DESVIO\|DECLARAD\|medido em\|invariante\|plano secao\|auditoria\|ZeroDC" -- "*.py" > /tmp/raciocinio.txt`

Cada acerto é candidato a entrada. Consolidar em 12–15 entradas — várias ocorrências da mesma ideia viram uma entrada só.

- [ ] **Step 2: Escrever o documento**

Uma entrada por decisão, no formato fixo:

```markdown
## Carga literal do CSV
**Decisão:** `keep_default_na=False, na_values=[]` na leitura dos dois CSVs.
**Por quê:** o pandas converte "N/A" em NaN por default. Em `ibu` do beers isso
funde a sentinela do dirty com o vazio do clean e apaga 1.005 erros (42% da
coluna) antes de qualquer algoritmo rodar.
**Onde:** `limpeza/dados.py::carregar`
```

As entradas obrigatórias, com o número que sustenta cada uma:

1. **Carga literal do CSV** — 1.005 erros de `ibu` apagados sem `keep_default_na=False`
2. **KMeans sobre valores distintos** — 2.410 células de `ounces` colapsam em 25 valores
3. **Representantes: o mais típico e o mais atípico** — o atípico expõe a corrupção, o típico ancora a norma
4. **Portão AST antes de executar código gerado** — o original faz `exec(code, globals())` sem validação
5. **Modo série do portão (`series_mode`)** — allowlist de atributos bloqueia I/O e cross-row
6. **Orçamento de rotulagem ≠ gabarito** — o limpo entra em 3 papéis; só o de medição vê a tabela toda
7. **Gates de 100% nos rótulos** — nenhuma regra que erre no rotulado é aplicada; é o que mantém o dano em ~0%
8. **Flag em vez de chute** — célula que nenhuma camada resolve fica sinalizada
9. **Veredito contra o fallback por célula** — 7 runs: resolve ~100% com acerto ~0–0,5%, milhares de chamadas, única camada sem gate
10. **Loop de refinamento com oráculo** — orçamento de ~10 valores por coluna; é o que produz os números publicados
11. **Ponto cego da corrupção universal** — em `ounces` 100% das células estão erradas, a forma corrompida É a norma, detecção F1=0
12. **Detecção intra-coluna** — `detectar(col) -> Series`; os 7 pontos da costura e por que ela é o próximo projeto
13. **Escalonamento célula a célula na cascata** — célula que a camada deixou intacta escala para a próxima
14. **`random_state=0` na amostragem** — determinismo permite reavaliar sem gastar API

- [ ] **Step 3: Verificar que nenhuma entrada ficou sem número**

Run: `grep -c "Por quê" docs/DECISOES.md`
Expected: 14

Ler o arquivo e confirmar que cada `**Por quê:**` cita um número, um arquivo ou uma medição concreta. Entrada que só diz "é melhor assim" não é entrada — ou ganha evidência ou sai.

- [ ] **Step 4: Commit**

```bash
git add docs/DECISOES.md
git commit -m "docs: registra as decisoes medidas antes da poda de comentarios"
```

---

### Task 3: Cortar a camada 3 (fallback por célula)

**Files:**
- Delete: `poc/fallback_celula.py`
- Modify: `poc/cascata.py` (bloco da camada 3, ~linhas 126-147), `poc/esquemas.py` (`CorrecaoCelula`), `poc/config.py` (`USAR_FALLBACK`, `LIMITE_FALLBACK`), `main.py` (import, agente, argumento `--limite-fallback`), `poc/relatorio.py` (coluna Fallback), `verificar_e2e.py`

- [ ] **Step 1: Rodar o invariante antes de mexer**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 2: Remover a camada 3 da cascata**

Em `poc/cascata.py`, apagar o bloco `if config.USAR_FALLBACK and pendentes:` inteiro e a variável `fallback_chamadas`. Remover `fallback_celula` do import da linha 21 e o parâmetro `limite_fallback` da assinatura de `rodar_cascata`.

Em `trilha`, remover a chave `"fallback_chamadas"`. A chave `contagem["fallback"]` sai do dicionário inicial e de todo consumidor.

- [ ] **Step 3: Remover os consumidores**

- `poc/esquemas.py`: apagar a classe `CorrecaoCelula`
- `poc/config.py`: apagar `USAR_FALLBACK` e `LIMITE_FALLBACK` com seus comentários
- `main.py`: remover `fallback_celula` do import, a entrada `"fallback"` no dict de agentes (linha ~213), o argumento `--limite-fallback` (linha ~433) e os dois usos de `args.limite_fallback`
- `poc/relatorio.py`: remover a coluna `Fallback` da tabela da cascata e a linha "camada 3 (fallback): N chamada(s)"
- `verificar_e2e.py`: remover os testes que exercitam o fallback
- `git rm poc/fallback_celula.py`

- [ ] **Step 4: Verificar que nada referencia o que saiu**

Run: `git grep -n "fallback\|CorrecaoCelula\|USAR_FALLBACK" -- "*.py"`
Expected: nenhuma saída

- [ ] **Step 5: Rodar o invariante**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: PASS — o fallback não participava dos números, então nada muda

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Corta a camada 3 da cascata (fallback por celula)"
```

---

### Task 4: Cortar o modo `blind` e a comparação

**Files:**
- Modify: `main.py` (`_preparar_itens`, `_avaliar`, `processar_coluna`, `carregar_codigos`, `reavaliar`, `imprimir`, `main`), `poc/identificador_regras.py` (`HUMANO_BLIND`, ramo do modo), `poc/config.py` (`MODOS`), `poc/avaliacao.py` (`avaliar`, `Medida`, `Resultado`, `_medir`, `_sem_holdout`), `poc/relatorio.py` (`escrever`, `comparativo`, `_bloco_cadeia`, `_linha_medida`), `poc/dados.py` (`particionar`)

- [ ] **Step 1: Remover o parâmetro `modo` da geração de regras**

Em `poc/identificador_regras.py`: apagar `HUMANO_BLIND`, e em `especificar()` remover o parâmetro `modo` — o prompt passa a ser sempre `HUMANO_BUDGET`. Em `_formatar_amostra()`, apagar o ramo `if modo != "budget"` e o parâmetro `modo`.

- [ ] **Step 2: Remover o caminho antigo do `main.py`**

Apagar as funções `_avaliar`, `processar_coluna`, `carregar_codigos`, `reavaliar` e `imprimir`. Em `_preparar_itens`, remover o parâmetro `modo` e o `if modo == "budget"` (o corpo passa a rodar sempre). Em `main()`, remover os argumentos `--modo`, `--reavaliar` e `--e2e`, e o despacho entre os caminhos: sobra apenas o antigo `rodar_e2e`.

- [ ] **Step 3: Remover as métricas e relatórios do modo antigo**

- `poc/avaliacao.py`: apagar `avaliar`, `Medida`, `Resultado`, `_medir`, `_sem_holdout`. Ficam `metricas_deteccao` e `metricas_correcao`
- `poc/relatorio.py`: apagar `escrever`, `comparativo`, `_bloco_cadeia`, `_linha_medida`
- `poc/dados.py`: apagar `particionar`
- `poc/config.py`: apagar `MODOS`

- [ ] **Step 4: Verificar que nada referencia o que saiu**

Run: `git grep -nw "blind\|MODOS\|comparativo\|particionar\|reavaliar" -- "*.py"`
Expected: nenhuma saída

- [ ] **Step 5: Rodar o invariante**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: Medir o progresso**

Run: `.venv/Scripts/python -c "import glob;print(sum(len(open(f,encoding='utf-8').read().splitlines()) for f in ['main.py','avaliar_limpador.py']+glob.glob('poc/*.py')))"`
Expected: bem abaixo de 4.969 (referência: ~3.400)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Corta o modo blind e a comparacao entre modos"
```

---

### Task 5: Promover o loop de refinamento a default

**Files:**
- Modify: `poc/config.py` (`ITERACOES_DETECCAO`), `main.py` (o `if args.iteracoes_deteccao > 1`)

**Interfaces:**
- Produces: `deteccao.refinar()` passa a ser sempre chamada; a Task 7 conta com isso na espinha.

- [ ] **Step 1: Trocar o default**

Em `poc/config.py`: `ITERACOES_DETECCAO = 5`, com comentário de uma linha: `# 5 iteracoes x 2 amostras = orcamento de ~10 rotulos por coluna.`

- [ ] **Step 2: Remover a bifurcação**

Em `main.py`, o bloco `if args.iteracoes_deteccao > 1 and ...` deixa de ser condicional pelo número de iterações. A guarda que **fica** é a de sentinela: uma coluna cuja regra é `DETECTA_NADA` não tem o que refinar.

```python
if saida_det["regra"].codigo != deteccao.DETECTA_NADA:
    saida_ref = deteccao.refinar_regra_deteccao(...)
```

- [ ] **Step 3: Rodar o invariante**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: PASS — o invariante mede um limpador congelado, que já nasceu com o loop ligado

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Promove o loop de refinamento da deteccao a comportamento default"
```

---

### Task 6: Reestruturar `poc/` em `limpeza/`

Movimentação pura, sem mudança de comportamento. Tudo com `git mv` para preservar histórico.

**Files:**
- Move: todo o pacote

- [ ] **Step 1: Renomear o pacote e criar os subpacotes**

```bash
git mv poc limpeza
mkdir limpeza/deteccao limpeza/correcao
git mv limpeza/avaliacao.py limpeza/metricas.py
git mv limpeza/cascata.py limpeza/correcao/cascata.py
git mv limpeza/fd.py limpeza/correcao/fd.py
git mv limpeza/identificador_regras.py limpeza/correcao/regras.py
git mv limpeza/deteccao.py limpeza/deteccao/regra.py
```

- [ ] **Step 2: Fundir os módulos de consumidor único**

- `limpeza/embeddings.py` → colar o conteúdo no topo de `limpeza/amostragem.py`, depois `git rm limpeza/embeddings.py`
- `limpeza/contexto.py` → colar `calc_mi` e `candidatos_determinantes` em `limpeza/correcao/fd.py`, depois `git rm limpeza/contexto.py`
- `limpeza/gerador_codigo.py` → colar em `limpeza/correcao/regras.py` (agente 2 depois do agente 1, na ordem em que rodam), depois `git rm limpeza/gerador_codigo.py`

- [ ] **Step 3: Quebrar `deteccao/regra.py` em quatro**

O arquivo tem 776 linhas, das quais 545 são código puro. Separar em quatro, e **não** em três: o bloco do refino sozinho tem 562 linhas e violaria a meta de nenhum arquivo acima de 300.

- `limpeza/deteccao/regra.py` — `construir_agente`, `_formatar_amostra`, `gerar_regra_deteccao`, `DETECTA_NADA`, `SISTEMA`, `HUMANO` (linhas 1–161 de hoje)
- `limpeza/deteccao/oraculo.py` — **o que perguntar**: `_classificar`, `_selecionar_diverso`, `_contar_chars_alnum`, `_score_artefato`, `_selecionar_suspeito`, `_amostrar_oraculo` (linhas 248–477)
- `limpeza/deteccao/refino.py` — **o loop**: `_montar_feedback_det`, `_precisao_no_oraculo`, `_tentar_update`, `refinar_regra_deteccao` (linhas 478–724)
- `limpeza/deteccao/mascara.py` — `construir_mascara` (linha 725 em diante)

O cabeçalho de comentário das linhas 162–247 (a introdução do bloco de refino) é raciocínio, não código: vai para `docs/DECISOES.md#loop-de-refinamento` na Task 2, não é copiado.

- [ ] **Step 3b: Criar `detector_nulo()`**

A espinha da Task 7 precisa de uma forma de dizer "esta coluna não é marcada" sem duplicar a construção do sentinela, que hoje aparece em dois lugares. Em `limpeza/deteccao/regra.py`:

```python
def detector_nulo(motivo: str = "") -> Detector:
    """Detector que nao marca celula nenhuma, para coluna que falhou."""
    return Detector(
        codigo=DETECTA_NADA,
        funcao=sandbox.materializar(DETECTA_NADA, nome_funcao="detectar",
                                    nome_argumento="col", series_mode=True),
        cadeia=motivo or "coluna nao marcada",
    )
```

- [ ] **Step 4: Escrever os `__init__.py` que só re-exportam**

`limpeza/deteccao/__init__.py`:

```python
"""Etapa 3: onde estao os erros."""
from .mascara import construir_mascara
from .refino import refinar_regra_deteccao
from .regra import (DETECTA_NADA, construir_agente, detector_nulo,
                    gerar_regra_deteccao)

__all__ = ["DETECTA_NADA", "construir_agente", "detector_nulo",
           "gerar_regra_deteccao", "refinar_regra_deteccao", "construir_mascara"]
```

`limpeza/correcao/__init__.py`:

```python
"""Etapa 4: como consertar."""
from .cascata import rodar_cascata

__all__ = ["rodar_cascata"]
```

- [ ] **Step 5: Corrigir todos os imports**

Run: `git grep -ln "from poc\|import poc\|from \. import" -- "*.py"`

Em cada arquivo, trocar `poc` por `limpeza` e ajustar os relativos que ficaram um nível mais fundo (dentro de `deteccao/` e `correcao/`, `from . import config` vira `from .. import config`).

- [ ] **Step 6: Verificar que importa**

Run: `.venv/Scripts/python -c "import limpeza.pipeline" 2>&1 | head -3` — ainda vai falhar (pipeline só existe na Task 7)
Run: `.venv/Scripts/python -c "from limpeza import deteccao, correcao, dados, metricas; print('ok')"`
Expected: `ok`

- [ ] **Step 7: Rodar o invariante**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 8: Confirmar que o histórico sobreviveu**

Run: `git log --follow --oneline limpeza/deteccao/regra.py | head -3`
Expected: mostra commits anteriores ao rename

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "Reestrutura poc/ em limpeza/ com subpacotes deteccao e correcao"
```

---

### Task 7: Os tipos e a espinha

**Files:**
- Create: `limpeza/tipos.py`, `limpeza/pipeline.py`
- Modify: `main.py` (reduzir a CLI)

**Interfaces:**

- Consumes (assinaturas **novas** — esta tarefa as adapta, e a Task 6 já moveu os módulos):

```python
dados.carregar(caminho_sujo, caminho_limpo, colunas=None) -> Tabela
amostragem.representantes(coluna: Coluna) -> Amostra
deteccao.gerar_regra_deteccao(coluna: Coluna, amostra: Amostra, agente) -> Detector
deteccao.refinar_regra_deteccao(detector: Detector, coluna: Coluna,
                                amostra: Amostra, agente) -> Detector
deteccao.detector_nulo(motivo: str = "") -> Detector
deteccao.construir_mascara(trabalhos: list[Trabalho], tabela: Tabela) -> pd.DataFrame
correcao.rodar_cascata(trabalhos, tabela, mascara, agentes) -> None  # preenche .correcao
empacotar.gerar_limpador(trabalhos: list[Trabalho], saida: Path) -> Path
metricas.avaliar(trabalhos, tabela, mascara) -> dict
```

As três primeiras trocam listas de parâmetros soltos (`representantes_sujos`, `total_linhas`, `total_distintos`, `contagem`) pelo objeto `Amostra`, que já carrega tudo. `metricas.avaliar` embrulha `metricas_deteccao` e `metricas_correcao`, que continuam existindo e testáveis isoladamente.

- Produces: `pipeline.gerar_limpador(caminho_sujo, caminho_limpo, colunas, saida) -> tuple[Path, dict]`; os dataclasses de `limpeza/tipos.py`

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_pipeline.py`:

```python
"""A espinha monta um Trabalho por coluna, na ordem das etapas."""
import pandas as pd

from limpeza.tipos import Coluna, Trabalho


def _coluna(nome="abv"):
    serie = pd.Series(["0.05", "0.06%"])
    return Coluna(nome=nome, sujo=serie, limpo=serie,
                  valores_distintos=["0.05", "0.06%"], contagem={"0.05": 1, "0.06%": 1})


def test_trabalho_nasce_sem_correcao_nem_medida():
    t = Trabalho(coluna=_coluna(), amostra=None, detector=None)
    assert t.correcao is None
    assert t.medida is None
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'limpeza.tipos'`

- [ ] **Step 3: Escrever `limpeza/tipos.py`**

```python
"""Os dados que viajam entre as etapas do pipeline."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Coluna:
    """Uma coluna da tabela, suja e (quando ha) a referencia limpa."""
    nome: str
    sujo: Any
    limpo: Any
    valores_distintos: list[str]
    contagem: dict


@dataclass
class Tabela:
    """Os dois CSVs alinhados por linha e as colunas a processar."""
    sujo: Any
    limpo: Any
    colunas: list[Coluna]


@dataclass
class Amostra:
    """O que o KMeans escolheu mostrar ao agente, com os rotulos do orcamento."""
    representantes: list[str]
    linhas: set[int]
    rotulados: list[dict]
    total_linhas: int
    total_distintos: int


@dataclass
class Detector:
    """Como achar o erro numa coluna."""
    codigo: str
    funcao: Any
    cadeia: str
    orcamento: dict = field(default_factory=dict)
    historico: list = field(default_factory=list)


@dataclass
class Correcao:
    """Como consertar as celulas marcadas de uma coluna."""
    passos: list[dict]
    trilha: dict
    cadeia: str


@dataclass
class Trabalho:
    """Tudo que se sabe sobre uma coluna, preenchido ao longo das etapas."""
    coluna: Coluna
    amostra: Amostra | None
    detector: Detector | None
    correcao: Correcao | None = None
    medida: dict | None = None
```

`Detector` guarda `codigo` e `funcao` sem nunca inspecionar a assinatura — é o que mantém a costura cross-column concentrada nos 7 pontos da seção 9 da spec.

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Escrever `limpeza/pipeline.py`**

Portar o corpo de `rodar_e2e` (218 linhas em `main.py`) para a espinha. Os oito dicionários paralelos — `funcoes_deteccao`, `regras_deteccao`, `orcamento_deteccao`, `historico_deteccao`, `contexto_por_coluna`, `deteccao_metricas`, `correcao_metricas`, `trilhas` — viram a lista `trabalhos`.

```python
"""A espinha: as etapas do gerador, em ordem."""


def gerar_limpador(caminho_sujo, caminho_limpo, colunas, saida):
    """Gera um limpador.py a partir do sujo e do orcamento de rotulos."""
    tabela = dados.carregar(caminho_sujo, caminho_limpo, colunas)
    agentes = _construir_agentes()

    trabalhos = []
    for coluna in tabela.colunas:
        trabalhos.append(_processar_coluna(coluna, tabela, agentes))

    mascara = deteccao.construir_mascara(trabalhos, tabela)
    correcao.rodar_cascata(trabalhos, tabela, mascara, agentes)
    limpador = empacotar.gerar_limpador(trabalhos, saida)
    medida = metricas.avaliar(trabalhos, tabela, mascara)
    relatorio.escrever(saida, trabalhos, medida)
    return limpador, medida
```

`_processar_coluna` concentra a **única** fronteira de resiliência (etapas 2 e 3 de uma coluna):

```python
def _processar_coluna(coluna, tabela, agentes) -> Trabalho:
    """Amostra, detecta e refina uma coluna. Falha vira DETECTA_NADA."""
    amostra = amostragem.representantes(coluna)
    try:
        detector = deteccao.gerar_regra_deteccao(coluna, amostra, agentes["deteccao"])
        if detector.codigo != deteccao.DETECTA_NADA:
            detector = deteccao.refinar_regra_deteccao(detector, coluna, amostra,
                                                       agentes["deteccao"])
    except Exception as exc:
        print(f"  {coluna.nome}: deteccao falhou ({type(exc).__name__}) -> nao marcada")
        detector = deteccao.detector_nulo()
    return Trabalho(coluna=coluna, amostra=amostra, detector=detector)
```

- [ ] **Step 6: Reduzir `main.py` a CLI**

Sobra: `argparse`, a chamada a `pipeline.gerar_limpador`, e a impressão do resultado. Nenhuma lógica de orquestração.

- [ ] **Step 7: Rodar o invariante e a suíte**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 8: Verificar que os dicts paralelos sumiram**

Run: `git grep -nc "funcoes_deteccao\|contexto_por_coluna\|historico_deteccao" -- "*.py"`
Expected: nenhuma saída

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "Introduz os tipos nomeados e a espinha linear do pipeline"
```

---

### Task 8: Entrada por caminho de arquivo

**Files:**
- Modify: `main.py`, `limpeza/config.py`, `limpeza/dados.py`, `avaliar_limpador.py`

- [ ] **Step 1: Escrever o teste que falha**

Acrescentar a `tests/test_pipeline.py`:

```python
from pathlib import Path
from limpeza import dados

FIXTURES = Path(__file__).parent / "fixtures"


def test_carregar_aceita_caminhos_diretos():
    tabela = dados.carregar(FIXTURES / "beers_dirty_300.csv",
                            FIXTURES / "beers_clean_300.csv")
    assert len(tabela.sujo) == 300
    assert list(tabela.sujo.columns) == list(tabela.limpo.columns)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py::test_carregar_aceita_caminhos_diretos -v`
Expected: FAIL

- [ ] **Step 3: Tornar os caminhos obrigatórios e explícitos**

Em `limpeza/config.py`: apagar `ZERODC_DIR`, `DIR_DATASET`, `CSV_SUJO`, `CSV_LIMPO`, `DIR_MODELO`, `DATASET`, `caminhos_dataset` e `COLUNAS_PADRAO`. Fica só o caminho do modelo MiniLM, que continua vindo de variável de ambiente por não ser dado do usuário:

```python
MODELO_EMBEDDING = Path(os.getenv("MODELO_EMBEDDING", "./all-MiniLM-L6-v2"))
```

Em `limpeza/dados.py`, `carregar` deixa de ter default: `def carregar(caminho_sujo, caminho_limpo, colunas=None)`.

Em `main.py`, os argumentos passam a ser:

```python
ap.add_argument("--sujo", required=True, help="CSV com os dados sujos")
ap.add_argument("--limpo", required=True, help="CSV de referencia (orcamento de rotulos)")
ap.add_argument("--colunas", default="todas")
ap.add_argument("--saida", default="runs")
```

Em `avaliar_limpador.py`, `main()` passa a receber `--sujo`/`--limpo` e a função `caminhos()` é apagada.

- [ ] **Step 4: Atualizar o `.env.example`**

Remover `DATASET` e `ZERODC_DIR`; acrescentar `MODELO_EMBEDDING` com uma linha explicando que aponta para a pasta do MiniLM.

- [ ] **Step 5: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: Confirmar que o acoplamento sumiu**

Run: `git grep -n "ZERODC\|zerodc" -- "*.py" "*.example"`
Expected: nenhuma saída

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Entrada por caminho de arquivo; remove o acoplamento ao ZERODC_DIR"
```

---

### Task 9: Poda de comentários

**Files:**
- Modify: todos os `.py` de `limpeza/`, `main.py`, `avaliar_limpador.py`

- [ ] **Step 1: Criar o medidor**

Criar `tests/medir_verbosidade.py`:

```python
"""Mede densidade de comentario por arquivo. Nao e' teste; e' instrumento."""
import ast
import glob
import io
import tokenize


def medir(caminho: str) -> dict:
    src = open(caminho, encoding="utf-8").read()
    total = len(src.splitlines())
    doc = 0
    for no in ast.walk(ast.parse(src)):
        if isinstance(no, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            d = ast.get_docstring(no, clean=False)
            if d:
                doc += len(d.splitlines()) + 2
    com = sum(1 for t in tokenize.generate_tokens(io.StringIO(src).readline)
              if t.type == tokenize.COMMENT)
    return {"arquivo": caminho, "total": total, "doc": doc, "comentario": com,
            "densidade": (doc + com) / total if total else 0}


if __name__ == "__main__":
    alvos = ["main.py", "avaliar_limpador.py"] + glob.glob("limpeza/**/*.py", recursive=True)
    linhas = [medir(a) for a in alvos]
    for m in sorted(linhas, key=lambda x: -x["densidade"]):
        print(f"{m['arquivo']:40s}{m['total']:6d}{m['densidade']:8.1%}")
    t = sum(m["total"] for m in linhas)
    d = sum(m["doc"] + m["comentario"] for m in linhas)
    print(f"\nTOTAL {t} linhas, {d/t:.1%} de comentario")
```

- [ ] **Step 2: Medir o ponto de partida**

Run: `.venv/Scripts/python tests/medir_verbosidade.py`

Anotar o total. Trabalhar dos arquivos de maior densidade para os de menor.

- [ ] **Step 3: Podar arquivo a arquivo, aplicando as regras**

Para cada arquivo, na ordem de densidade decrescente:

1. Docstring de módulo → **1 linha** dizendo qual etapa do pipeline ele é
2. Docstring de função → **1 linha** dizendo o que a função faz
3. Comentário inline → só onde o código engana; teto de 2 linhas
4. Onde o porquê protege o código → 1 linha com o número + `Ver docs/DECISOES.md#ancora`

Exemplo obrigatório, `limpeza/dados.py::carregar` — 12 linhas de docstring para 7 de código:

```python
# ANTES
"""Le os dois CSVs como texto literal, byte a byte.

DESVIO DELIBERADO DO ZERODC. O original faz `pd.read_csv(dtype=str).fillna('null')`,
e o pandas converte "N/A" em NaN por default -- entao o "N/A" do dirty e o vazio
do clean viram o MESMO token e a diferenca some. Na coluna `ibu` do beers isso
apaga 1005 erros reais (42% da coluna) antes de qualquer algoritmo rodar.

`keep_default_na=False` mantem "N/A", "NA", "null", "-" como o texto que sao.
Sentinela de ausencia vira uma coisa que o agente PODE detectar, em vez de um
artefato de carga que ninguem ve.
"""

# DEPOIS
"""Le os CSVs como texto literal e confere que estao alinhados por linha."""
# keep_default_na=False: sem isso o pandas converte "N/A" em NaN e apaga
# 1.005 erros reais de `ibu`. Ver docs/DECISOES.md#carga-literal.
```

- [ ] **Step 4: Verificar que o proibido saiu**

Run: `git grep -niE "DESVIO|DECLARAD|medido em [0-9]|invariante [0-9]|plano secao|auditoria|correction\.py:|detection\.py:" -- "limpeza/*.py" "limpeza/**/*.py" "main.py"`
Expected: nenhuma saída

- [ ] **Step 5: Verificar a meta**

Run: `.venv/Scripts/python tests/medir_verbosidade.py | tail -2`
Expected: densidade total ≤ 10%

- [ ] **Step 6: Rodar o invariante**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: PASS — comentário não muda comportamento; se falhar, uma edição escorregou para o código

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Poda os comentarios: mecanica no codigo, porque em DECISOES.md"
```

---

### Task 10: Exceções e loops

**Files:**
- Modify: `limpeza/correcao/cascata.py`, `limpeza/deteccao/*.py`, `limpeza/metricas.py`, `limpeza/empacotar.py`

- [ ] **Step 1: Inventariar**

Run: `git grep -c "except" -- "limpeza/**/*.py" "main.py"`
Run: `git grep -cE "^\s*(for|while) " -- "limpeza/**/*.py" "main.py"`

- [ ] **Step 2: Remover os `except` que só escondem**

Regra: um `except Exception` sobrevive apenas se responder "sim" a *este erro tem um comportamento de recuperação declarado e diferente de propagar?*.

**Ficam, com comentário de 1 linha explicando por quê:**

- `limpeza/correcao/cascata.py`, o guarda por célula na aplicação do corretor (hoje `cascata.py:94`):

```python
try:
    novo = funcao(antigo)
except Exception:
    novo = antigo  # regex gerado que estoura num valor: celula fica intacta
```

- os três de `limpeza/empacotar.py` e os três de `limpeza/sandbox.py`: são portões de segurança
- o de `limpeza/pipeline.py::_processar_coluna`: a única fronteira de resiliência

**Saem:** todos os demais. Onde um deles hoje engole erro no meio de um módulo, o erro passa a subir até `_processar_coluna`.

- [ ] **Step 3: Colapsar as passagens paralelas restantes**

Procurar laços que percorrem colunas mais de uma vez no mesmo escopo e fundi-los na passagem única de `pipeline.py`. Os laços que **ficam** são os que carregam estado — em especial o escalonamento da cascata, que é a semântica do algoritmo.

- [ ] **Step 4: Verificar as metas**

Run: `git grep -c "except" -- "limpeza/**/*.py" "main.py" | awk -F: '{s+=$2} END {print s" except"}'`
Expected: ≤ 8

Run: `git grep -cE "^\s*(for|while) " -- "limpeza/**/*.py" "main.py" | awk -F: '{s+=$2} END {print s" loops"}'`
Expected: ≤ 40

- [ ] **Step 5: Rodar o invariante**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Concentra a resiliencia numa fronteira e colapsa passagens paralelas"
```

---

### Task 11: A suíte pytest

Converte o que `verificar_e2e.py` (1.118 linhas) e `verificar_ambiente.py` (116) verificam hoje via `print`, para asserts que falham sozinhos.

**Files:**
- Create: `tests/test_sandbox.py`, `tests/test_mascara.py`, `tests/test_fd.py`, `tests/test_metricas.py`, `tests/test_refino.py`, `tests/test_amostragem.py`
- Modify: `tests/conftest.py`
- Delete: `verificar_e2e.py`, `verificar_ambiente.py`

- [ ] **Step 1: Fixtures compartilhadas**

Em `tests/conftest.py`:

```python
"""Fixtures compartilhadas pela suite."""
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
LER = dict(dtype=str, keep_default_na=False, na_values=[])


@pytest.fixture
def beers_sujo():
    return pd.read_csv(FIXTURES / "beers_dirty_300.csv", **LER)


@pytest.fixture
def beers_limpo():
    return pd.read_csv(FIXTURES / "beers_clean_300.csv", **LER)
```

- [ ] **Step 2: Converter os testes do portão AST**

`tests/test_sandbox.py` cobre os 6 casos que o `verificar_ambiente.py` já exercita, mais os de modo série do `verificar_e2e.py`:

```python
import pytest

from limpeza import sandbox
from limpeza.sandbox import CodigoRejeitado


def test_aceita_funcao_valida():
    sandbox.validar("def corrigir(valor):\n    return valor.strip()\n")


@pytest.mark.parametrize("codigo,trecho", [
    ("def fix(valor):\n    return valor\n", "deve se chamar"),
    ("def corrigir(v, w):\n    return v\n", "exatamente um argumento"),
    ("import re\ndef corrigir(valor):\n    return valor\n", "exatamente uma definicao"),
    ("def corrigir(valor):\n    return eval(valor)\n", "chamada proibida"),
    ("x = 1\ndef corrigir(valor):\n    return valor\n", "exatamente uma definicao"),
])
def test_rejeita(codigo, trecho):
    with pytest.raises(CodigoRejeitado, match=trecho):
        sandbox.validar(codigo)


def test_modo_serie_rejeita_atributo_fora_da_allowlist():
    with pytest.raises(CodigoRejeitado, match="allowlist"):
        sandbox.validar("def detectar(col):\n    return col.duplicated()\n",
                        nome_funcao="detectar", nome_argumento="col", series_mode=True)


def test_modo_serie_aceita_o_idioma_contains():
    sandbox.validar("def detectar(col):\n    return col.astype(str).str.contains('N/A')\n",
                    nome_funcao="detectar", nome_argumento="col", series_mode=True)
```

- [ ] **Step 3: Converter as duas invariantes sutis**

Estas duas guardam propriedades que um refactor quebra sem avisar, então vão com código explícito.

`tests/test_metricas.py` — o caso degenerado devolve "não mensurável", nunca `0.0`:

```python
import pandas as pd

from limpeza import metricas


def test_coluna_sem_erro_real_e_nao_mensuravel():
    serie = pd.Series(["a", "b", "c"])
    m = metricas.metricas_deteccao(pd.Series([0, 0, 0]), serie, serie, pd.Index([]))
    assert m["mensuravel"] is False
    assert m["f1"] is None, "0.0 mentiria: nao ha erro para achar"


def test_deteccao_perfeita_da_f1_1():
    sujo = pd.Series(["N/A", "42", "55"])
    limpo = pd.Series(["", "42", "55"])
    m = metricas.metricas_deteccao(pd.Series([1, 0, 0]), sujo, limpo, pd.Index([]))
    assert m["precisao"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
```

`tests/test_mascara.py` — a máscara não pode ler o gabarito:

```python
import pandas as pd

from limpeza import sandbox
from limpeza.deteccao import construir_mascara


def _detector(codigo):
    return sandbox.materializar(codigo, nome_funcao="detectar",
                                nome_argumento="col", series_mode=True)


def test_marca_so_o_marcador():
    df = pd.DataFrame({"ibu": ["N/A", "42", "N/A"]})
    funcao = _detector("def detectar(col):\n    return col == 'N/A'\n")
    mascara = construir_mascara({"ibu": funcao}, df)
    assert list(mascara["ibu"]) == [1, 0, 1]


def test_funcao_que_devolve_escalar_zera_a_coluna_sem_derrubar_o_run():
    df = pd.DataFrame({"ibu": ["N/A", "42"]})
    funcao = _detector("def detectar(col):\n    return True\n")
    mascara = construir_mascara({"ibu": funcao}, df)
    assert list(mascara["ibu"]) == [0, 0]
```

- [ ] **Step 3b: Converter os demais**

Mesma forma — o que hoje é `print("[ok] ...")` vira `assert`. Para cada um, ler o teste correspondente em `verificar_e2e.py` e transportar os mesmos casos:

- `test_amostragem.py` — mesma entrada dá os mesmos representantes (`random_state=0`); a seleção traz o mais típico e o mais atípico de cada grupo
- `test_fd.py` — `validar_fd` reprova com 1 erro no conjunto rotulado e passa sem nenhum
- `test_refino.py` — `_amostrar_oraculo` é determinístico e respeita `usados`; `_montar_feedback_det` marca FP/FN e é cumulativo; `refinar_regra_deteccao` com agente dublê: sucesso troca a função, rejeição mantém a anterior

- [ ] **Step 4: Apagar os scripts antigos**

```bash
git rm verificar_e2e.py verificar_ambiente.py
```

- [ ] **Step 5: Rodar a suíte inteira**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: todos passam, incluindo o invariante

- [ ] **Step 6: Confirmar que nenhum teste chama a API**

Run: `git grep -n "ChatOpenAI\|OPENAI_API_KEY" -- "tests/*.py"`
Expected: nenhuma saída — os testes de agente usam duble, não rede

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Converte a verificacao offline em suite pytest"
```

---

### Task 12: `CLAUDE.md`, README e CHANGELOG

Escrito por último, de propósito: um arquivo de orientação que descreve arquitetura que ainda não existe é pior que nenhum, porque a próxima sessão confia nele.

**Files:**
- Create: `CLAUDE.md`
- Modify: `README.md`, `CHANGELOG.md`

- [ ] **Step 1: Medir o resultado final**

Run: `.venv/Scripts/python tests/medir_verbosidade.py`
Run: `git grep -c "except" -- "limpeza/**/*.py" "main.py" | awk -F: '{s+=$2} END {print s}'`

Anotar os números reais — eles entram no `CLAUDE.md` e no `CHANGELOG.md`.

- [ ] **Step 2: Escrever o `CLAUDE.md`**

Seções, nesta ordem:

1. **O que é** — entra sujo + amostra limpa; sai `limpador.py` + cadeia + F1. O produto é o programa de limpeza, não a tabela limpa.
2. **Como rodar** — o comando com `--sujo`/`--limpo`, e o aviso de que `main.py` gasta API.
3. **A arquitetura** — a árvore de `limpeza/` com a etapa de cada módulo, e o corpo de `pipeline.gerar_limpador` colado inteiro. Quem ler isso sabe a ordem.
4. **Os tipos** — `Amostra`, `Detector`, `Correcao`, `Trabalho`; e o aviso de que `Detector` **não interpreta a assinatura** da função de detecção.
5. **Invariante de aceitação** — os números do `beers` e o comando que os verifica. Frase explícita: *alterar esses números é bug, não melhoria*.
6. **Políticas** — comentários (1 linha, teto de 2, nada de histórico/medição/ZeroDC/plano de IA), exceções (uma fronteira em `_processar_coluna`; a lista das que ficam e por quê), loops (colapsar paralelas, manter as que carregam estado).
7. **A costura cross-column** — a tabela dos 7 pontos da seção 9 da spec, com a nota de que é o próximo projeto.
8. **Onde não mexer sem ler** — `sandbox.py` (portão de segurança), `empacotar.py` (cópia congelada da semântica da cascata; divergir dele faz o limpador mentir sobre o run).
9. **Idioma** — português em código, comentário, documento e resposta.

- [ ] **Step 3: Atualizar o README**

Reescrever a seção "Rodando" para a nova CLI. Apagar as seções que descrevem o modo `blind`, a comparação e as métricas de acerto/dano — elas documentam código que não existe mais. Manter as seções de segurança e de métricas de reparo. Acrescentar ponteiro para `CLAUDE.md` e `docs/DECISOES.md`.

- [ ] **Step 4: Entrada no CHANGELOG**

Sob `## [Não publicado]`, com `### Removido` e `### Alterado`, citando os números medidos no Step 1 (linhas antes/depois, densidade antes/depois).

- [ ] **Step 5: Verificação final completa**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: tudo passa

Run: `.venv/Scripts/python tests/medir_verbosidade.py | tail -2`
Expected: total entre 2.100 e 2.400 linhas, densidade ≤ 10%

Run: `.venv/Scripts/python -c "import glob;print(max((len(open(f,encoding='utf-8').read().splitlines()),f) for f in glob.glob('limpeza/**/*.py',recursive=True)))"`
Expected: nenhum arquivo acima de 300 linhas

- [ ] **Step 6: Verificar o README contra a realidade**

Rodar cada comando que o README apresenta e confirmar que funciona como descrito. README que documenta comando quebrado é pior que README ausente.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "CLAUDE.md, README e CHANGELOG para a estrutura simplificada"
```

---

## Nota sobre o custo de API

Nenhuma tarefa deste plano exige rodar `main.py`, que gasta chamadas de LLM. O invariante mede um limpador **congelado** e os testes usam dublês. Uma execução end-to-end real é recomendável ao final, para confirmar que o caminho de geração ainda funciona — mas é decisão do usuário, não passo do plano, porque custa dinheiro.
