# Detecção por dependência funcional — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fazer a dependência funcional **encontrar** erros, não só consertá-los, fechando a lacuna cross-column sem tocar no contrato `detectar(col)`.

**Architecture:** Um módulo novo, `limpeza/deteccao/dependencia.py`, roda entre a máscara intra-coluna e a cascata. Por coluna com candidato de informação mútua, ele propõe uma FD pelo agente que já existe, valida com o gate de 100% que já existe, e marca quem desvia da moda condicionada. A FD aprovada viaja no `Trabalho` até a cascata, que a reaproveita em vez de propor outra.

**Tech Stack:** Python 3.12, pandas, pydantic 2, langchain-core. Sem dependência nova.

**Spec:** `docs/superpowers/specs/2026-09-20-deteccao-por-dependencia-funcional-design.md`

## Global Constraints

- **INVARIANTE BLOQUEANTE:** `tests/test_invariante.py` trava `erros=495, mudancas=121, tp=121, precisao=1.0, recall=0.2444, f1=0.3929, flags=71`, e `tests/test_estatico_congelado.py` trava `_ESTATICO`. Os dois continuam exatos. A suíte tem **114 testes** hoje.
- **O `beers` tem de ganhar ZERO marca de FD.** Ele não tem erro cross-column; uma marca sequer significa que a detecção vazou.
- **Idioma:** PORTUGUÊS em tudo. Sem acento em identificadores nem em strings de log; documentos usam acentuação normal.
- **Comentários:** docstring de módulo e de função com **1 linha**; inline com teto de **2 linhas**.
- **Tetos verificáveis:** densidade ≤ **14%** (`tests/medir_verbosidade.py`; hoje 12,8%); nenhum arquivo novo acima de **150 linhas**; suíte no máximo **~130** testes.
- **A costura cross-column continua 10 pontos em 7 arquivos** (seção 8 do `CLAUDE.md`). Este trabalho NÃO acrescenta ponto: `detectar(col)` fica intacto, o portão AST não ganha modo novo.
- **NÃO tocar** em `limpeza/sandbox.py` nem em `limpeza/empacotar.py::_ESTATICO`.
- **Marca só a coluna dependente.** Violação de `City -> State` torna a célula de `State` suspeita, não a de `City`.
- **FD reprovada no gate NÃO é guardada** no `Trabalho`: a cascata a reaproveitaria como válida e o gate teria sido contornado.

---

## File Structure

**Criados:**

| Arquivo | Responsabilidade |
|---|---|
| `limpeza/deteccao/dependencia.py` | propõe a FD, valida, marca o desvio da moda |
| `tests/test_dependencia.py` | a mecânica da marcação, com agente dublê |
| `tests/fixtures/environment_dirty_300.csv` | 300 linhas com 54 erros de `State` e 26 de `Climate_Zone` |
| `tests/fixtures/environment_clean_300.csv` | o gabarito correspondente |

**Modificados:**

| Arquivo | O quê |
|---|---|
| `limpeza/correcao/fd.py` | `_moda_condicionada` vira pública (dois consumidores agora) |
| `limpeza/tipos.py` | campo `dependencia` no `Trabalho` |
| `limpeza/deteccao/__init__.py` | re-exporta `detectar_dependencia` |
| `limpeza/pipeline.py` | chama a etapa nova e combina as máscaras |
| `limpeza/correcao/cascata.py` | prefere a FD já proposta |
| `limpeza/relatorio.py` | publica a justificativa; corrige a frase "INTRA-COLUNA" |
| `tests/test_pipeline.py` | o teste do `beers` zero |
| `CLAUDE.md`, `docs/DECISOES.md`, `README.md`, `CHANGELOG.md` | documentação |

---

### Task 1: Fixtures e a moda condicionada pública

**Files:**
- Create: `tests/fixtures/environment_dirty_300.csv`, `tests/fixtures/environment_clean_300.csv`, `tests/test_dependencia.py`
- Modify: `limpeza/correcao/fd.py`

**Interfaces:**
- Produces: `fd.moda_condicionada(df, mascara, determinante, dependente, valor_det, excluir=None)` — pública, mesma implementação de antes; devolve `None` quando o duplo filtro esvazia o grupo

- [ ] **Step 1: Copiar as fixtures**

Os arquivos `_300` oficiais do benchmark, 94 KB os dois:

```bash
cp "../ZeroDC/datasets/environment/environment_dirty_300.csv" tests/fixtures/
cp "../ZeroDC/datasets/environment/environment_clean_300.csv" tests/fixtures/
```

- [ ] **Step 2: Escrever o teste que fixa a fixture**

Criar `tests/test_dependencia.py`:

```python
"""A deteccao por dependencia funcional, com agente duble (sem rede)."""
from pathlib import Path

import pandas as pd

FIXTURES = Path(__file__).parent / "fixtures"
LER = dict(dtype=str, keep_default_na=False, na_values=[])


def _environment():
    return (pd.read_csv(FIXTURES / "environment_dirty_300.csv", **LER),
            pd.read_csv(FIXTURES / "environment_clean_300.csv", **LER))


def test_fixture_tem_os_erros_cross_column_esperados():
    """Fixa o fenomeno que este projeto existe para atacar."""
    sujo, limpo = _environment()
    assert len(sujo) == 300
    assert int((sujo["State"] != limpo["State"]).sum()) == 54
    assert int((sujo["Climate_Zone"] != limpo["Climate_Zone"]).sum()) == 26


def test_fixture_nenhum_erro_e_alcancavel_intra_coluna():
    """Todo erro dessas duas colunas usa um valor que tambem aparece correto."""
    sujo, limpo = _environment()
    for col in ("State", "Climate_Zone"):
        err = sujo[col] != limpo[col]
        ambiguo = pd.DataFrame({"v": sujo[col], "e": err}).groupby("v")["e"].transform(
            lambda s: s.any() and (~s).any())
        assert int((err & ~ambiguo).sum()) == 0, f"{col} tem erro alcancavel intra-coluna"
```

- [ ] **Step 3: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_dependencia.py -v`
Expected: PASS, 2 testes. Se `54`/`26` não baterem, a fixture copiada não é a esperada — pare e relate.

- [ ] **Step 4: Promover a moda condicionada**

Em `limpeza/correcao/fd.py`, renomear `_moda_condicionada` para `moda_condicionada` e ajustar as **duas** chamadas internas (em `validar_fd` e em `aplicar_fd`). Nenhuma mudança de corpo.

Run: `git grep -n "_moda_condicionada" -- "*.py"`
Expected: nenhuma saída

- [ ] **Step 5: Rodar a suíte**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: 116 passed (114 + 2), invariante intacto

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/environment_*_300.csv tests/test_dependencia.py limpeza/correcao/fd.py
git commit -m "Fixtures do environment e moda condicionada publica"
```

---

### Task 2: O módulo de detecção por FD

**Files:**
- Create: `limpeza/deteccao/dependencia.py`
- Modify: `tests/test_dependencia.py`, `limpeza/deteccao/__init__.py`, `limpeza/tipos.py`

**Interfaces:**
- Consumes: `fd.candidatos_determinantes(df, alvo, limiar=None) -> list[str]`; `fd.propor_fd(dependente, rotulados, candidatos_mi, df, agente=None) -> DependenciaFuncional`; `fd.validar_fd(fd, rotulados, df, mascara, dependente) -> bool`; `fd.moda_condicionada(...)` da Task 1
- Produces:

```python
limpeza.deteccao.dependencia.detectar_dependencia(
    trabalhos: list, tabela, mascara_intra: pd.DataFrame, agente
) -> pd.DataFrame          # mascara 0/1, mesmo shape de mascara_intra

limpeza.tipos.Trabalho.dependencia   # DependenciaFuncional aprovada, ou None
```

**O campo `dependencia` nasce nesta tarefa, não na Task 3.** Ele é criado aqui
porque é aqui que passa a ser escrito; deixá-lo para depois faria este módulo
depender de `setattr` num atributo inexistente — funciona em dataclass sem
`slots`, mas é acoplamento invisível. A Task 3 só ensina a cascata a **ler** o
campo.

Em `limpeza/tipos.py`, na classe `Trabalho`, depois de `detector`:

```python
    dependencia: Any = None  # DependenciaFuncional aprovada na deteccao, ou None
```

> `Any` e não `DependenciaFuncional` porque `tipos.py` não importa de
> `esquemas.py` hoje, e criar esse import por um campo documental acoplaria os
> dois módulos. O comentário de 1 linha carrega o tipo real.

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar a `tests/test_dependencia.py`. O `rotulados` tem a forma que `amostragem._rotular` produz: `{"indice": int, "sujo": str, "limpo": str, "eh_erro": bool}`.

```python
import pytest

from limpeza.deteccao import dependencia
from limpeza.esquemas import DependenciaFuncional
from limpeza.tipos import Amostra, Coluna, Tabela, Trabalho


class _AgenteFalso:
    """Devolve sempre a mesma FD; nenhuma chamada sai da maquina."""

    def __init__(self, resposta):
        self._resposta = resposta

    def invoke(self, _args):
        if isinstance(self._resposta, Exception):
            raise self._resposta
        return self._resposta


def _tabela_environment(colunas):
    sujo, limpo = _environment()
    cols = []
    for nome in colunas:
        cols.append(Coluna(nome=nome, sujo=sujo[nome], limpo=limpo[nome],
                           valores_distintos=sorted(sujo[nome].unique()),
                           contagem=sujo[nome].value_counts().to_dict()))
    return Tabela(sujo=sujo, limpo=limpo, nome="environment", colunas=cols)


def _trabalhos(tabela, linhas=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)):
    """Um Trabalho por coluna, com orcamento rotulado nas primeiras linhas."""
    saida = []
    for col in tabela.colunas:
        rot = [{"indice": i, "sujo": col.sujo.at[i], "limpo": col.limpo.at[i],
                "eh_erro": col.sujo.at[i] != col.limpo.at[i]} for i in linhas]
        saida.append(Trabalho(
            coluna=col,
            amostra=Amostra(representantes=[], linhas=set(linhas), rotulados=rot,
                            total_linhas=len(col.sujo), total_distintos=0),
            detector=None))
    return saida


def _mascara_zerada(tabela):
    return pd.DataFrame(0, index=tabela.sujo.index, columns=tabela.sujo.columns, dtype=int)


def test_marca_os_54_erros_de_state(monkeypatch):
    tabela = _tabela_environment(["State"])
    trabalhos = _trabalhos(tabela)
    agente = _AgenteFalso(DependenciaFuncional(
        determinante="City", dependente="State", justificativa="cidade fixa o estado"))
    m = dependencia.detectar_dependencia(trabalhos, tabela, _mascara_zerada(tabela), agente)
    assert int(m["State"].sum()) == 54
    err = tabela.sujo["State"] != tabela.limpo["State"]
    assert int((m["State"].astype(bool) & ~err).sum()) == 0, "marcou celula correta"


def test_marca_os_26_erros_de_climate_zone():
    tabela = _tabela_environment(["Climate_Zone"])
    agente = _AgenteFalso(DependenciaFuncional(
        determinante="City", dependente="Climate_Zone", justificativa="cidade fixa o clima"))
    m = dependencia.detectar_dependencia(_trabalhos(tabela), tabela,
                                         _mascara_zerada(tabela), agente)
    assert int(m["Climate_Zone"].sum()) == 26


def test_fd_reprovada_no_gate_nao_marca_nada():
    """Determinante que nao explica a coluna: o gate barra antes de marcar."""
    tabela = _tabela_environment(["State"])
    agente = _AgenteFalso(DependenciaFuncional(
        determinante="PM2.5", dependente="State", justificativa="ruim de proposito"))
    m = dependencia.detectar_dependencia(_trabalhos(tabela), tabela,
                                         _mascara_zerada(tabela), agente)
    assert int(m["State"].sum()) == 0


def test_fd_reprovada_nao_e_guardada_no_trabalho():
    """Guardar uma FD reprovada faria a cascata contornar o gate."""
    tabela = _tabela_environment(["State"])
    trabalhos = _trabalhos(tabela)
    agente = _AgenteFalso(DependenciaFuncional(
        determinante="PM2.5", dependente="State", justificativa="ruim de proposito"))
    dependencia.detectar_dependencia(trabalhos, tabela, _mascara_zerada(tabela), agente)
    assert getattr(trabalhos[0], "dependencia", None) is None


def test_fd_aprovada_e_guardada_no_trabalho():
    tabela = _tabela_environment(["State"])
    trabalhos = _trabalhos(tabela)
    agente = _AgenteFalso(DependenciaFuncional(
        determinante="City", dependente="State", justificativa="cidade fixa o estado"))
    dependencia.detectar_dependencia(trabalhos, tabela, _mascara_zerada(tabela), agente)
    assert trabalhos[0].dependencia.determinante == "City"


def test_grupo_sem_moda_nao_marca_ninguem():
    """Duplo filtro esvaziou o grupo: sem referencia, nao ha o que acusar."""
    tabela = _tabela_environment(["State"])
    mascara = _mascara_zerada(tabela)
    mascara["City"] = 1  # nenhuma linha passa no duplo filtro
    mascara["State"] = 1
    agente = _AgenteFalso(DependenciaFuncional(
        determinante="City", dependente="State", justificativa="cidade fixa o estado"))
    m = dependencia._desvios_da_moda(tabela.sujo, mascara, "City", "State")
    assert int(m.sum()) == 0, "marcou sem ter moda para comparar"


def test_excecao_numa_coluna_nao_derruba_as_outras():
    tabela = _tabela_environment(["State", "Climate_Zone"])
    trabalhos = _trabalhos(tabela)
    chamadas = {"n": 0}

    class _AgenteQueFalhaUmaVez:
        def invoke(self, _args):
            chamadas["n"] += 1
            if chamadas["n"] == 1:
                raise RuntimeError("boom")
            return DependenciaFuncional(determinante="City", dependente="Climate_Zone",
                                        justificativa="cidade fixa o clima")

    m = dependencia.detectar_dependencia(trabalhos, tabela, _mascara_zerada(tabela),
                                         _AgenteQueFalhaUmaVez())
    assert int(m["State"].sum()) == 0, "a coluna que falhou nao marca"
    assert int(m["Climate_Zone"].sum()) == 26, "a seguinte segue normalmente"
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_dependencia.py -v`
Expected: FAIL com `ImportError: cannot import name 'dependencia'`

- [ ] **Step 3: Escrever o módulo**

Criar `limpeza/deteccao/dependencia.py`:

```python
"""Etapa 3b: marca celulas que violam uma dependencia funcional entre colunas."""
import pandas as pd

from .. import config, erros
from ..correcao import fd as fd_mod


def detectar_dependencia(trabalhos: list, tabela, mascara_intra, agente) -> pd.DataFrame:
    """Propoe uma FD por coluna e marca quem desvia da moda condicionada."""
    df = tabela.sujo
    mascara = pd.DataFrame(0, index=df.index, columns=df.columns, dtype=int)
    for trabalho in trabalhos:
        nome = trabalho.coluna.nome
        candidatos = fd_mod.candidatos_determinantes(df, nome, limiar=config.MI_THRESHOLD)
        if not candidatos:
            continue
        try:
            marca = _marcar_coluna(trabalho, df, mascara_intra, candidatos, agente)
        except Exception as exc:  # noqa: BLE001 -- 1 coluna ruim nao derruba a etapa
            print(f"  {nome}: FD falhou ({erros.descrever(exc)}) -> sem marca", flush=True)
            continue
        if marca is not None:
            mascara[nome] = marca.astype(int)
    return mascara


def _marcar_coluna(trabalho, df, mascara_intra, candidatos, agente):
    """Devolve a marca booleana da coluna, ou None quando o gate reprova."""
    nome = trabalho.coluna.nome
    rotulados = trabalho.amostra.rotulados
    fd = fd_mod.propor_fd(nome, rotulados, candidatos, df, agente=agente)
    if not fd_mod.validar_fd(fd, rotulados, df, mascara_intra, nome):
        print(f"  {nome}: FD `{fd.determinante}` reprovada no gate -> sem marca", flush=True)
        return None
    trabalho.dependencia = fd
    print(f"  {nome}: FD `{fd.determinante}` -> `{nome}` aprovada; marcando desvios...",
          flush=True)
    return _desvios_da_moda(df, mascara_intra, fd.determinante, nome)


def _desvios_da_moda(df, mascara, determinante, dependente) -> pd.Series:
    """True onde o valor difere da moda do grupo; grupo sem moda nao marca ninguem."""
    marca = pd.Series(False, index=df.index)
    for valor_det in df[determinante].unique():
        moda = fd_mod.moda_condicionada(df, mascara, determinante, dependente, valor_det)
        if moda is None:
            continue
        no_grupo = df[determinante] == valor_det
        marca |= no_grupo & (df[dependente] != moda)
    return marca
```

- [ ] **Step 4: Re-exportar no pacote**

Em `limpeza/deteccao/__init__.py`, acrescentar `from .dependencia import detectar_dependencia` e o nome em `__all__`.

- [ ] **Step 5: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_dependencia.py -v`
Expected: PASS, 10 testes

- [ ] **Step 6: Conferir o teto de linhas**

Run: `.venv/Scripts/python -c "print(len(open('limpeza/deteccao/dependencia.py',encoding='utf-8').read().splitlines()))"`
Expected: abaixo de 150

- [ ] **Step 7: Rodar a suíte**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: 124 passed, invariante intacto

- [ ] **Step 8: Commit**

```bash
git add limpeza/deteccao/ tests/test_dependencia.py
git commit -m "Deteccao por dependencia funcional: marca desvio da moda condicionada"
```

---

### Task 3: O campo no Trabalho e o reaproveitamento pela cascata

**Files:**
- Modify: `limpeza/tipos.py`, `limpeza/correcao/cascata.py`, `tests/test_dependencia.py`

**Interfaces:**
- Produces: `Trabalho.dependencia: DependenciaFuncional | None = None`

- [ ] **Step 1: Escrever o teste que falha**

Acrescentar a `tests/test_dependencia.py`:

```python
def test_cascata_reaproveita_a_fd_em_vez_de_propor(monkeypatch):
    """A FD que marcou e' a que corrige; propor de novo poderia discordar."""
    from limpeza.correcao import cascata, fd as fd_real

    chamou = []
    monkeypatch.setattr(fd_real, "propor_fd",
                        lambda *a, **k: chamou.append(a) or DependenciaFuncional(
                            determinante="OUTRA", dependente="State", justificativa="x"))

    tabela = _tabela_environment(["State"])
    trabalhos = _trabalhos(tabela)
    trabalhos[0].dependencia = DependenciaFuncional(
        determinante="City", dependente="State", justificativa="ja proposta na deteccao")
    mascara = _mascara_zerada(tabela)
    mascara.loc[1, "State"] = 1  # uma celula marcada, para a cascata ter o que fazer

    cascata.rodar_cascata(trabalhos, tabela, mascara,
                          {"especificador": None, "codigo": None, "fd": None})
    assert chamou == [], "a cascata propos FD de novo em vez de reaproveitar"
```

> Se `rodar_cascata` precisar de agentes reais para a camada 1, monkeypatche
> `cascata._camada_codigo` para devolver uma correção vazia — o alvo deste
> teste é a camada 2, não a 1.

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_dependencia.py -k reaproveita -v`
Expected: FAIL — hoje a cascata sempre propõe

- [ ] **Step 3: Fazer a cascata preferir o que existe**

> O campo `Trabalho.dependencia` **já existe** — nasceu na Task 2, junto com o
> código que o escreve. Esta tarefa só ensina a cascata a lê-lo.

Em `limpeza/correcao/cascata.py`, camada 2. Hoje:

```python
    candidatos = fd_mod.candidatos_determinantes(df, coluna, limiar=mi_threshold)
    if pendentes and candidatos:
        fd = fd_mod.propor_fd(coluna, rows, candidatos, df, agente=agentes.get("fd"))
        gate_fd = fd_mod.validar_fd(fd, rows, df, mascara_completa, coluna)
```

Passa a ser:

```python
    fd = trabalho.dependencia  # ja proposta e aprovada na deteccao
    if fd is None and pendentes:
        candidatos = fd_mod.candidatos_determinantes(df, coluna, limiar=mi_threshold)
        if candidatos:
            fd = fd_mod.propor_fd(coluna, rows, candidatos, df, agente=agentes.get("fd"))
    if fd is not None and pendentes:
        gate_fd = fd_mod.validar_fd(fd, rows, df, mascara_completa, coluna)
```

`rodar_coluna` precisa receber o `trabalho` (ou só a `dependencia`) — ajuste a
assinatura e a chamada em `rodar_cascata`. Mantenha o resto do corpo intacto.

> **Por que validar de novo uma FD já aprovada:** o gate da detecção usou
> `mascara_intra`; o da cascata usa a máscara **combinada**, que exclui da moda
> as células que a FD acabou de marcar. É o duplo filtro trabalhando com
> informação nova, e revalidar é mais barato que raciocinar sobre quando
> divergiria.

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_dependencia.py -v`
Expected: PASS, 10 testes

- [ ] **Step 5: Rodar a suíte**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: 124 passed, invariante intacto

- [ ] **Step 6: Commit**

```bash
git add limpeza/tipos.py limpeza/correcao/cascata.py tests/test_dependencia.py
git commit -m "A FD aprovada viaja no Trabalho e a cascata reaproveita"
```

---

### Task 4: Ligar no pipeline, e o teste do `beers` zero

**Files:**
- Modify: `limpeza/pipeline.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Escrever o teste mais importante do projeto**

Acrescentar a `tests/test_pipeline.py`:

```python
def test_beers_nao_ganha_nenhuma_marca_de_fd():
    """O beers nao tem erro cross-column; uma marca sequer e' vazamento."""
    import pandas as pd
    from limpeza.deteccao import dependencia
    from limpeza.esquemas import DependenciaFuncional
    from limpeza.tipos import Amostra, Coluna, Tabela, Trabalho

    LER = dict(dtype=str, keep_default_na=False, na_values=[])
    raiz = Path(__file__).parent / "fixtures"
    sujo = pd.read_csv(raiz / "beers_dirty_300.csv", **LER)
    limpo = pd.read_csv(raiz / "beers_clean_300.csv", **LER)

    col = Coluna(nome="state", sujo=sujo["state"], limpo=limpo["state"],
                 valores_distintos=sorted(sujo["state"].unique()),
                 contagem=sujo["state"].value_counts().to_dict())
    tabela = Tabela(sujo=sujo, limpo=limpo, nome="beers", colunas=[col])
    linhas = list(range(10))
    rot = [{"indice": i, "sujo": col.sujo.at[i], "limpo": col.limpo.at[i],
            "eh_erro": col.sujo.at[i] != col.limpo.at[i]} for i in linhas]
    trabalho = Trabalho(coluna=col, detector=None,
                        amostra=Amostra(representantes=[], linhas=set(linhas),
                                        rotulados=rot, total_linhas=len(sujo),
                                        total_distintos=0))

    class _Agente:
        def invoke(self, _a):
            return DependenciaFuncional(determinante="brewery-name", dependente="state",
                                        justificativa="cervejaria fixa o estado")

    zerada = pd.DataFrame(0, index=sujo.index, columns=sujo.columns, dtype=int)
    m = dependencia.detectar_dependencia([trabalho], tabela, zerada, _Agente())
    marcadas = int(m["state"].sum())
    erradas = int((sujo["state"] != limpo["state"]).sum())
    assert marcadas <= erradas, (
        f"marcou {marcadas} celulas em state, mas so' {erradas} estao erradas: "
        "a FD esta marcando celula correta")
```

> **Por que `<=` e não `== 0`:** o `beers` TEM erros em `state` (células
> vazias), e eles são alcançáveis intra-coluna. A FD pode legitimamente marcar
> algumas. O que não pode acontecer é marcar **célula correta** — e é isso que
> o teste trava. Se você conseguir uma asserção mais forte depois de rodar,
> aperte-a.

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -k beers_nao_ganha -v`
Expected: FAIL com `ImportError` ou `AttributeError`, porque a etapa ainda não está ligada

- [ ] **Step 3: Ligar no pipeline**

Em `limpeza/pipeline.py::gerar_limpador`, trocar:

```python
    mascara = deteccao.construir_mascara(trabalhos, tabela)
```

por:

```python
    mascara_intra = deteccao.construir_mascara(trabalhos, tabela)
    mascara_fd = deteccao.detectar_dependencia(
        trabalhos, tabela, mascara_intra, agentes["fd"])
    # A etapa nova recebe a mascara ANTERIOR: e' isso que evita a circularidade.
    mascara = ((mascara_intra + mascara_fd) > 0).astype(int)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -k beers_nao_ganha -v`
Expected: PASS

- [ ] **Step 5: Rodar a suíte inteira**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: 124 passed. **`test_invariante.py` e `test_estatico_congelado.py` TÊM de continuar passando** — se algum quebrou, a etapa nova mudou a máscara do `beers`.

- [ ] **Step 6: Confirmar que a costura não cresceu**

Run: `git grep -c 'nome_funcao="detectar"' -- "limpeza/"`
Expected: só `limpeza/deteccao/regra.py`

- [ ] **Step 7: Commit**

```bash
git add limpeza/pipeline.py tests/test_pipeline.py
git commit -m "Liga a deteccao por FD no pipeline, combinando as mascaras"
```

---

### Task 5: O relatório

**Files:**
- Modify: `limpeza/relatorio.py`, `tests/test_dependencia.py`

- [ ] **Step 1: Escrever o teste que falha**

Acrescentar a `tests/test_dependencia.py`:

```python
def test_a_justificativa_da_fd_sai_no_relatorio(tmp_path):
    """A cadeia de pensamento e' produto; a FD tambem tem uma."""
    from limpeza import relatorio
    from limpeza.esquemas import DependenciaFuncional

    from limpeza.tipos import Detector

    tabela = _tabela_environment(["State"])
    trabalhos = _trabalhos(tabela)
    # `_cadeias_md` le detector.erro_provavel sem guarda; em producao o detector
    # nunca e' None (o pipeline sempre poe um, real ou nulo).
    trabalhos[0].detector = Detector(codigo="def detectar(col):\n    return col.isin([])\n",
                                     funcao=None, cadeia="sem regra intra-coluna")
    trabalhos[0].medida = {"deteccao": {}}
    trabalhos[0].dependencia = DependenciaFuncional(
        determinante="City", dependente="State",
        justificativa="cada cidade pertence a um unico estado")

    md = relatorio._cadeias_md("environment", trabalhos)
    assert "cada cidade pertence a um unico estado" in md
    assert "City" in md
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_dependencia.py -k justificativa -v`
Expected: FAIL — a justificativa não é publicada

- [ ] **Step 3: Publicar a justificativa e corrigir a frase que ficou falsa**

Em `limpeza/relatorio.py::_cadeias_md`, o cabeçalho hoje afirma:

```python
        f"\nDeteccao INTRA-COLUNA (a regra ve um escalar, nao a linha). "
```

Isso deixou de ser toda a verdade. Trocar por:

```python
        f"\nDuas vias: a regra intra-coluna ve um escalar, e a dependencia "
        f"funcional ve a linha. Modelo `{config.MODELO_LLM}`.\n",
```

E, no laço por coluna, acrescentar um bloco quando `trabalho.dependencia` não
é `None`:

```python
        if trabalho.dependencia is not None:
            fd = trabalho.dependencia
            partes.append(
                f"\n**Dependencia funcional:** `{fd.determinante}` -> `{fd.dependente}`\n")
            partes.append(f"**Justificativa:**\n\n{fd.justificativa}\n")
```

`detector` pode ser `None` no caminho de falha — o bloco existente já lida com
isso ou precisa de uma guarda; confira antes de acrescentar o novo.

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_dependencia.py -v`
Expected: PASS, 10 testes

- [ ] **Step 5: Rodar a suíte**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: 125 passed, invariante intacto

- [ ] **Step 6: Commit**

```bash
git add limpeza/relatorio.py tests/test_dependencia.py
git commit -m "Publica a justificativa da FD no cadeias_deteccao.md"
```

---

### Task 6: Documentação

**Files:**
- Modify: `CLAUDE.md`, `docs/DECISOES.md`, `README.md`, `CHANGELOG.md`

- [ ] **Step 1: Medir os números finais**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Run: `.venv/Scripts/python tests/medir_verbosidade.py | tail -1`

Anotar os dois — eles entram nos documentos.

- [ ] **Step 2: `CLAUDE.md`**

Três lugares:

- **Seção 3 (arquitetura):** acrescentar `deteccao/dependencia.py` na árvore, e
  atualizar o corpo colado de `gerar_limpador`, que mudou na Task 4.
- **Seção 4 (os tipos):** o campo `dependencia` no `Trabalho`, com o aviso de
  que ele só é preenchido quando a FD **passou no gate**.
- **Seção 8 (a costura):** confirmar que continua **10 pontos em 7 arquivos**.
  A detecção por FD não é ponto novo — ela não conhece `detectar(col)`.

E uma seção nova, curta, sobre as duas vias de detecção: a intra-coluna vê o
valor, a FD vê a linha; a FD roda depois e recebe a máscara anterior (isso
evita a circularidade); a moda só acha erro que é minoria no grupo, e o gate
de 100% é a proteção.

- [ ] **Step 3: `docs/DECISOES.md`**

Uma entrada nova, no formato das existentes (âncora HTML, Decisão / Por quê /
Onde). Conteúdo: por que a FD detecta e não só corrige, com os números que
autorizaram — `beers` 0% de erro inalcançável, `environment` 21,5% com
`State` 165/165 e `Climate_Zone` 82/82, e o F1=1,00 da moda condicionada.
Registrar também por que o contrato geral `detectar(df)` foi descartado: nenhum
dataset disponível tem erro cross-column não-FD, e a rota mudaria `_ESTATICO`,
obrigando a regerar a fixture congelada.

- [ ] **Step 4: `README.md` e `CHANGELOG.md`**

README: uma subseção curta explicando que a detecção agora tem duas vias, e o
que isso muda para quem lê o `cadeias_deteccao.md`.

CHANGELOG: entrada sob `## [Não publicado]`, com `### Adicionado`, citando os
números da medição.

- [ ] **Step 5: Verificação final**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: 125 passed

Run: `.venv/Scripts/python tests/medir_verbosidade.py | tail -1`
Expected: densidade ≤ 14%

Run: `.venv/Scripts/python -c "import glob;print(max((len(open(f,encoding='utf-8').read().splitlines()),f) for f in glob.glob('limpeza/**/*.py',recursive=True)))"`
Expected: nenhum arquivo novo acima de 150 linhas

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md docs/DECISOES.md README.md CHANGELOG.md
git commit -m "Documenta a deteccao por dependencia funcional"
```

---

## Validação com LLM real — depois do plano, e é decisão do usuário

Nenhuma tarefa acima gasta API: todos os testes usam agente dublê.

O **nível 2** do critério de aceitação da spec fica para depois, porque custa
dinheiro e é escolha do usuário:

```bash
.venv/Scripts/python main.py \
  --sujo tests/fixtures/environment_dirty_300.csv \
  --limpo tests/fixtures/environment_clean_300.csv \
  --colunas City,State,Country,Climate_Zone
```

O que se espera ver: `State` e `Climate_Zone` saindo de **recall 0** para algo
alto, e a justificativa da FD no `cadeias_deteccao.md`. O número é **relatado,
não travado** — a FD é escolhida pelo LLM, que pode propor um determinante
diferente e igualmente válido.
