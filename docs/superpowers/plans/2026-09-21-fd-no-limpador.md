# A detecção por FD viaja para o limpador gerado — Plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** o limpador gerado passa a detectar com as FDs aprovadas na detecção da POC, e a máscara dele fica idêntica à da POC, célula por célula.

**Architecture:** `escrever_limpador` emite um dicionário novo, `DEPENDENCIAS`, com as FDs de `trabalho.dependencia`. `_ESTATICO::_mascara` passa a somar à máscara intra os desvios da moda condicionada de cada FD, sempre lendo a máscara intra. Um teste de equivalência compara a máscara da POC com a de um limpador gerado de verdade. A fixture do `beers` recebe o `_ESTATICO` novo sem mudar os sete números; uma fixture nova do `environment` sai de um run pago.

**Tech Stack:** Python 3.12, pandas 3, pytest. Nenhuma dependência nova.

**Spec:** `docs/superpowers/specs/2026-09-21-fd-no-limpador-design.md`

## Global Constraints

- Idioma: português em tudo. Sem acento em identificadores, strings de log e em todo o texto do limpador gerado (ASCII-only); documentos `.md` com acentuação normal.
- Docstring de 1 linha no código do repositório; comentário inline com teto de 2 linhas. As docstrings DENTRO de `_ESTATICO` seguem o estilo multi-linha que já existe lá (são documentação do arquivo entregue ao usuário).
- Densidade de comentário ≤ 14% (`.venv/Scripts/python tests/medir_verbosidade.py`).
- Invariante do `beers` (`tests/test_invariante.py`): erros=495, mudancas=121, tp=121, precisao=1.0, recall=0.2444, f1=0.3929, flags=71. **Não pode mudar em nenhuma tarefa.**
- `limpeza/sandbox.py`: não tocar.
- A costura cross-column continua em 10 pontos / 7 arquivos (CLAUDE.md §8).
- Nenhum teste faz chamada de rede. Ninguém roda `main.py` sem `--help`, exceto o controlador na Task 4.
- Empate da moda: sempre `valores.mode().iloc[0]`, em todas as cópias.
- Todo teste novo vem com prova por mutação no relatório: a mutação indicada, o teste falhando, `git checkout --` revertendo.
- Python: `.venv/Scripts/python`. Branch: `fd-no-limpador`.

---

## Mapa de arquivos

| Arquivo | Tarefa | Responsabilidade |
|---|---|---|
| `limpeza/empacotar.py` (fora de `_ESTATICO`) | 1, 3 | emitir `DEPENDENCIAS`; tirar o aviso "LIMITACAO CONHECIDA" |
| `limpeza/empacotar.py::_ESTATICO` | 2 | `_mascara` combinada; `_desvios_da_moda` novo |
| `tests/fixtures/limpador_beers_congelado.py` | 2 | `DEPENDENCIAS = {}` em cima; `_ESTATICO` novo embaixo |
| `tests/test_empacotar.py` | 1, 3 | `DEPENDENCIAS` só com FD de detecção; ponta a ponta com FD injetada |
| `tests/test_equivalencia.py` (novo) | 2 | máscara POC × limpador, 4 cenários |
| `tests/test_estatico_congelado.py` | 2, 4 | vale para toda fixture `limpador_*_congelado.py` |
| `tests/fixtures/limpador_environment_congelado.py` (novo) | 4 | saída do run pago |
| `tests/test_invariante.py` | 4 | invariante do `environment` |
| `CLAUDE.md`, `README.md`, `docs/DECISOES.md`, `CHANGELOG.md` | 5 | tirar a limitação, registrar as decisões |

---

### Task 1: `escrever_limpador` emite `DEPENDENCIAS`

**Files:**
- Modify: `limpeza/empacotar.py` (`escrever_limpador`, `gerar_limpador`)
- Test: `tests/test_empacotar.py`

**Interfaces:**
- Produces: `escrever_limpador(caminho, dataset, detectores_codigo, plano_correcao, colunas, dependencias=None) -> Path`. `dependencias` é `{coluna: (determinante, dependente)}`; `None` vira `{}`. O arquivo gerado ganha a linha `DEPENDENCIAS = {...}` entre `_CORRETORES` e `FDS`, no mesmo formato de `FDS`.
- Produces: `gerar_limpador(trabalhos, saida, dataset)` passa `dependencias` com uma entrada por trabalho cujo `dependencia` não é `None`.

Esta tarefa **não** toca em `_ESTATICO`. O limpador gerado ganha o dicionário, mas nada o lê ainda. Por isso a fixture congelada e o invariante não mudam.

- [ ] **Step 1: Escrever o teste que falha**

Acrescentar ao fim de `tests/test_empacotar.py`:

```python
def test_dependencias_so_leva_fd_aprovada_na_deteccao(tmp_path):
    """FD que so' a cascata usou fica em FDS; so' a da deteccao entra em DEPENDENCIAS."""
    from limpeza.esquemas import DependenciaFuncional
    from limpeza.tipos import Coluna, Correcao, Detector, Trabalho

    nada = "def detectar(col):\n    return col.isin([])\n"

    def trabalho(nome, passos, dependencia=None):
        serie = pd.Series(["a"], name=nome)
        return Trabalho(
            coluna=Coluna(nome=nome, sujo=serie, limpo=serie,
                          valores_distintos=["a"], contagem={"a": 1}),
            amostra=None,
            detector=Detector(codigo=nada, funcao=None, cadeia=""),
            dependencia=dependencia,
            correcao=Correcao(passos=passos, trilha={}, cadeia=""),
        )

    fd_state = {"tipo": "fd", "determinante": "City", "dependente": "State"}
    fd_city = {"tipo": "fd", "determinante": "brewery", "dependente": "city"}
    trabalhos = [
        trabalho("City", []),
        trabalho("State", [fd_state], DependenciaFuncional(
            determinante="City", dependente="State", justificativa="teste")),
        trabalho("city", [fd_city]),
    ]
    caminho = empacotar.gerar_limpador(trabalhos, tmp_path / "limpador_dep.py", "teste")
    spec = importlib.util.spec_from_file_location("limpador_dep", caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.DEPENDENCIAS == {"State": ("City", "State")}
    assert mod.FDS == {"State": ("City", "State"), "city": ("brewery", "city")}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_empacotar.py -k dependencias -v`
Expected: FAIL com `AttributeError: module 'limpador_dep' has no attribute 'DEPENDENCIAS'`

- [ ] **Step 3: Implementar**

Em `escrever_limpador`, acrescentar o parâmetro:

```python
def escrever_limpador(
    caminho,
    dataset: str,
    detectores_codigo: dict,
    plano_correcao: dict,
    colunas: list,
    dependencias: dict | None = None,
) -> Path:
```

E, na lista `blocos`, entre `_CORRETORES` e `FDS`:

```python
        f"_CORRETORES = {_dict_funcs(corretores_por_col)}",
        "",
        f"DEPENDENCIAS = {_dict_fds(dependencias or {})}",
        "",
        f"FDS = {_dict_fds(fds_por_col)}",
```

Em `gerar_limpador`, passar o argumento novo:

```python
        colunas=[t.coluna.nome for t in trabalhos],
        dependencias={t.coluna.nome: (t.dependencia.determinante, t.dependencia.dependente)
                      for t in trabalhos if t.dependencia is not None},
    )
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_empacotar.py -v`
Expected: PASS em todos

- [ ] **Step 5: Prova por mutação**

Em `gerar_limpador`, trocar `if t.dependencia is not None` por `if t.correcao and t.correcao.passos` e montar a tupla a partir do passo `fd` (o erro plausível: tirar a FD de `FDS` em vez de `dependencia`). O teste tem de falhar em `assert mod.DEPENDENCIAS == ...`, porque `city` entraria. Reverter com `git checkout -- limpeza/empacotar.py` e colar as duas saídas no relatório.

- [ ] **Step 6: Suíte inteira**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: tudo verde (130 testes), invariante e `test_estatico_congelado` inclusos.

- [ ] **Step 7: Commit**

```bash
git add limpeza/empacotar.py tests/test_empacotar.py
git commit -m "O limpador gerado leva DEPENDENCIAS: so' FD aprovada na deteccao"
```

---

### Task 2: `_mascara` combinada, fixture do `beers` e teste de equivalência

**Files:**
- Modify: `limpeza/empacotar.py::_ESTATICO` (`_mascara`; `_desvios_da_moda` novo logo depois dela)
- Modify: `tests/fixtures/limpador_beers_congelado.py`
- Modify: `tests/test_estatico_congelado.py`
- Create: `tests/test_equivalencia.py`

**Interfaces:**
- Consumes: `DEPENDENCIAS` emitido pela Task 1; `deteccao.construir_mascara(trabalhos, tabela)`, `deteccao.detectar_dependencia(trabalhos, tabela, mascara_intra, agente)`, `deteccao.materializar(codigo)`.
- Produces: no limpador gerado, `_mascara(df) -> DataFrame[bool]` (intra OU desvios das FDs) e `_desvios_da_moda(df, mascara, det, dep) -> Series[bool]`.

As três mudanças vão num commit só porque não existem separadas: mudar `_ESTATICO` sem mudar a fixture quebra `test_estatico_congelado`, e a fixture nova sem `DEPENDENCIAS = {}` levanta `NameError` no invariante.

- [ ] **Step 1: Escrever o teste de equivalência**

Criar `tests/test_equivalencia.py`:

```python
"""A mascara do limpador gerado e' a mesma da POC, celula por celula."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from limpeza import deteccao, empacotar
from limpeza.correcao import fd as fd_mod
from limpeza.esquemas import DependenciaFuncional
from limpeza.tipos import Amostra, Coluna, Detector, Tabela, Trabalho

FIXTURES = Path(__file__).parent / "fixtures"
LER = dict(dtype=str, keep_default_na=False, na_values=[])
NADA = "def detectar(col):\n    return col.isin([])\n"
ENV_FDS = {"State": ("City", "State"), "Climate_Zone": ("City", "Climate_Zone")}


def _environment():
    return pd.read_csv(FIXTURES / "environment_dirty_300.csv", **LER)


def _encadeado():
    # City -> State marca as linhas 5-7; em State -> Country elas mudam a moda de S1.
    return pd.DataFrame({
        "City": ["X", "Y", "Y", "Y", "Y", "Y", "Y", "Y"],
        "State": ["S1", "S3", "S3", "S3", "S3", "S1", "S1", "S1"],
        "Country": ["C1", "C3", "C3", "C3", "C3", "C2", "C2", "C2"],
    })


def _empate():
    return pd.DataFrame({"City": ["Bangalore", "Bangalore"], "State": ["KA", "AA"]})


CENARIOS = {
    "intra_vazia": (_environment, {}, ENV_FDS),
    "intra_realista": (_environment,
                       {"City": "def detectar(col):\n    return col.eq('Hamburg')\n"},
                       ENV_FDS),
    "empate": (_empate, {}, {"State": ("City", "State")}),
    "encadeadas": (_encadeado, {},
                   {"State": ("City", "State"), "Country": ("State", "Country")}),
}


def _trabalhos(df, detectores, fds):
    saida = []
    for nome in df.columns:
        codigo = detectores.get(nome, NADA)
        det = fds.get(nome)
        saida.append(Trabalho(
            coluna=Coluna(nome=nome, sujo=df[nome], limpo=df[nome],
                          valores_distintos=sorted(df[nome].unique()),
                          contagem=df[nome].value_counts().to_dict()),
            amostra=Amostra(representantes=[], linhas=set(), rotulados=[],
                            total_linhas=len(df), total_distintos=0),
            detector=Detector(codigo=codigo, funcao=deteccao.materializar(codigo),
                              cadeia=""),
            dependencia=DependenciaFuncional(determinante=det[0], dependente=nome,
                                             justificativa="teste") if det else None,
        ))
    return saida


def _mascara_poc(df, detectores, fds, monkeypatch):
    """O caminho do pipeline: intra, depois FD sobre a intra, depois OU."""
    monkeypatch.setattr(fd_mod, "candidatos_determinantes",
                        lambda _df, alvo, limiar=None: [fds[alvo][0]] if alvo in fds else [])
    monkeypatch.setattr(fd_mod, "propor_fd", lambda dep, *_a, **_k: DependenciaFuncional(
        determinante=fds[dep][0], dependente=dep, justificativa="teste"))
    monkeypatch.setattr(fd_mod, "validar_fd", lambda *_a, **_k: True)
    trabalhos = _trabalhos(df, detectores, fds)
    tabela = Tabela(sujo=df, limpo=df, nome="teste", colunas=[t.coluna for t in trabalhos])
    intra = deteccao.construir_mascara(trabalhos, tabela)
    fd = deteccao.detectar_dependencia(trabalhos, tabela, intra, None)
    return (intra + fd) > 0


def _mascara_limpador(df, detectores, fds, pasta):
    """O caminho do produto: gera o limpador de verdade e chama a _mascara dele."""
    caminho = empacotar.gerar_limpador(
        _trabalhos(df, detectores, fds), pasta / "limpador_eq.py", "teste")
    spec = importlib.util.spec_from_file_location("limpador_eq", caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._mascara(df)


@pytest.mark.parametrize("nome", list(CENARIOS))
def test_mascara_do_limpador_e_a_da_poc(nome, monkeypatch, tmp_path):
    fabrica, detectores, fds = CENARIOS[nome]
    df = fabrica()
    poc = _mascara_poc(df, detectores, fds, monkeypatch)
    limpador = _mascara_limpador(df, detectores, fds, tmp_path)
    pd.testing.assert_frame_equal(limpador, poc, check_dtype=False)


def test_cenarios_discriminam(monkeypatch):
    """Sem isto os cenarios podem passar por coincidencia, nao por equivalencia."""
    vazia = _mascara_poc(_environment(), {}, ENV_FDS, monkeypatch)
    realista = _mascara_poc(_environment(), CENARIOS["intra_realista"][1], ENV_FDS,
                            monkeypatch)
    assert int(vazia["State"].sum()) == 54
    assert int(realista["State"].sum()) == 42
    empate = _mascara_poc(_empate(), {}, {"State": ("City", "State")}, monkeypatch)
    assert empate["State"].tolist() == [True, False]
    encadeada = _mascara_poc(_encadeado(), {}, CENARIOS["encadeadas"][2], monkeypatch)
    assert encadeada["Country"].tolist() == [True] + [False] * 7
```

Os números de `test_cenarios_discriminam` foram medidos antes do plano: 54 marcas em `State` com a intra vazia; 42 quando a intra marca `City == 'Hamburg'` (o duplo filtro tira essas linhas da moda); no empate, a moda de `[KA, AA]` é `AA` e marca `KA`; no encadeado, a POC marca só a linha 0 de `Country`, e uma cópia circular marcaria as linhas 5-7.

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_equivalencia.py -v`
Expected: `test_cenarios_discriminam` PASS (só usa a POC); os quatro `test_mascara_do_limpador_e_a_da_poc` FAIL. O limpador de hoje não marca nada por FD, e o `assert_frame_equal` aponta a coluna `State`.

- [ ] **Step 3: Mudar `_mascara` em `_ESTATICO`**

Em `limpeza/empacotar.py`, dentro de `_ESTATICO`, trocar a docstring de `_mascara` e o final dela. A docstring nova:

```python
def _mascara(df):
    """Espelho de deteccao.construir_mascara + detectar_dependencia: mascara BOOL.

    1. Intra-coluna: inicializa tudo-False no shape do df. Por coluna COM
       detector, chama `_detectar_<col>(df[col])` UMA vez e coage
       `res.fillna(False).astype(bool)`, atribuindo por POSICAO via `.to_numpy()`
       (nunca boolean-index por indice). Se o detector lanca, nao devolve Series
       ou devolve tamanho errado, a coluna INTEIRA fica nao-marcada.
    2. Dependencia funcional: por coluna em DEPENDENCIAS, soma (OU) as celulas que
       desviam da moda condicionada. Toda FD le a mascara INTRA do passo 1, nunca
       a que esta sendo acumulada -- a mesma ordem da POC, que evita circularidade.
    """
```

O laço intra continua **exatamente** como está. Depois dele, no lugar do `return mascara`:

```python
        mascara[coluna] = marca
    intra = mascara.copy()
    for coluna, (det, dep) in DEPENDENCIAS.items():
        if det not in df.columns or dep not in df.columns:
            continue
        mascara[coluna] = mascara[coluna] | _desvios_da_moda(df, intra, det, dep)
    return mascara


def _desvios_da_moda(df, mascara, det, dep):
    """Espelho de deteccao/dependencia.py: True onde `dep` difere da moda do grupo.

    A moda de `dep` em cada grupo `df[det] == valor` usa so' as linhas com
    `mascara[det]` e `mascara[dep]` nao-marcados (DUPLO FILTRO, igual a
    fd.moda_condicionada). Empate: `mode().iloc[0]`, o menor valor. Grupo sem
    linha compativel nao marca ninguem.
    """
    marca = pd.Series(False, index=df.index)
    for valor_det in df[det].unique():
        no_grupo = df[det] == valor_det
        condicao = (
            no_grupo
            & (mascara[det] == False)  # noqa: E712 -- espelha mascara[det]==0 da POC
            & (mascara[dep] == False)  # noqa: E712
        )
        valores = df.loc[condicao, dep]
        if valores.empty:
            continue
        marca |= no_grupo & (df[dep] != valores.mode().iloc[0])
    return marca
```

`_aplicar_fd` e `aplicar` **não mudam**: `aplicar` já passa `m` a `_aplicar_fd`, e `m` agora é a máscara combinada, a mesma que `cascata.rodar_coluna` usa (`mascara_completa`).

- [ ] **Step 4: Rodar a equivalência e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_equivalencia.py -v`
Expected: PASS nos 5.

- [ ] **Step 5: Atualizar a fixture do `beers`, só onde o `_ESTATICO` e o `DEPENDENCIAS` mudam**

A metade de cima (o que o LLM escreveu em 20/ago) fica intocada, a não ser pela linha `DEPENDENCIAS = {}`, que é o que `escrever_limpador` emitiria para um run sem FD de detecção. Rodar este script **uma vez**, sem commitá-lo (salvar no diretório temporário):

```python
from pathlib import Path

from limpeza.empacotar import _ESTATICO

p = Path("tests/fixtures/limpador_beers_congelado.py")
texto = p.read_text(encoding="utf-8")
corte = texto.index("\ndef _mascara(df):")
cabeca = texto[:corte]
assert cabeca.count("\nFDS = {") == 1 and "DEPENDENCIAS" not in cabeca
cabeca = cabeca.replace("\nFDS = {", "\nDEPENDENCIAS = {}\n\nFDS = {")
p.write_text((cabeca + _ESTATICO).rstrip() + "\n", encoding="utf-8")
```

Conferir o diff: `git diff --stat tests/fixtures/limpador_beers_congelado.py` e `git diff tests/fixtures/limpador_beers_congelado.py | head -40`. Antes de `def _mascara`, a **única** linha acrescentada pode ser `DEPENDENCIAS = {}` com a linha em branco que a segue. Qualquer outra diferença na metade de cima é erro do script.

- [ ] **Step 6: `test_estatico_congelado` vale para toda fixture congelada**

Substituir o conteúdo de `tests/test_estatico_congelado.py`:

```python
"""Trava que o corpo estatico de empacotar.py bate com cada fixture congelada."""
from pathlib import Path

import pytest

from limpeza.empacotar import _ESTATICO

FIXTURES = Path(__file__).parent / "fixtures"
CONGELADAS = sorted(FIXTURES.glob("limpador_*_congelado.py"))


def test_a_fixture_do_beers_esta_entre_as_congeladas():
    assert "limpador_beers_congelado.py" in [p.name for p in CONGELADAS]


@pytest.mark.parametrize("fixture", CONGELADAS, ids=lambda p: p.name)
def test_estatico_bate_com_a_fixture_congelada(fixture):
    corpo = _ESTATICO.strip("\n")
    assert corpo, "_ESTATICO vazio -- extracao quebrada, nao um match vacuoso"
    texto = fixture.read_text(encoding="utf-8")
    assert texto.rstrip("\n").endswith(corpo), (
        f"o corpo estatico atual de limpeza/empacotar.py diverge do final de "
        f"{fixture.name} -- o limpador que empacotar.py gera hoje nao e' mais o "
        f"mesmo que o invariante dessa fixture mede"
    )
```

- [ ] **Step 7: Suíte inteira — o invariante do `beers` não pode ter se movido**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: tudo verde (136 testes).

Run: `.venv/Scripts/python -m pytest tests/test_invariante.py -v`
Expected: PASS, com os mesmos 495/121/121/1.0/0.2444/0.3929/71. Se um número mudou, o maquinário novo está interferindo num limpador sem FD de detecção: é bug, não se mexe no `assert`.

- [ ] **Step 8: Prova por mutação (três, uma por propriedade)**

1. Em `_ESTATICO`, trocar o laço de `DEPENDENCIAS` por `pass` → os cenários `intra_vazia`, `intra_realista`, `empate` e `encadeadas` falham.
2. Trocar `_desvios_da_moda(df, intra, det, dep)` por `_desvios_da_moda(df, mascara, det, dep)` (circularidade) → **só** `encadeadas` falha. Se ele não falhar, o cenário não discrimina e a tarefa não está pronta.
3. Trocar `valores.mode().iloc[0]` por `valores.mode().iloc[-1]` em `_desvios_da_moda` → pelo menos `empate` falha.

Não é preciso regenerar nada, porque os testes geram o limpador na hora. Reverter cada mutação com `git checkout -- limpeza/empacotar.py`. Colar as saídas no relatório.

- [ ] **Step 9: Commit**

```bash
git add limpeza/empacotar.py tests/fixtures/limpador_beers_congelado.py tests/test_estatico_congelado.py tests/test_equivalencia.py
git commit -m "A mascara do limpador soma a FD: igual a da POC, celula por celula"
```

---

### Task 3: Ponta a ponta com FD injetada, e sai o aviso de limitação

**Files:**
- Modify: `limpeza/empacotar.py::_montar_cabecalho`
- Test: `tests/test_empacotar.py`

**Interfaces:**
- Consumes: `escrever_limpador(..., dependencias=...)` da Task 1 e a `_mascara` da Task 2; `avaliar_limpador.avaliar(caminho_limpador, caminho_sujo, caminho_limpo)`, que devolve `{"colunas": {col: {"erros", "mudancas", "tp", ...}}, "total": {...}}`.

- [ ] **Step 1: Escrever o teste**

Acrescentar a `tests/test_empacotar.py`:

```python
def test_limpador_com_fd_injetada_corrige_o_environment(tmp_path):
    """Sem nenhum detector intra, so' a FD empacotada acha e corrige State e Climate_Zone."""
    from avaliar_limpador import avaliar

    fixtures = Path(__file__).parent / "fixtures"
    sujo = fixtures / "environment_dirty_300.csv"
    fds = {"State": ("City", "State"), "Climate_Zone": ("City", "Climate_Zone")}
    colunas = list(pd.read_csv(sujo, nrows=0).columns)
    caminho = empacotar.escrever_limpador(
        caminho=tmp_path / "limpador_env.py", dataset="environment",
        detectores_codigo={}, colunas=colunas,
        plano_correcao={dep: [{"tipo": "fd", "determinante": det, "dependente": dep}]
                        for dep, (det, _) in fds.items()},
        dependencias=fds,
    )
    medida = avaliar(caminho, sujo, fixtures / "environment_clean_300.csv")["colunas"]
    for coluna, erros in (("State", 54), ("Climate_Zone", 26)):
        assert medida[coluna]["erros"] == erros
        assert medida[coluna]["mudancas"] == erros
        assert medida[coluna]["tp"] == erros
```

Os números foram medidos antes do plano: com `City` como determinante, a FD marca os 54 erros de `State` e os 26 de `Climate_Zone`, nenhuma célula correta, e a moda do grupo acerta todas as correções. Antes da Task 2 esse mesmo limpador dava 0 mudanças.

- [ ] **Step 2: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_empacotar.py -k environment -v`
Expected: PASS

- [ ] **Step 3: Prova por mutação**

Trocar `dependencias=fds` por `dependencias={}` no teste → FAIL com `mudancas == 0`. Reverter. Colar as saídas no relatório. É esta a prova de que o problema da spec (seção 1) acabou.

- [ ] **Step 4: Tirar o aviso de limitação do cabeçalho**

Em `limpeza/empacotar.py::_montar_cabecalho`, apagar estas sete linhas da lista `linhas`:

```python
        "",
        "LIMITACAO CONHECIDA -- deteccao por dependencia funcional (FD): a POC",
        "roda uma segunda via de deteccao, entre colunas (ex.: State fixado por",
        "City), que entra nas metricas do run (precisao/recall/F1 publicados).",
        "Essa via NAO esta reproduzida neste arquivo -- a mascara abaixo usa so'",
        "os detectores intra-coluna de _DETECTORES. Celula que so' a FD",
        "encontrou no run passa batido neste limpador.",
```

Com a Task 2 esse aviso virou falso. Nenhum teste o afirma; `git grep -n "LIMITACAO" -- limpeza/ tests/` tem de responder vazio depois.

- [ ] **Step 5: Suíte inteira**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: tudo verde (137 testes), invariante intacto.

- [ ] **Step 6: Commit**

```bash
git add limpeza/empacotar.py tests/test_empacotar.py
git commit -m "Ponta a ponta: o limpador com FD corrige State e Climate_Zone; sai o aviso"
```

---

### Task 4: Run pago do `environment` e o invariante novo — **CONTROLADOR, NÃO SUBAGENTE**

**Files:**
- Create: `tests/fixtures/limpador_environment_congelado.py`
- Modify: `tests/test_invariante.py`

Esta tarefa gasta API paga. O controlador a executa pessoalmente e **anuncia ao usuário antes de disparar o run**. Nenhum subagente roda `main.py`.

- [ ] **Step 1: Anunciar e rodar**

```bash
.venv/Scripts/python main.py \
  --sujo tests/fixtures/environment_dirty_300.csv \
  --limpo tests/fixtures/environment_clean_300.csv \
  --saida runs/fd-no-limpador-environment
```

Todas as colunas (decisão do usuário, spec §2), modelo e iterações padrão do `.env`. Guardar o stdout.

- [ ] **Step 2: Ler o que o run fez**

Em `runs/fd-no-limpador-environment/`: `cadeias_deteccao.md` (quais colunas tiveram FD aprovada), `deteccao_metricas.json`, `correcao_metricas.json` e o `limpador_*.py`. Registrar no ledger:
- as entradas de `DEPENDENCIAS` e de `FDS` no limpador gerado;
- se `State` e `Climate_Zone` tiveram FD aprovada.

Se nenhuma das duas teve, **congelar assim mesmo** (spec §2 e §6): os testes das Tasks 2 e 3 já provam a detecção nova, e `DECISOES.md` registra que este run não aprovou FD.

- [ ] **Step 3: Congelar**

```bash
cp runs/fd-no-limpador-environment/limpador_*.py tests/fixtures/limpador_environment_congelado.py
```

Run: `.venv/Scripts/python -m pytest tests/test_estatico_congelado.py -v`
Expected: PASS nas duas fixtures (a do run já nasce com o `_ESTATICO` novo).

- [ ] **Step 4: Medir o limpador congelado**

```bash
.venv/Scripts/python avaliar_limpador.py \
  --limpador tests/fixtures/limpador_environment_congelado.py \
  --sujo tests/fixtures/environment_dirty_300.csv \
  --limpo tests/fixtures/environment_clean_300.csv
```

Anotar os sete números do total. Comparar com o que o run publicou nos dois `.json`. **Não precisam coincidir:** o run mede fora das linhas rotuladas (o holdout, CLAUDE.md §4), e o `avaliar_limpador` mede a tabela inteira. Toda diferença vai explicada no relatório e em `DECISOES.md`, nunca travada como igualdade.

- [ ] **Step 5: Travar o invariante**

Acrescentar a `tests/test_invariante.py`, com os números medidos no Step 4 no lugar de cada `<medido>`:

```python
def test_f1_reparo_do_limpador_environment_congelado():
    """Trava o limpador do run pago do environment: DEPENDENCIAS empacotada."""
    total = avaliar(
        caminho_limpador=FIXTURES / "limpador_environment_congelado.py",
        caminho_sujo=FIXTURES / "environment_dirty_300.csv",
        caminho_limpo=FIXTURES / "environment_clean_300.csv",
    )["total"]
    assert total["erros"] == <medido>
    assert total["mudancas"] == <medido>
    assert total["tp"] == <medido>
    assert total["precisao"] == <medido>
    assert total["recall"] == <medido>
    assert total["f1"] == <medido>
    assert total["flags"] == <medido>
```

Os `<medido>` são o único ponto do plano que não dá para preencher antes: saem do run. Nenhum pode ficar no arquivo commitado.

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: tudo verde (139 testes: o invariante novo, mais o caso novo de `test_estatico_congelado` para a fixture do `environment`).

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/limpador_environment_congelado.py tests/test_invariante.py
git commit -m "Invariante do environment: limpador de um run pago, com a FD empacotada"
```

---

### Task 5: Documentação

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `docs/DECISOES.md`, `CHANGELOG.md`

- [ ] **Step 1: Medir**

```bash
.venv/Scripts/python -m pytest tests/ -q
.venv/Scripts/python tests/medir_verbosidade.py | tail -2
git grep -cE "^\s*(for|while) " -- "limpeza/" "main.py" "avaliar_limpador.py" "escolher_modelo.py"
git grep -c "except " -- "limpeza/" "main.py" "avaliar_limpador.py" "escolher_modelo.py"
git grep -n 'nome_funcao="detectar"' -- "limpeza/"
```

Esperado: loops 49 (os 2 novos de `_ESTATICO`), exceções 20 (inalteradas), `nome_funcao` só em `regra.py`. Todo número escrito nos documentos vem destes comandos ou do ledger da Task 4.

- [ ] **Step 2: `CLAUDE.md`**

- Seção "Duas vias de detecção": apagar o parágrafo **"A via da FD não é empacotada."** e escrever o estado novo: o limpador leva `DEPENDENCIAS` (FDs aprovadas na detecção) e `FDS` (FDs de correção); `_mascara` soma os desvios sempre sobre a máscara intra; `tests/test_equivalencia.py` compara a máscara da POC com a do limpador célula por célula, e é o teste que impede as duas cópias de divergirem.
- §6 (invariante): agora são **duas** fixtures e dois invariantes. A do `beers` continua 495/121/121/1.0/0.2444/0.3929/71; a do `environment` com os números da Task 4. `test_estatico_congelado` vale para as duas.
- §7: loops 47 → 49; densidade com o número medido.
- §9 (`_ESTATICO`): acrescentar que mudar `_ESTATICO` obriga a atualizar **as duas** fixtures, só na metade de baixo.

- [ ] **Step 3: `README.md`**

A seção "Duas vias de detecção — e uma que não é empacotada" passa a se chamar "Duas vias de detecção". Tirar a afirmação de que a FD não vai para o limpador e corrigir o diagrama: as duas vias vão para as métricas **e** para o limpador. Atualizar a contagem de testes.

- [ ] **Step 4: `docs/DECISOES.md`**

- Em `#deteccao-por-fd`, trocar o parágrafo que registra a limitação por uma frase apontando para a entrada nova.
- Entrada nova `<a id="fd-no-limpador"></a>`, no formato Decisão / Por quê / Onde, com:
  - os dois dicionários e por que não um (FD de correção que o run nunca usou para detectar);
  - o teste de equivalência como guarda contra a divergência;
  - **por que o invariante do `beers` não se moveu:** o run de 20/ago não tem FD de detecção, `DEPENDENCIAS = {}`, e a fixture mudou só na metade de baixo, justamente para isolar o efeito do maquinário;
  - o invariante do `environment`: origem (run pago, todas as colunas), os sete números, quais FDs o LLM aprovou, e a comparação com as métricas publicadas pelo run, com a explicação do holdout se houver diferença.

- [ ] **Step 5: `CHANGELOG.md`**

Sob `## [Não publicado]`, `### Adicionado`: a FD empacotada no limpador, os dois invariantes, a suíte com o número medido.

- [ ] **Step 6: Verificação**

```bash
git grep -n -i "não é empacotada\|nunca viajou\|LIMITACAO" -- "*.md" "*.py" ":!docs/superpowers"
```

Expected: vazio. `.venv/Scripts/python -m pytest tests/ -q` verde e densidade ≤ 14%.

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md README.md docs/DECISOES.md CHANGELOG.md
git commit -m "Documenta a FD no limpador gerado e os dois invariantes"
```
