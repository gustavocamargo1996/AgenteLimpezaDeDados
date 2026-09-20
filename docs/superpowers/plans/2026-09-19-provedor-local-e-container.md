# Provedor local (Ollama) e implantação em container — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pôr OpenAI e Ollama atrás de um seletor num repositório só, e empacotar a POC num stack do Portainer ao lado de um serviço Ollama, para que dados sensíveis nunca saiam da rede.

**Architecture:** Um módulo novo, `limpeza/llm.py`, é o único lugar do repositório que sabe que existe mais de um provedor; os quatro construtores de agente viram uma linha cada. A escolha de modelo por papel e o timeout por provedor saem de variáveis de ambiente. A POC vira uma imagem de ~500 MB com o MiniLM assado dentro, disparada como job one-shot num stack que o Portainer builda direto do Git.

**Tech Stack:** Python 3.12, langchain-core, langchain-openai, **langchain-ollama 1.1.0**, pydantic 2, pandas, onnxruntime, Docker, docker-compose, Portainer.

**Spec:** `docs/superpowers/specs/2026-09-19-provedor-local-e-container-design.md`

## Global Constraints

- **INVARIANTE BLOQUEANTE:** com `PROVEDOR=openai` (o default), o comportamento é indistinguível do atual. Os **89 testes** passam e `tests/test_invariante.py` continua travando `erros=495, mudancas=121, tp=121, precisao=1.0, recall=0.2444, f1=0.3929, flags=71`. Se um teste mudar de resultado, o seletor vazou para onde não devia.
- **Idioma:** PORTUGUÊS em tudo — nomes, docstrings, comentários, logs, documentos, mensagens de commit. Sem acento em identificadores nem em strings de log; documentos usam acentuação normal.
- **Comentários:** docstring de módulo e de função com **1 linha**; comentário inline com teto de **2 linhas**. Nada de histórico, medição, comparação com ZeroDC ou plano de IA.
- **`PROVEDOR` desconhecido FALHA ALTO**, com mensagem clara. Nunca cai em silêncio para OpenAI — sob restrição de privacidade, um typo que mande dado para fora da rede é o pior defeito possível.
- **Nenhum teste faz chamada de rede ou de API.** Exceção já existente: `tests/test_embeddings.py`, que usa o modelo ONNX local em disco (não é rede) e tem `skipif`.
- **NÃO tocar** em `limpeza/empacotar.py::_ESTATICO` — `tests/test_estatico_congelado.py` trava o casamento byte a byte com a fixture congelada.
- **NÃO acrescentar** um décimo primeiro ponto à costura cross-column (seção 7 do `CLAUDE.md`).
- **`ChatOllama` 1.1.0 não aceita `max_retries` nem `timeout`** — verificado por inspeção. Timeout vai por `client_kwargs={"timeout": N}`; retry não existe e é omitido.

---

## File Structure

**Criados:**

| Arquivo | Responsabilidade |
|---|---|
| `limpeza/llm.py` | A fábrica: único lugar que conhece os provedores |
| `tests/test_llm.py` | Fábrica, resolução de modelo, timeout, `PROVEDOR` inválido |
| `escolher_modelo.py` | CLI que mede se um modelo serve, antes de gastar um run |
| `Dockerfile` | Imagem da POC com o MiniLM assado |
| `.dockerignore` | Mantém `runs/`, `.venv/`, `.git/` fora do contexto de build |
| `docker-compose.yml` | O stack: ollama + arquivos + poc |

**Modificados:**

| Arquivo | O quê |
|---|---|
| `limpeza/config.py` | `PROVEDOR`, `OLLAMA_URL`, modelos por papel, timeout por provedor |
| `limpeza/deteccao/regra.py` | `construir_agente` vira uma linha |
| `limpeza/correcao/regras.py` | os dois construtores viram uma linha cada |
| `limpeza/correcao/fd.py` | `construir_agente` vira uma linha |
| `main.py` | `SUJO`/`LIMPO` como default dos argumentos |
| `tests/test_pipeline.py` | testes do default por variável de ambiente |
| `requirements.txt` | `langchain-ollama>=1.1` |
| `.env.example` | as variáveis novas |
| `CLAUDE.md` | seção nova sobre provedores; seção 2 atualizada |
| `README.md`, `CHANGELOG.md` | como rodar local e em container |

---

### Task 1: A fábrica de provedores

**Files:**
- Create: `limpeza/llm.py`, `tests/test_llm.py`
- Modify: `limpeza/config.py`, `requirements.txt`

**Interfaces:**
- Produces, e estas são as assinaturas **finais** — nenhuma tarefa posterior as altera:

```python
limpeza.llm.construir(schema, papel: str, modelo: str | None = None)  # -> agente com .invoke()
limpeza.llm.cliente(provedor: str, modelo: str, segundos: int)        # -> BaseChatModel cru
limpeza.llm.modelo_do_papel(papel: str) -> str
limpeza.llm.timeout_do_provedor(provedor: str) -> int
limpeza.llm.ProvedorDesconhecido(ValueError)
limpeza.config.PROVEDOR, OLLAMA_URL, MODELOS_POR_PAPEL, TIMEOUT_POR_PROVEDOR
```

`cliente()` existe separada de `construir()` de propósito: `construir` devolve o
resultado de `with_structured_output`, que é um `RunnableSequence` e não diz
qual provedor o produziu. Os testes que precisam afirmar **qual cliente foi
escolhido** olham `cliente()`.

- Consumes: nada de tarefas anteriores

- [ ] **Step 1: Acrescentar a dependência**

Acrescentar a `requirements.txt`, depois de `langchain-openai>=0.2`:

```
langchain-ollama>=1.1
```

Rodar: `uv pip install -r requirements.txt`

- [ ] **Step 2: Escrever os testes que falham**

Criar `tests/test_llm.py`:

```python
"""A fabrica de provedores: quem e' construido, com qual modelo e qual timeout."""
import pytest
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from limpeza import config, llm
from limpeza.esquemas import RegraDeteccao


@pytest.fixture(autouse=True)
def _isola_config(monkeypatch):
    """Cada teste mexe em config; isola para nao vazar entre eles."""
    monkeypatch.setattr(config, "PROVEDOR", "openai")
    monkeypatch.setattr(config, "MODELO_LLM", "gpt-4o-mini")
    monkeypatch.setattr(config, "MODELOS_POR_PAPEL", {})
    monkeypatch.delenv("TIMEOUT_LLM", raising=False)


def test_openai_e_o_default():
    assert isinstance(llm.cliente("openai", "gpt-4o-mini", 180), ChatOpenAI)


def test_ollama_quando_configurado():
    c = llm.cliente("ollama", "qwen2.5:3b", 900)
    assert isinstance(c, ChatOllama)
    assert c.model == "qwen2.5:3b"


def test_timeout_do_ollama_vai_por_client_kwargs():
    # ChatOllama 1.1 nao tem `timeout`; o httpx recebe por client_kwargs.
    assert llm.cliente("ollama", "m", 900).client_kwargs["timeout"] == 900


def test_provedor_desconhecido_falha_alto():
    config.PROVEDOR = "gemini"
    with pytest.raises(llm.ProvedorDesconhecido, match="gemini"):
        llm.construir(RegraDeteccao, "deteccao")


def test_provedor_desconhecido_nao_cai_para_openai(monkeypatch):
    """A falha e' o ponto: cair para openai mandaria dado para fora da rede."""
    config.PROVEDOR = "vazio"
    chamou = []
    monkeypatch.setattr(llm, "ChatOpenAI", lambda **k: chamou.append(k))
    with pytest.raises(llm.ProvedorDesconhecido):
        llm.construir(RegraDeteccao, "deteccao")
    assert chamou == [], "construiu um cliente OpenAI apesar do provedor invalido"


def test_modelo_do_papel_vence_o_geral():
    config.MODELOS_POR_PAPEL = {"deteccao": "qwen2.5-coder:14b"}
    assert llm.modelo_do_papel("deteccao") == "qwen2.5-coder:14b"


def test_sem_modelo_do_papel_cai_no_geral():
    config.MODELOS_POR_PAPEL = {"codigo": "outro"}
    assert llm.modelo_do_papel("deteccao") == "gpt-4o-mini"


def test_modelo_explicito_vence_o_do_papel():
    config.MODELOS_POR_PAPEL = {"deteccao": "do-papel"}
    registrado = {}
    original = llm.cliente

    def _espiao(provedor, modelo, segundos):
        registrado["modelo"] = modelo
        return original(provedor, modelo, segundos)

    llm.cliente = _espiao
    try:
        llm.construir(RegraDeteccao, "deteccao", modelo="explicito")
    finally:
        llm.cliente = original
    assert registrado["modelo"] == "explicito"


def test_timeout_muda_com_o_provedor():
    assert llm.timeout_do_provedor("openai") == 180
    assert llm.timeout_do_provedor("ollama") == 900


def test_timeout_llm_explicito_vence_o_default(monkeypatch):
    monkeypatch.setenv("TIMEOUT_LLM", "42")
    assert llm.timeout_do_provedor("ollama") == 42
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_llm.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'limpeza.llm'`

- [ ] **Step 4: Acrescentar a configuração**

Em `limpeza/config.py`, depois da linha `MODELO_LLM = os.getenv(...)`:

```python
PROVEDOR = os.getenv("PROVEDOR", "openai")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Sobrepoe MODELO_LLM num papel so'; vazio significa "usa o geral".
MODELOS_POR_PAPEL = {
    papel: os.getenv(f"MODELO_{papel.upper()}")
    for papel in ("deteccao", "especificador", "codigo", "fd")
}
MODELOS_POR_PAPEL = {p: m for p, m in MODELOS_POR_PAPEL.items() if m}

# Ollama local em CPU ja levou 808s numa chamada; 180 derrubaria o run.
TIMEOUT_POR_PROVEDOR = {"openai": 180, "ollama": 900}
```

Manter `TIMEOUT_LLM` como está — ele passa a ser a sobreposição explícita.

- [ ] **Step 5: Escrever a fábrica**

Criar `limpeza/llm.py`:

```python
"""O unico lugar do repositorio que sabe qual provedor de LLM existe."""
import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from . import config


class ProvedorDesconhecido(ValueError):
    """PROVEDOR com valor que a fabrica nao sabe construir."""


def modelo_do_papel(papel: str) -> str:
    """Modelo especifico do papel, senao o geral."""
    return config.MODELOS_POR_PAPEL.get(papel) or config.MODELO_LLM


def timeout_do_provedor(provedor: str) -> int:
    """Timeout explicito de TIMEOUT_LLM, senao o default do provedor."""
    if os.getenv("TIMEOUT_LLM"):
        return int(os.getenv("TIMEOUT_LLM"))
    return config.TIMEOUT_POR_PROVEDOR.get(provedor, config.TIMEOUT_LLM)


def cliente(provedor: str, modelo: str, segundos: int):
    """Constroi o cliente cru do provedor, sem amarrar schema."""
    if provedor == "openai":
        return ChatOpenAI(model=modelo, temperature=config.TEMPERATURA,
                          timeout=segundos, max_retries=2)
    if provedor == "ollama":
        # ChatOllama 1.1 nao tem timeout nem max_retries; o httpx recebe por aqui.
        return ChatOllama(model=modelo, base_url=config.OLLAMA_URL,
                          temperature=config.TEMPERATURA,
                          client_kwargs={"timeout": segundos})
    raise ProvedorDesconhecido(
        f"PROVEDOR={provedor!r} nao e' suportado. Use 'openai' ou 'ollama'. "
        "Nao caio para openai em silencio: isso mandaria dado para fora da rede.")


def construir(schema, papel: str, modelo: str | None = None):
    """Devolve um agente que preenche `schema`, no provedor configurado."""
    provedor = config.PROVEDOR.strip().lower()
    escolhido = modelo or modelo_do_papel(papel)
    return cliente(provedor, escolhido,
                   timeout_do_provedor(provedor)).with_structured_output(schema)
```

- [ ] **Step 6: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_llm.py -v`
Expected: PASS, 10 testes

- [ ] **Step 7: Confirmar que o invariante não se mexeu**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: 99 passed (89 + 10 novos)

- [ ] **Step 8: Commit**

```bash
git add limpeza/llm.py limpeza/config.py tests/test_llm.py requirements.txt
git commit -m "Fabrica de provedores: OpenAI e Ollama atras de um seletor"
```

---

### Task 2: Os quatro construtores passam a usar a fábrica

**Files:**
- Modify: `limpeza/deteccao/regra.py`, `limpeza/correcao/regras.py` (dois), `limpeza/correcao/fd.py`

**Interfaces:**
- Consumes: `limpeza.llm.construir(schema, papel)` da Task 1
- Produces: as assinaturas públicas dos quatro construtores **não mudam** — `construir_agente(modelo=None)`, `construir_agente_especificador(modelo=None)`, `construir_agente_codigo(modelo=None)`, e o `construir_agente(modelo=None)` de `fd.py`

- [ ] **Step 1: Rodar o invariante antes de mexer**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: 99 passed

- [ ] **Step 2: Trocar os quatro corpos**

Os quatro são idênticos exceto pelo schema. Cada um perde o `ChatOpenAI(...)` e o import de `langchain_openai`, e ganha `from .. import llm` (ou `from . import llm` conforme a profundidade).

`limpeza/deteccao/regra.py`:

```python
def construir_agente(modelo: str | None = None):
    """Agente que devolve a regra de deteccao de uma coluna."""
    return llm.construir(RegraDeteccao, "deteccao", modelo=modelo)
```

`limpeza/correcao/regras.py`:

```python
def construir_agente_especificador(modelo: str | None = None):
    """Agente 1: especifica a regra de correcao a partir dos rotulados."""
    return llm.construir(RegraCorrecao, "especificador", modelo=modelo)


def construir_agente_codigo(modelo: str | None = None):
    """Agente 2: traduz a especificacao JSON em `corrigir(valor)`."""
    return llm.construir(CodigoGerado, "codigo", modelo=modelo)
```

`limpeza/correcao/fd.py`:

```python
def construir_agente(modelo: str | None = None):
    """Agente que propoe a dependencia funcional de uma coluna."""
    return llm.construir(DependenciaFuncional, "fd", modelo=modelo)
```

> **Atenção ao comportamento que NÃO pode mudar:** `pipeline._construir_agentes`
> chama os quatro passando `config.MODELO_LLM` explicitamente. A fábrica já
> trata isso — `construir(schema, papel, modelo=...)` faz o explícito vencer a
> resolução por papel. Não é preciso alterar a assinatura de nada: a Task 1 já
> a definiu na forma final.

- [ ] **Step 3: Confirmar que ninguém mais fala com o provedor direto**

Run: `git grep -n "ChatOpenAI\|ChatOllama\|langchain_openai\|langchain_ollama" -- "limpeza/"`
Expected: só `limpeza/llm.py`

- [ ] **Step 4: Rodar tudo**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: 99 passed, e `test_invariante.py` intacto

- [ ] **Step 5: Commit**

```bash
git add limpeza/
git commit -m "Os quatro construtores de agente passam pela fabrica"
```

---

### Task 3: `escolher_modelo.py` — a sondagem versionada

A escolha de modelo é a decisão central deste projeto e só se resolve no servidor de destino. Esta ferramenta responde "esse modelo serve?" em doze chamadas, antes de gastar um run inteiro.

**Files:**
- Create: `escolher_modelo.py`
- Test: `tests/test_escolher_modelo.py`

**Interfaces:**
- Consumes: `limpeza.llm.construir` da Task 1; os prompts e schemas do repositório
- Produces: `escolher_modelo.medir(modelo, repeticoes) -> dict` com `{papel: {"schema_ok": int, "portao_ok": int | None, "total": int}}`

- [ ] **Step 1: Escrever o teste que falha**

O teste NÃO chama LLM: injeta um agente dublê. Criar `tests/test_escolher_modelo.py`:

```python
"""A ferramenta de escolha de modelo, com agente duble (sem rede)."""
import escolher_modelo
from limpeza.esquemas import CodigoGerado


class _AgenteFalso:
    """Devolve sempre o mesmo objeto; nenhuma chamada sai da maquina."""

    def __init__(self, resposta):
        self._resposta = resposta

    def invoke(self, _args):
        if isinstance(self._resposta, Exception):
            raise self._resposta
        return self._resposta


def test_conta_schema_preenchido_e_portao_aceito(monkeypatch):
    bom = CodigoGerado(codigo="def corrigir(valor):\n    return valor.strip()\n",
                       nota_de_traducao="ok")
    monkeypatch.setattr(escolher_modelo, "_agente",
                        lambda schema, papel, modelo: _AgenteFalso(bom))
    r = escolher_modelo.medir("modelo-falso", repeticoes=2)
    assert r["codigo"]["schema_ok"] == 2
    assert r["codigo"]["portao_ok"] == 2


def test_codigo_que_o_portao_rejeita_e_contado_a_parte(monkeypatch):
    ruim = CodigoGerado(codigo="import re\ndef corrigir(valor):\n    return valor\n",
                        nota_de_traducao="com import")
    monkeypatch.setattr(escolher_modelo, "_agente",
                        lambda schema, papel, modelo: _AgenteFalso(ruim))
    r = escolher_modelo.medir("modelo-falso", repeticoes=2)
    assert r["codigo"]["schema_ok"] == 2, "o schema foi preenchido"
    assert r["codigo"]["portao_ok"] == 0, "mas o portao AST rejeitou"


def test_excecao_nao_derruba_a_medicao(monkeypatch):
    monkeypatch.setattr(escolher_modelo, "_agente",
                        lambda schema, papel, modelo: _AgenteFalso(RuntimeError("boom")))
    r = escolher_modelo.medir("modelo-falso", repeticoes=2)
    assert r["codigo"]["schema_ok"] == 0
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_escolher_modelo.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'escolher_modelo'`

- [ ] **Step 3: Escrever a ferramenta**

Criar `escolher_modelo.py` na raiz, ao lado de `avaliar_limpador.py`:

```python
"""Mede se um modelo serve para a POC, antes de gastar um run inteiro."""
import argparse
import time

from langchain_core.prompts import ChatPromptTemplate

from limpeza import llm, sandbox
from limpeza.correcao import fd as fd_mod
from limpeza.correcao import regras
from limpeza.deteccao import regra as regra_mod
from limpeza.esquemas import (CodigoGerado, DependenciaFuncional, RegraCorrecao,
                              RegraDeteccao)

# Amostras fixas do beers: o que o KMeans mostraria para `abv`.
_SUJOS = """  - "0.075"   (12x na coluna)
  - "0.05%"   (8x na coluna)
  - "0.061"   (15x na coluna)
  - ""   (5x na coluna)"""

_ROTULADOS = """  - sujo: "0.05%"  ->  correto: "0.05"   (8x na coluna)
  - sujo: "0.075"  ->  correto: "0.075"   (12x na coluna)"""

_SPEC = """{"coluna": "abv", "erro_detectado": true, "tipo_erro": "formato",
 "descricao_padrao": "algumas celulas trazem o sinal de porcentagem no fim",
 "condicao_regex": "%$",
 "transformacao": {"tipo": "regex_sub", "padrao": "%$", "substituicao": ""},
 "exemplos": [{"de": "0.05%", "para": "0.05"}], "confianca": 0.95}"""

# (schema, sistema, humano, argumentos, alvo do portao AST ou None)
CASOS = {
    "deteccao": (RegraDeteccao, regra_mod.SISTEMA, regra_mod.HUMANO,
                 {"coluna": "abv", "amostra": _SUJOS,
                  "total_linhas": 300, "total_distintos": 132},
                 dict(nome_funcao="detectar", nome_argumento="col", series_mode=True)),
    "especificador": (RegraCorrecao, regras.SISTEMA_ESPECIFICADOR, regras.HUMANO_BUDGET,
                      {"coluna": "abv", "amostra": _ROTULADOS,
                       "total_linhas": 300, "total_distintos": 132},
                      None),
    "codigo": (CodigoGerado, regras.SISTEMA_CODIGO, regras.HUMANO_CODIGO,
               {"spec": _SPEC},
               dict(nome_funcao="corrigir", nome_argumento="valor")),
    "fd": (DependenciaFuncional, fd_mod.SISTEMA, fd_mod.HUMANO,
           {"dependente": "state", "candidatos": "brewery-name, city",
            "exemplos": '  - `brewery-name`="21st Amendment Brewery" -> `state`="CA"'},
           None),
}


def _agente(schema, papel, modelo):
    """Ponto unico de construcao; o teste substitui esta funcao."""
    sistema, humano = CASOS[papel][1], CASOS[papel][2]
    prompt = ChatPromptTemplate.from_messages([("system", sistema), ("human", humano)])
    return prompt | llm.construir(schema, papel, modelo=modelo)


def medir(modelo: str, repeticoes: int = 3) -> dict:
    """Invoca cada papel `repeticoes` vezes e conta schema e portao."""
    placar = {}
    for papel, (schema, _s, _h, args, alvo) in CASOS.items():
        schema_ok = portao_ok = 0
        for _ in range(repeticoes):
            try:
                obj = _agente(schema, papel, modelo).invoke(args)
            except Exception:  # falha de parse ou de rede conta como nao preenchido
                continue
            if obj is None:
                continue
            schema_ok += 1
            if alvo is None:
                continue
            try:
                sandbox.validar(obj.codigo, **alvo)
                portao_ok += 1
            except Exception:  # codigo rejeitado pelo portao: conta a parte
                pass
        placar[papel] = {"schema_ok": schema_ok, "total": repeticoes,
                         "portao_ok": None if alvo is None else portao_ok}
    return placar


def _imprimir(modelo: str, placar: dict, segundos: float) -> None:
    """Uma linha por papel, mais o criterio de aprovacao."""
    print(f"\nmodelo: {modelo}   ({segundos:.0f}s)\n")
    print(f"  {'papel':16s}{'schema':>10s}{'portao AST':>14s}")
    for papel, r in placar.items():
        portao = "-" if r["portao_ok"] is None else f"{r['portao_ok']}/{r['total']}"
        print(f"  {papel:16s}{r['schema_ok']:>6d}/{r['total']:<3d}{portao:>14s}")
    det = placar["deteccao"]
    if det["portao_ok"]:
        print("\n  Serve: a deteccao gera codigo que o portao aceita.")
    else:
        print("\n  NAO serve: sem deteccao aceita pelo portao, o limpador sai vazio.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Mede se um modelo consegue preencher os schemas da POC")
    ap.add_argument("--modelo", required=True, help="nome do modelo no provedor ativo")
    ap.add_argument("--repeticoes", type=int, default=3)
    args = ap.parse_args(argv)

    t0 = time.time()
    placar = medir(args.modelo, args.repeticoes)
    _imprimir(args.modelo, placar, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**CUIDADO com os nomes das variáveis dos prompts** — errar um faz o caso falhar por `KeyError` e parecer culpa do modelo. Os nomes acima foram conferidos contra o código, mas confirme antes de rodar:

Run: `git grep -oE "\{[a-z_]+\}" -- limpeza/deteccao/regra.py limpeza/correcao/regras.py limpeza/correcao/fd.py | sort -u`

O `HUMANO_CODIGO` usa `{spec}`, **não** `{especificacao}`.

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_escolher_modelo.py -v`
Expected: PASS, 3 testes

- [ ] **Step 5: Conferir que a CLI responde**

Run: `.venv/Scripts/python escolher_modelo.py --help`
Expected: mostra `--modelo` e `--repeticoes`

- [ ] **Step 6: Rodar tudo**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: 102 passed

- [ ] **Step 7: Commit**

```bash
git add escolher_modelo.py tests/test_escolher_modelo.py
git commit -m "escolher_modelo.py: mede se um modelo serve antes de gastar um run"
```

---

### Task 4: `SUJO`/`LIMPO` como default dos argumentos

Na interface do Portainer, editar variável de ambiente é fácil e editar o comando é desconfortável. Esta mudança é **aditiva**: a linha de comando continua sendo a interface.

**Files:**
- Modify: `main.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces: `main.main(argv)` aceita `argv` sem `--sujo`/`--limpo` quando `SUJO`/`LIMPO` estão no ambiente

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar a `tests/test_pipeline.py`:

```python
import main as cli


def test_sujo_e_limpo_saem_do_ambiente(monkeypatch):
    monkeypatch.setenv("SUJO", "/dados/a_dirty.csv")
    monkeypatch.setenv("LIMPO", "/dados/a_clean.csv")
    args = cli._argumentos([])
    assert args.sujo == "/dados/a_dirty.csv"
    assert args.limpo == "/dados/a_clean.csv"


def test_argumento_vence_a_variavel(monkeypatch):
    monkeypatch.setenv("SUJO", "/dados/do_ambiente.csv")
    args = cli._argumentos(["--sujo", "/dados/do_argumento.csv",
                            "--limpo", "/dados/b.csv"])
    assert args.sujo == "/dados/do_argumento.csv"


def test_sem_argumento_e_sem_variavel_e_erro(monkeypatch):
    monkeypatch.delenv("SUJO", raising=False)
    monkeypatch.delenv("LIMPO", raising=False)
    with pytest.raises(SystemExit):
        cli._argumentos([])
```

Confirmar que `import pytest` já existe no topo do arquivo; se não, acrescentar.

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -k ambiente -v`
Expected: FAIL — hoje `--sujo` é `required=True` e `_argumentos([])` sai com `SystemExit` mesmo com a variável presente

- [ ] **Step 3: Tornar os argumentos opcionais quando o ambiente supre**

Em `main.py`, trocar as duas linhas:

```python
ap.add_argument("--sujo", default=os.getenv("SUJO"),
                help="CSV com os dados sujos (ou a variavel SUJO)")
ap.add_argument("--limpo", default=os.getenv("LIMPO"),
                help="CSV de referencia (ou a variavel LIMPO)")
```

E, depois de `args = ap.parse_args(argv)`, antes do `return`:

```python
    faltando = [n for n in ("sujo", "limpo") if not getattr(args, n)]
    if faltando:
        ap.error(f"faltam --{' e --'.join(faltando)} (ou as variaveis "
                 f"{' e '.join(n.upper() for n in faltando)})")
```

Acrescentar `import os` ao topo.

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -k "ambiente or argumento or variavel" -v`
Expected: PASS

- [ ] **Step 5: Confirmar que a CLI antiga não mudou**

Run: `.venv/Scripts/python main.py --help`
Expected: `--sujo` e `--limpo` aparecem, agora mencionando as variáveis

- [ ] **Step 6: Rodar tudo**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: 105 passed

- [ ] **Step 7: Commit**

```bash
git add main.py tests/test_pipeline.py
git commit -m "SUJO/LIMPO como default de --sujo/--limpo, para o Portainer"
```

---

### Task 5: A imagem

**Files:**
- Create: `Dockerfile`, `.dockerignore`

- [ ] **Step 1: Escrever o `.dockerignore`**

Sem ele, o contexto de build sobe `.venv/` (centenas de MB) e `runs/`.

```
.venv/
.git/
runs/
__pycache__/
**/__pycache__/
*.pyc
.pytest_cache/
.env
docs/
.superpowers/
```

- [ ] **Step 2: Escrever o `Dockerfile`**

O MiniLM vai **assado**: sob restrição de privacidade, uma imagem que baixa modelo em runtime faz chamada externa a cada start. São 43,8 MB — só `onnx/model_O4.onnx` e `tokenizer.json`; o resto da pasta é formato PyTorch que a POC não usa.

```dockerfile
FROM python:3.12-slim AS base

WORKDIR /app

# Camada de dependencias separada: muda pouco, cacheia bem.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# So' os dois arquivos que o Embedder abre.
COPY all-MiniLM-L6-v2/onnx/model_O4.onnx  /modelo/onnx/model_O4.onnx
COPY all-MiniLM-L6-v2/tokenizer.json      /modelo/tokenizer.json
ENV MODELO_EMBEDDING=/modelo

COPY limpeza/ ./limpeza/
COPY main.py avaliar_limpador.py escolher_modelo.py ./

ENTRYPOINT ["python", "main.py"]
```

**O modelo precisa estar no contexto de build.** O repositório não versiona `all-MiniLM-L6-v2/`. Documentar no README que, antes do build, é preciso copiar os dois arquivos para essa pasta na raiz — e acrescentá-la ao `.gitignore` se ainda não estiver.

- [ ] **Step 3: Preparar o modelo no contexto e buildar**

```bash
mkdir -p all-MiniLM-L6-v2/onnx
cp "$MODELO_EMBEDDING/onnx/model_O4.onnx" all-MiniLM-L6-v2/onnx/
cp "$MODELO_EMBEDDING/tokenizer.json"     all-MiniLM-L6-v2/
docker build -t limpeza-poc .
```

- [ ] **Step 4: Conferir o tamanho e que o entrypoint responde**

Run: `docker images limpeza-poc --format "{{.Size}}"`
Expected: em torno de 500 MB; se passar de 1 GB, o `.dockerignore` não pegou

Run: `docker run --rm limpeza-poc --help`
Expected: o mesmo `--help` da CLI local

- [ ] **Step 5: Confirmar que o modelo assado funciona**

Run: `docker run --rm --entrypoint python limpeza-poc -c "from limpeza.amostragem import Embedder; m = Embedder().codificar(['12.0 oz','12.0','abacaxi']); print('similar', float(m[0] @ m[1]), 'distante', float(m[0] @ m[2]))"`
Expected: `similar` maior que `distante` — os mesmos ~0,54 contra ~0,18 de `tests/test_embeddings.py`

- [ ] **Step 6: Acrescentar a pasta do modelo ao `.gitignore`**

```
all-MiniLM-L6-v2/
```

- [ ] **Step 7: Commit**

```bash
git add Dockerfile .dockerignore .gitignore
git commit -m "Imagem da POC com o MiniLM assado dentro"
```

---

### Task 6: O stack do Portainer

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: Escrever o compose**

```yaml
services:
  ollama:
    image: ollama/ollama
    volumes:
      - ollama-modelos:/root/.ollama
    restart: unless-stopped
    # Acesso a GPU depende de configuracao no host, feita por quem administra.
    # deploy:
    #   resources:
    #     reservations:
    #       devices: [{capabilities: [gpu]}]

  arquivos:
    image: filebrowser/filebrowser
    volumes:
      - limpeza-dados:/srv
    ports:
      - "8080:80"
    restart: unless-stopped

  poc:
    build: .
    environment:
      PROVEDOR: ollama
      OLLAMA_URL: http://ollama:11434
      MODELO_LLM: qwen2.5-coder:14b
      SUJO: /dados/entrada_dirty.csv
      LIMPO: /dados/entrada_clean.csv
      SAIDA: /dados/runs
    volumes:
      - limpeza-dados:/dados
    depends_on: [ollama]
    restart: "no"

volumes:
  ollama-modelos:
  limpeza-dados:
```

- [ ] **Step 2: Fazer o `--saida` também ler do ambiente**

O compose define `SAIDA`, então `main.py` precisa respeitá-la, pela mesma razão das outras duas. Em `main.py`:

```python
ap.add_argument("--saida", default=os.getenv("SAIDA", "runs"),
                help="pasta base onde o run e' escrito (ou a variavel SAIDA)")
```

Acrescentar a `tests/test_pipeline.py`:

```python
def test_saida_sai_do_ambiente(monkeypatch):
    monkeypatch.setenv("SUJO", "/d/a.csv")
    monkeypatch.setenv("LIMPO", "/d/b.csv")
    monkeypatch.setenv("SAIDA", "/dados/runs")
    assert cli._argumentos([]).saida == "/dados/runs"
```

- [ ] **Step 3: Validar o compose sem subir nada**

Run: `docker compose config`
Expected: imprime o YAML resolvido, sem erro de sintaxe

- [ ] **Step 4: Subir só o Ollama e confirmar que a POC o alcança pela rede do stack**

```bash
docker compose up -d ollama
docker compose run --rm --entrypoint python poc -c "import urllib.request, os; print(urllib.request.urlopen(os.environ['OLLAMA_URL'] + '/api/tags', timeout=10).status)"
```

Expected: `200` — prova que o nome `ollama` resolve dentro do stack

- [ ] **Step 5: Derrubar**

Run: `docker compose down`

- [ ] **Step 6: Rodar os testes**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: 106 passed

- [ ] **Step 7: Commit**

```bash
git add docker-compose.yml main.py tests/test_pipeline.py
git commit -m "Stack do Portainer: ollama, arquivos e a POC como job one-shot"
```

---

### Task 7: Documentação

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `CHANGELOG.md`, `.env.example`

- [ ] **Step 1: `.env.example`**

Acrescentar, depois de `MODELO_LLM`:

```
# openai (default) ou ollama. Valor desconhecido FALHA -- nunca cai para openai
# em silencio, porque isso mandaria dado para fora da rede.
PROVEDOR=openai
OLLAMA_URL=http://localhost:11434

# Sobrepoe MODELO_LLM num papel so'. Vazio = usa o geral.
# MODELO_DETECCAO=qwen2.5-coder:14b
# MODELO_ESPECIFICADOR=
# MODELO_CODIGO=
# MODELO_FD=

# Caminhos, para quando a POC roda em container (o Portainer edita isto).
# SUJO=/dados/entrada_dirty.csv
# LIMPO=/dados/entrada_clean.csv
# SAIDA=/dados/runs
```

- [ ] **Step 2: `CLAUDE.md` — seção nova sobre provedores**

Acrescentar entre a seção 4 (Os tipos) e a 5 (Invariante), renumerando as seguintes. Cobrir:

- `limpeza/llm.py` é o **único** lugar que sabe que existe mais de um provedor; os quatro construtores só chamam a fábrica. Não acrescentar um segundo lugar.
- A precedência de modelo: explícito → papel → `MODELO_LLM`.
- Que `ChatOllama` 1.1 **não tem** `timeout` nem `max_retries`; timeout vai por `client_kwargs`, retry não existe e o pipeline degrada sozinho.
- Que `PROVEDOR` desconhecido **falha alto**, e por quê — não é rigor estético, é a restrição de privacidade.
- `escolher_modelo.py`: como se usa e qual é o critério de aprovação (**a linha da detecção**; modelo que não gera `detectar(col)` aceito pelo portão produz limpador vazio).

Atualizar a seção 2 (Como rodar) com o caminho em container.

- [ ] **Step 3: `README.md`**

Acrescentar uma seção "Rodando com modelo local" e outra "Em container / Portainer", cobrindo: as variáveis, o ciclo de trabalho no Portainer (subir CSVs → editar `SUJO`/`LIMPO` → start → log → baixar artefatos), que o build precisa dos dois arquivos do MiniLM no contexto, e o comportamento contra-intuitivo do job one-shot — **no primeiro deploy o serviço `poc` executa uma vez**, e se os CSVs ainda não estiverem no volume ele falha com `DadosInvalidos`, o que é inofensivo mas assusta.

- [ ] **Step 4: `CHANGELOG.md`**

Entrada sob `## [Não publicado]`, com `### Adicionado`: seletor de provedor, `escolher_modelo.py`, imagem e stack. Citar o resultado da sondagem que autorizou o trabalho (qwen2.5:3b: detecção 3/3 no schema e 0/3 no portão; código 3/3 e 3/3).

- [ ] **Step 5: Rodar cada comando que o README apresenta**

README que documenta comando quebrado é pior que README ausente. **Exceção:** não rodar `main.py` sem `--help` nem `docker compose run poc` de verdade — gastam LLM.

- [ ] **Step 6: Verificação final**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: 106 passed, invariante intacto

Run: `.venv/Scripts/python tests/medir_verbosidade.py | tail -1`
Expected: densidade em torno de 11%; se passar de 13%, podar

Run: `git grep -n "ChatOpenAI\|ChatOllama" -- "limpeza/"`
Expected: só `limpeza/llm.py`

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md README.md CHANGELOG.md .env.example
git commit -m "Documenta o seletor de provedor e a implantacao em container"
```

---

## Nota sobre custo de API e de download

Nenhuma tarefa deste plano gasta chamadas de LLM pagas: os testes usam dublês, e o `escolher_modelo.py` só é exercitado com agente falso. A Task 6, Step 4 sobe o container do Ollama (imagem de **8,45 GB**) — se ele já não estiver na máquina, o download é demorado; é o único custo de tempo relevante do plano.

Validar a qualidade de um modelo real é o passo seguinte ao plano, no servidor de destino, com `escolher_modelo.py`.
