"""Camada 2 da cascata: dependencia funcional `A -> B`.

Porta correction.py:808-866 do ZeroDC, com duas simplificacoes DECLARADAS:
  - determinante UNICO (o ZeroDC aceita composto, correction.py:812);
  - o candidato a determinante vem da MI normalizada (contexto.calc_mi), nao de
    um retriever.

O gate (validar_fd) exige 100% no conjunto rotulado, INCLUINDO os negativos
limpos -- uma FD que conserta o sujo mas mexeria num limpo e' reprovada. A
aplicacao usa o DUPLO FILTRO deteccao==0 nas duas colunas (determinante e
dependente), como correction.py:824-826,853-864: so' linhas cujas duas colunas
nao foram marcadas entram na moda.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from . import config
from .esquemas import DependenciaFuncional

SISTEMA = """Voce identifica uma DEPENDENCIA FUNCIONAL para reparar uma coluna suja.

Uma dependencia funcional `A -> B` diz: o valor da coluna A determina o valor \
correto da coluna B. Ex.: `brewery-name -> city` (a cervejaria fixa a cidade).

Voce recebe a coluna DEPENDENTE (a que tem erros), uma lista de colunas \
CANDIDATAS a determinante (pre-selecionadas por informacao mutua) e exemplos \
rotulados (tupla suja + valor errado + valor correto).

Escolha UM determinante entre as candidatas -- aquele que, dado seu valor, fixa \
o valor correto da dependente. Se nenhuma candidata determina a dependente, \
devolva `determinante: "None"`."""

HUMANO = """Coluna dependente (a corrigir): `{dependente}`
Colunas candidatas a determinante: {candidatos}

Exemplos rotulados:
{exemplos}

Escolha o determinante unico (ou "None")."""


def construir_agente(modelo: str | None = None):
    llm = ChatOpenAI(
        model=modelo or config.MODELO_LLM,
        temperature=config.TEMPERATURA,
        timeout=config.TIMEOUT_LLM,
        max_retries=2,
    )
    return llm.with_structured_output(DependenciaFuncional)


def _formatar_exemplos(rotulados: list[dict], df, dependente: str) -> str:
    linhas = []
    for r in rotulados:
        idx = r["indice"]
        tupla = {c: str(df.at[idx, c]) for c in df.columns}
        linhas.append(
            f"  - tupla_suja: {tupla}\n"
            f"    valor_errado ({dependente}): \"{r['sujo']}\"  ->  correto: \"{r['limpo']}\""
        )
    return "\n".join(linhas)


def propor_fd(
    dependente: str,
    rotulados: list[dict],
    candidatos_mi: list[str],
    df,
    agente=None,
) -> DependenciaFuncional:
    """Um passe de LLM. `dependente` = coluna a corrigir; candidatos vem da MI."""
    agente = agente or construir_agente()
    prompt = ChatPromptTemplate.from_messages([("system", SISTEMA), ("human", HUMANO)])
    fd: DependenciaFuncional = (prompt | agente).invoke(
        {
            "dependente": dependente,
            "candidatos": ", ".join(f"`{c}`" for c in candidatos_mi) or "(nenhuma)",
            "exemplos": _formatar_exemplos(rotulados, df, dependente),
        }
    )
    fd.dependente = dependente  # o nome e' fato, nao opiniao do modelo
    return fd


def _moda_condicionada(df, mascara, determinante, dependente, valor_det, excluir=None):
    """Moda de `dependente` entre linhas com `determinante == valor_det` e as
    DUAS colunas com deteccao == 0 (duplo filtro). `excluir` tira um indice do
    pool (usado para negativos limpos no gate). Devolve None se o pool e' vazio.
    """
    condicao = (
        (df[determinante] == valor_det)
        & (mascara[determinante] == 0)
        & (mascara[dependente] == 0)
    )
    if excluir is not None:
        condicao &= df.index != excluir
    valores = df.loc[condicao, dependente]
    if valores.empty:
        return None
    return valores.mode().iloc[0]


def _colunas_validas(fd: DependenciaFuncional, df) -> tuple[str, str] | None:
    det = (fd.determinante or "").strip()
    dep = (fd.dependente or "").strip()
    if not det or det == "None":
        return None
    if det not in df.columns or dep not in df.columns:
        return None
    return det, dep


def validar_fd(fd: DependenciaFuncional, rotulados: list[dict], df, mascara, dependente: str) -> bool:
    """Gate 100% no conjunto rotulado (sujos + limpos). Determinante ausente do
    df -> reprova sem lancar."""
    colunas = _colunas_validas(fd, df)
    if colunas is None:
        return False
    determinante, dependente = colunas

    for r in rotulados:
        idx = r["indice"]
        valor_det = df.at[idx, determinante]
        excluir = idx if not r["eh_erro"] else None
        corrigido = _moda_condicionada(
            df, mascara, determinante, dependente, valor_det, excluir=excluir
        )
        if corrigido is None:
            corrigido = df.at[idx, dependente]  # sem pool: mantem o sujo
        if corrigido != r["limpo"]:
            return False
    return True


def aplicar_fd(fd: DependenciaFuncional, df, indices_marcados, mascara, dependente: str) -> dict:
    """Aplica a FD nas celulas marcadas. Duplo filtro deteccao==0.

    Devolve {indice: novo_valor} SO' das celulas cujo valor mudou. FD invalida ou
    sem pool -> nenhuma mudanca.
    """
    colunas = _colunas_validas(fd, df)
    if colunas is None:
        return {}
    determinante, dependente = colunas

    mudancas = {}
    for idx in indices_marcados:
        valor_det = df.at[idx, determinante]
        corrigido = _moda_condicionada(df, mascara, determinante, dependente, valor_det)
        if corrigido is None:
            continue  # sem pool: deixa escalar
        atual = df.at[idx, dependente]
        if corrigido != atual:
            mudancas[idx] = corrigido
    return mudancas
