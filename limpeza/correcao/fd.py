"""Etapa 4, camada 2 da cascata: propoe e aplica uma dependencia funcional `A -> B`."""
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics import mutual_info_score

from .. import config, llm
from ..esquemas import DependenciaFuncional


def calc_mi(df, alvo: str) -> dict:
    # Devolve {coluna: MI normalizada 1 casa}; `alvo` tambem aparece no resultado.
    def mi_par(col: str) -> float:
        if col == alvo:
            return float(mutual_info_score(df[alvo], df[col]))
        # pre-filtro: pares (alvo, col) que ocorrem ao menos 2x
        contagem = df.groupby([alvo, col])[alvo].transform("size")
        filtrado = df[contagem >= 2]
        if filtrado.empty:
            return 0.0
        a = filtrado[alvo].reset_index(drop=True)
        b = filtrado[col].reset_index(drop=True)
        return float(mutual_info_score(a, b))

    brutos = {col: mi_par(col) for col in df.columns}
    maximo = max(brutos.values()) if brutos else 0.0
    if maximo == 0.0:
        return {col: 0.0 for col in brutos}
    return {col: round(v / maximo, 1) for col, v in brutos.items()}


def candidatos_determinantes(df, alvo: str, limiar: float | None = None) -> list[str]:
    limiar = config.MI_THRESHOLD if limiar is None else limiar
    mi = calc_mi(df, alvo)
    return [col for col, v in mi.items() if col != alvo and v >= limiar]


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
    """Agente que propoe a dependencia funcional de uma coluna."""
    return llm.construir(DependenciaFuncional, "fd", modelo=modelo)


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


def moda_condicionada(df, mascara, determinante, dependente, valor_det, excluir=None):
    # Duplo filtro: so' linhas com deteccao==0 nas duas colunas entram na moda.
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
    """Gate de 100% no conjunto rotulado, incluindo os limpos. Ver docs/DECISOES.md#gates-de-100."""
    colunas = _colunas_validas(fd, df)
    if colunas is None:
        return False
    determinante, dependente = colunas

    for r in rotulados:
        idx = r["indice"]
        valor_det = df.at[idx, determinante]
        excluir = idx if not r["eh_erro"] else None
        corrigido = moda_condicionada(
            df, mascara, determinante, dependente, valor_det, excluir=excluir
        )
        if corrigido is None:
            corrigido = df.at[idx, dependente]  # sem pool: mantem o sujo
        if corrigido != r["limpo"]:
            return False
    return True


def aplicar_fd(fd: DependenciaFuncional, df, indices_marcados, mascara, dependente: str) -> dict:
    """Aplica a FD nas celulas marcadas com duplo filtro deteccao==0; devolve so' o que mudou."""
    colunas = _colunas_validas(fd, df)
    if colunas is None:
        return {}
    determinante, dependente = colunas

    mudancas = {}
    for idx in indices_marcados:
        valor_det = df.at[idx, determinante]
        corrigido = moda_condicionada(df, mascara, determinante, dependente, valor_det)
        if corrigido is None:
            continue  # sem pool: deixa escalar
        atual = df.at[idx, dependente]
        if corrigido != atual:
            mudancas[idx] = corrigido
    return mudancas
