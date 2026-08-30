"""Camada 3 (ultimo recurso): correcao de UM valor distinto pelo LLM.

Sem gate -- e' o fim da linha. Duas travas herdadas da filosofia da amostragem e
do custo:
  - DEDUP por valor distinto: um valor sujo -> UMA chamada, resultado cacheado
    para todas as celulas iguais (nao paga a mesma correcao N vezes);
  - TETO `--limite-fallback` por coluna, contado por VALOR distinto enviado; ao
    atingir, as celulas restantes ficam SUJAS e sao logadas -- o custo nao escapa.

Contexto = tupla da linha FILTRADA por MI (so' colunas candidatas com deteccao
== 0 naquela linha), sem retriever vetorial (simplificacao declarada).

CONTRA o vies do ZeroDC: o prompt PERMITE valor vazio e PROIBE inventar. NAO se
replica correction.py:1162-1163 (que forca 'null' quando o modelo omite a chave)
nem o "evite NULL" do SystemMessage_EC.txt. Vazio e' resposta legitima.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from . import config
from .esquemas import CorrecaoCelula

SISTEMA = """Voce corrige UM valor de uma celula suja de tabela.

Recebe a coluna, o valor errado e a tupla da linha (apenas colunas confiaveis, \
filtradas por informacao mutua). Devolve o valor corrigido e a cadeia que leva \
ate ele.

REGRAS:
1. Use o contexto da tupla para inferir o valor correto quando ele estiver \
   determinado (ex.: a cidade dada a cervejaria).
2. Se NAO houver base confiavel para corrigir, devolva `valor` VAZIO (''). Isso \
   e' uma resposta legitima e preferivel a chutar.
3. E' PROIBIDO inventar um valor plausivel so' para preencher, e PROIBIDO \
   escrever 'null', 'N/A', '-' ou qualquer sentinela. Ou voce sabe o valor pelo \
   contexto, ou devolve vazio.
4. Nao acrescente unidades, sufixos nem formatacao que o dado limpo nao teria."""

HUMANO = """Coluna: `{coluna}`
Valor errado: "{valor}"
Tupla da linha (colunas confiaveis): {tupla}

Devolva a cadeia e o valor corrigido (vazio se nao houver base confiavel)."""


def construir_agente(modelo: str | None = None):
    llm = ChatOpenAI(
        model=modelo or config.MODELO_LLM,
        temperature=config.TEMPERATURA,
        timeout=config.TIMEOUT_LLM,
        max_retries=2,
    )
    return llm.with_structured_output(CorrecaoCelula)


def _tupla_filtrada(df, idx, coluna, colunas_contexto, mascara) -> dict:
    """Valores da linha `idx` nas colunas de contexto com deteccao == 0.

    Exclui a propria coluna alvo. Se a coluna de contexto esta marcada como erro
    naquela linha, ela e' omitida (nao contamina a correcao).
    """
    tupla = {}
    for col in colunas_contexto:
        if col == coluna:
            continue
        if int(mascara.at[idx, col]) == 0:
            tupla[col] = str(df.at[idx, col])
    return tupla


def corrigir_valor(tupla_filtrada: dict, valor: str, coluna: str, agente=None) -> CorrecaoCelula:
    """Uma chamada de LLM para um valor distinto. Pode devolver valor vazio."""
    agente = agente or construir_agente()
    prompt = ChatPromptTemplate.from_messages([("system", SISTEMA), ("human", HUMANO)])
    return (prompt | agente).invoke(
        {"coluna": coluna, "valor": valor, "tupla": tupla_filtrada or "(sem contexto confiavel)"}
    )


def aplicar_fallback(
    coluna: str,
    df,
    indices_marcados,
    mascara,
    colunas_contexto: list[str],
    agente=None,
    limite: int | None = None,
) -> dict:
    """Laco com dedup + cache + teto. Devolve dict com:

      correcoes   -> {indice: novo_valor} das celulas ENVIADAS (dentro do teto);
                     novo_valor pode ser '' (vazio legitimo).
      nao_enviados-> [indices] alem do teto, que ficam sujos.
      log         -> registros dos valores nao enviados (teto atingido).
      chamadas    -> numero de valores distintos efetivamente enviados ao LLM.
    """
    limite = config.LIMITE_FALLBACK if limite is None else limite
    indices_marcados = list(indices_marcados)

    # Agrupa indices por valor sujo distinto, preservando ordem de aparicao.
    por_valor: dict = {}
    for idx in indices_marcados:
        por_valor.setdefault(df.at[idx, coluna], []).append(idx)

    correcoes: dict = {}
    nao_enviados: list = []
    log: list = []
    chamadas = 0

    for valor, indices in por_valor.items():
        if chamadas >= limite:
            nao_enviados.extend(indices)
            log.append({"coluna": coluna, "valor": valor, "celulas": len(indices),
                        "motivo": f"teto de fallback ({limite}) atingido"})
            continue
        idx0 = indices[0]
        tupla = _tupla_filtrada(df, idx0, coluna, colunas_contexto, mascara)
        try:
            resposta = corrigir_valor(tupla, valor, coluna, agente=agente)
            novo = "" if resposta.valor is None else str(resposta.valor)
        except Exception as exc:  # noqa: BLE001 -- fallback nao pode derrubar o pipeline
            log.append({"coluna": coluna, "valor": valor, "celulas": len(indices),
                        "motivo": f"excecao {type(exc).__name__}: {exc}"})
            nao_enviados.extend(indices)
            continue
        chamadas += 1
        for idx in indices:
            correcoes[idx] = novo

    return {"correcoes": correcoes, "nao_enviados": nao_enviados, "log": log, "chamadas": chamadas}
