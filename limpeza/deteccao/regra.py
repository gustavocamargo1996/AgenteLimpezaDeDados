"""Etapa 3: gera a regra de deteccao intra-coluna `detectar(col) -> pd.Series[bool]`."""
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from .. import config, llm, sandbox
from ..esquemas import RegraDeteccao
from ..tipos import Detector

SISTEMA = """Voce escreve regras de DETECCAO de erro para uma coluna de tabela suja.

A regra e' uma funcao `detectar(col) -> pd.Series[bool]`: recebe a COLUNA inteira \
(`col`, uma pd.Series de strings) e devolve uma pd.Series booleana do MESMO \
tamanho -- True na linha corrompida, False na intacta. Ela trabalha a coluna \
elemento a elemento (vetorizado), NAO ve outras colunas.

Sua saida tem a CADEIA (como voce chegou ao criterio, citando valores concretos \
dos representantes) e a REGRA em JSON com o campo `codigo`.

IDIOMA (use este padrao vetorizado):
  - marcador literal ('%', 'oz', 'N/A'): \
`col.astype(str).str.contains('%', regex=False, na=False)`;
  - padrao estrutural (sufixo de 2 letras, barra): \
`col.astype(str).str.contains(r' [A-Z]{{2}}$', regex=True, na=False)`;
  - combinar marcadores com OR dentro da MESMA funcao: \
`col.str.contains('oz', na=False) | col.str.contains('ounce', na=False)`;
  - sentinela exata: `col.astype(str).eq('null')`; vazio: `col.astype(str).eq('')`.
Sempre `na=False` (celula ausente NAO e' marcada por engano).

CONTRATO RIGIDO do `codigo` -- ele passa por validacao AST antes de rodar:
  - exatamente UMA funcao, nada mais (sem import: o modulo `re` JA existe no escopo);
  - a funcao se chama exatamente `detectar` e recebe um unico argumento `col`;
  - devolve uma pd.Series[bool] do tamanho de `col` (True = erro, False = intacto);
  - so' metodos elementwise de `col` (str.contains/startswith/endswith/match, \
astype, eq/ne, isin, isna/notna, fillna, strip, lower, upper). PROIBIDO usar \
operacoes cross-row (duplicated, mode, groupby, shift, value_counts) e \
ler/escrever arquivo. Nada de eval, exec, open, getattr, globals, __import__.

DISCIPLINA:
1. So' marque como erro o que a amostra sustenta. Se os representantes nao mostram
   corrupcao, use `erro_provavel: false` e um `detectar` que devolve tudo False
   (ex.: `col.isin([])`).
2. Detecte pela FORMA/MARCADOR compartilhado (sufixo estranho, sentinela como
   'N/A', vazio, caractere fora do padrao), NAO enumere os valores observados:
   nada de `col.isin(['0.061%', '0.125%'])` nem blacklist -- isso e' memorizar e
   falha em todo valor-erro novo.
3. Prefira nao marcar a marcar demais: um falso positivo vira dano na correcao.

Escreva a cadeia em portugues do Brasil."""

HUMANO = """Coluna: `{coluna}`
Tabela: {total_linhas} linhas, {total_distintos} valores distintos nesta coluna.

Estes sao os valores representativos SUJOS (voce NAO tem o valor correto), \
escolhidos por agrupamento semantico:

{amostra}

Escreva a cadeia e o `codigo` de `detectar(col)`."""

# Sentinela de fallback: nao marca nada. Usa so' `isin`, permitido pela
# allowlist do modo serie (ver docs/DECISOES.md#modo-serie).
DETECTA_NADA = "def detectar(col):\n    return col.isin([])\n"

_AMOSTRAS_FUMACA = ["12.0 oz", "", "N/A", "17.0", "Portland CA"]


def construir_agente(modelo: str | None = None):
    """Agente que devolve a regra de deteccao de uma coluna."""
    return llm.construir(RegraDeteccao, "deteccao", modelo=modelo)


def _formatar_amostra(representantes_sujos: list[str], contagem: dict) -> str:
    return "\n".join(
        f'  - "{v}"   ({int(contagem.get(v, 0))}x na coluna)' for v in representantes_sujos)


def gerar_regra_deteccao(coluna, amostra, agente=None) -> Detector:
    """Um passe de LLM sobre os representantes SUJOS da coluna, sem ver o clean."""
    agente = agente or construir_agente()
    prompt = ChatPromptTemplate.from_messages([("system", SISTEMA), ("human", HUMANO)])
    regra: RegraDeteccao = (prompt | agente).invoke(
        {
            "coluna": coluna.nome,
            "amostra": _formatar_amostra(amostra.representantes, coluna.contagem),
            "total_linhas": amostra.total_linhas,
            "total_distintos": amostra.total_distintos,
        }
    )

    # Codigo rejeitado pelo portao sobe ate a fronteira em pipeline.py (coluna nao marcada).
    funcao = None
    if (regra.codigo or "").strip():
        funcao = materializar(regra.codigo.strip())
        sandbox.testar_fumaca(funcao, pd.Series(_AMOSTRAS_FUMACA), series_mode=True)
    else:
        # Sem codigo algum: marca zero celulas, nunca inventa deteccao.
        funcao = materializar(DETECTA_NADA)
        regra.codigo = DETECTA_NADA
    return detector_de(regra, funcao)


def materializar(codigo: str):
    return sandbox.materializar(codigo, nome_funcao="detectar",
                                nome_argumento="col", series_mode=True)


def detector_de(regra: RegraDeteccao, funcao) -> Detector:
    return Detector(
        codigo=regra.codigo or DETECTA_NADA,
        funcao=funcao,
        cadeia=regra.cadeia,
        erro_provavel=bool(regra.erro_provavel),
        condicao_regex=regra.condicao_regex,
    )


def detector_nulo(motivo: str = "") -> Detector:
    """Detector que nao marca celula nenhuma, para coluna que falhou."""
    return Detector(
        codigo=DETECTA_NADA,
        funcao=materializar(DETECTA_NADA),
        cadeia=motivo or "coluna nao marcada",
    )
