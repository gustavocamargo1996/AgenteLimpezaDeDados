"""Deteccao INTRA-COLUNA: constroi a propria mascara de erros, sem ver clean.

Decisao de desenho DECLARADA: a regra e' `detectar(col) -> pd.Series[bool]`;
recebe a COLUNA suja inteira (uma pd.Series de strings) e devolve uma pd.Series
booleana do mesmo tamanho (True por linha corrompida), SEM olhar outras colunas.
O idioma e' `col.str.contains(marcador)` -- o mesmo com que o ZeroDC atinge
abv/city F1=1.0 (regras reais: abv=`df['abv'].str.contains('%')`, city=` [A-Z]{2}$`).
Suficiente para o beers (todo erro e' visivel na propria coluna: sufixo em
ounces, '%' em abv, 'N/A' em ibu, ' XX' final em city, vazio em state). NAO e' o
`fun(df) -> Series` cross-column do ZeroDC (detection.py:147-155,269): so' a
propria coluna -- erros dependentes de contexto passam batido aqui, e essa
limitacao e' aceita para esta POC.

Um unico passe de LLM por coluna (sem loop de active learning): o agente ve so'
os representantes SUJOS e emite a RegraDeteccao ja com o `codigo` de
`detectar(col)`. O codigo passa pelo portao AST do sandbox em modo serie
(`series_mode=True`: allowlist de atributos; `pd` fora do namespace).
`construir_mascara` aplica `detectar(col)` uma vez por coluna e le o resultado
POSICIONALMENTE; NAO importa nem le clean -- auditavel neste arquivo.
"""
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .. import config, sandbox
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

# Fallback de 1 passe: se o unico invoke nao devolve codigo que passe no portao,
# nao marca nada (mascara toda 0 nesta coluna). Nao ha laco de reparo -- a decisao
# foi deteccao em 1 passe. Nunca inventa deteccao. No idioma Series, tudo-False
# alinhado a `col` = `col.isin([])` (usa so' o atributo permitido `isin`, sem pd
# nem `.index`); passa na allowlist do series_mode.
DETECTA_NADA = "def detectar(col):\n    return col.isin([])\n"

_AMOSTRAS_FUMACA = ["12.0 oz", "", "N/A", "17.0", "Portland CA"]


def construir_agente(modelo: str | None = None):
    llm = ChatOpenAI(
        model=modelo or config.MODELO_LLM,
        temperature=config.TEMPERATURA,
        timeout=config.TIMEOUT_LLM,
        max_retries=2,
    )
    return llm.with_structured_output(RegraDeteccao)


def _formatar_amostra(representantes_sujos: list[str], contagem: dict) -> str:
    linhas = []
    for valor in representantes_sujos:
        freq = int(contagem.get(valor, 0))
        linhas.append(f'  - "{valor}"   ({freq}x na coluna)')
    return "\n".join(linhas)


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

    funcao = None
    if (regra.codigo or "").strip():
        try:
            funcao = materializar(regra.codigo.strip())
            sandbox.testar_fumaca(
                funcao, pd.Series(_AMOSTRAS_FUMACA), series_mode=True
            )
        except sandbox.CodigoRejeitado:
            funcao = None

    if funcao is None:
        # Codigo reprovado no portao: marca zero celulas, nunca inventa deteccao.
        funcao = materializar(DETECTA_NADA)
        regra.codigo = DETECTA_NADA
    return detector_de(regra, funcao)


def materializar(codigo: str):
    """Compila `detectar(col)` pelo portao AST do sandbox, em modo Series."""
    return sandbox.materializar(codigo, nome_funcao="detectar",
                                nome_argumento="col", series_mode=True)


def detector_de(regra: RegraDeteccao, funcao) -> Detector:
    """Embrulha a regra validada e sua funcao viva num Detector."""
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
