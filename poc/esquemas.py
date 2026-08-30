"""Contrato entre os dois agentes.

O JSON e' apertado de proposito. Se ele fosse frouxo, o tradutor teria que
inferir tambem -- e quando a correcao saisse errada voce nao saberia se falhou
a inferencia (agente 1) ou a traducao (agente 2). Schema rigido mantem o
diagnostico limpo.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

TipoErro = Literal[
    "sufixo_ou_prefixo",        # '12.0 oz' -> '12.0'
    "formato_numerico",         # '17.0' onde se espera '17'
    "valor_sentinela",          # 'N/A', '-', 'desconhecido'
    "valor_ausente",            # celula vazia: NAO recuperavel por regra local
    "abreviacao_inconsistente", # 'CA' vs 'California'
    "nenhum",
    "outro",
]


class CadeiaDePensamento(BaseModel):
    """A entrega principal da POC. Quatro secoes fixas, na ordem do raciocinio."""

    observacao_da_amostra: str = Field(
        description="O que voce ve nos valores mostrados. Descreva, nao conclua ainda."
    )
    identificacao_do_padrao: str = Field(
        description="Qual regularidade separa o corrompido do intacto, e por que voce "
        "acredita que e' regularidade e nao coincidencia da amostra."
    )
    formulacao_da_regra: str = Field(
        description="Como o padrao vira uma transformacao. Justifique cada escolha."
    )
    limites_da_regra: str = Field(
        description="Quando a regra NAO deve agir, e o que ela e' incapaz de consertar. "
        "Seja concreto: cite um valor da amostra que a regra deve deixar em paz."
    )


class Transformacao(BaseModel):
    tipo: Literal["regex_sub", "mapeamento", "constante", "nenhuma"]
    padrao: Optional[str] = Field(None, description="regex de busca, para tipo=regex_sub")
    substituicao: Optional[str] = Field(None, description="troca, para tipo=regex_sub")
    mapa: Optional[dict[str, str]] = Field(None, description="de->para, para tipo=mapeamento")
    valor: Optional[str] = Field(None, description="valor fixo, para tipo=constante")


class Exemplo(BaseModel):
    de: str
    para: str


class RegraCorrecao(BaseModel):
    coluna: str
    erro_detectado: bool = Field(
        description="False se a amostra nao sustenta a existencia de erro nesta coluna."
    )
    tipo_erro: TipoErro
    descricao_padrao: str = Field(description="Uma frase: qual e' o defeito.")
    condicao_regex: Optional[str] = Field(
        None,
        description="Regex que decide QUANDO aplicar. Obrigatoria se a transformacao "
        "nao for 'nenhuma'. E' o unico freio contra estragar celula que ja estava certa.",
    )
    transformacao: Transformacao
    exemplos: list[Exemplo] = Field(default_factory=list)
    confianca: float = Field(ge=0.0, le=1.0)
    cadeia_de_pensamento: CadeiaDePensamento


class CodigoGerado(BaseModel):
    codigo: str = Field(
        description="Uma unica funcao `def corrigir(valor):` retornando string. "
        "Sem import (o modulo `re` ja existe no escopo), sem codigo fora da funcao."
    )
    nota_de_traducao: str = Field(
        description="Uma ou duas frases: como o JSON virou este codigo, e qualquer "
        "decisao que voce teve que tomar por conta propria."
    )


# --------------------------------------------------------------------------- #
# Modo end-to-end (--e2e): deteccao intra-coluna + cascata de correcao.
# --------------------------------------------------------------------------- #


class RegraDeteccao(BaseModel):
    """Contrato da deteccao INTRA-COLUNA (idioma Series-contains).

    A funcao gerada e' `detectar(col) -> pd.Series[bool]`: recebe a COLUNA suja
    inteira (uma pd.Series de strings) e devolve uma pd.Series booleana do mesmo
    tamanho (True por linha corrompida). Ela NAO ve outras colunas -- so' a
    propria coluna, elemento a elemento (o idioma `col.str.contains(...)`). Isto
    NAO e' o `fun(df) -> Series` cross-column do ZeroDC; a limitacao (erros
    dependentes de contexto passam batido) e' aceita para o beers.
    """

    erro_provavel: bool = Field(
        description="False se os representantes sujos nao sustentam a existencia "
        "de erro detectavel olhando so' a propria coluna."
    )
    condicao_regex: Optional[str] = Field(
        None,
        description="Regex que caracteriza a celula errada (documenta o criterio; "
        "o veredito efetivo vem do campo `codigo`).",
    )
    codigo: str = Field(
        description="Uma unica funcao `def detectar(col):` que devolve uma "
        "pd.Series[bool] do tamanho de `col` (True por linha errada), no idioma "
        "elementwise `col.astype(str).str.contains(marcador, regex=False, na=False)`. "
        "Sem import (o modulo `re` ja existe no escopo), sem codigo fora da funcao, "
        "sem operacoes cross-row (duplicated/mode/groupby/shift)."
    )
    cadeia: str = Field(
        description="Como voce chegou ao criterio de deteccao, citando valores "
        "concretos dos representantes sujos."
    )


class DependenciaFuncional(BaseModel):
    """Uma FD `determinante -> dependente`, determinante UNICO (simplificacao
    declarada; o ZeroDC aceita determinante composto)."""

    determinante: str = Field(description="Nome da coluna determinante (unica).")
    dependente: str = Field(description="Nome da coluna dependente (a que se corrige).")
    justificativa: str = Field(
        description="Por que uma coluna determina a outra, citando o padrao observado."
    )


class CorrecaoCelula(BaseModel):
    """Ultima camada: correcao de UM valor distinto pelo LLM.

    `valor` PODE vir vazio quando nao ha correcao confiavel -- e proibido forcar
    'null' ou inventar um valor plausivel so' para nao deixar vazio.
    """

    cadeia: str = Field(description="O raciocinio que leva ao valor corrigido.")
    valor: str = Field(
        description="O valor corrigido. Deixe vazio ('') se nao houver correcao "
        "confiavel; NUNCA invente um valor nem escreva 'null'."
    )
