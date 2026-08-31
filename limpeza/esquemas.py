"""Contratos Pydantic (schemas JSON) trocados entre os agentes do pipeline."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

TipoErro = Literal[
    "sufixo_ou_prefixo",
    "formato_numerico",
    "valor_sentinela",
    "valor_ausente",  # celula vazia: nao recuperavel por regra local
    "abreviacao_inconsistente",
    "nenhum",
    "outro",
]


class CadeiaDePensamento(BaseModel):
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


# --- Modo end-to-end (--e2e): deteccao intra-coluna + cascata de correcao ---


class RegraDeteccao(BaseModel):
    """Contrato `detectar(col) -> pd.Series[bool]`. Ver docs/DECISOES.md#deteccao-intra-coluna."""

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
    """Uma FD `determinante -> dependente`, com determinante unico."""

    determinante: str = Field(description="Nome da coluna determinante (unica).")
    dependente: str = Field(description="Nome da coluna dependente (a que se corrige).")
    justificativa: str = Field(
        description="Por que uma coluna determina a outra, citando o padrao observado."
    )
