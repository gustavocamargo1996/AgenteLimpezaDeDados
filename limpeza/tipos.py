"""Os dados que viajam entre as etapas do pipeline."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Coluna:
    """Uma coluna da tabela, suja e (quando ha) a referencia limpa."""
    nome: str
    sujo: Any
    limpo: Any
    valores_distintos: list[str]
    contagem: dict


@dataclass
class Tabela:
    """Os dois CSVs alinhados por linha e as colunas a processar."""
    sujo: Any
    limpo: Any
    colunas: list[Coluna]


@dataclass
class Amostra:
    """O que o KMeans escolheu mostrar ao agente, com os rotulos do orcamento."""
    representantes: list[str]
    linhas: set[int]
    rotulados: list[dict]
    total_linhas: int
    total_distintos: int


@dataclass
class Detector:
    """Como achar o erro numa coluna."""
    codigo: str
    funcao: Any
    cadeia: str
    orcamento: dict = field(default_factory=dict)
    historico: list = field(default_factory=list)


@dataclass
class Correcao:
    """Como consertar as celulas marcadas de uma coluna."""
    passos: list[dict]
    trilha: dict
    cadeia: str


@dataclass
class Trabalho:
    """Tudo que se sabe sobre uma coluna, preenchido ao longo das etapas."""
    coluna: Coluna
    amostra: Amostra | None
    detector: Detector | None
    correcao: Correcao | None = None
    medida: dict | None = None
