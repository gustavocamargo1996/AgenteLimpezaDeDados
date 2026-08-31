"""Os dados que viajam entre as etapas do pipeline."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Coluna:
    nome: str
    sujo: Any
    limpo: Any
    valores_distintos: list[str]
    contagem: dict


@dataclass
class Tabela:
    sujo: Any
    limpo: Any
    nome: str
    colunas: list[Coluna]


@dataclass
class Amostra:
    representantes: list[str]
    linhas: set[int]
    rotulados: list[dict]
    total_linhas: int
    total_distintos: int


@dataclass
class Detector:
    codigo: str
    funcao: Any
    cadeia: str
    orcamento: dict = field(default_factory=dict)
    historico: list = field(default_factory=list)
    # Documentais: o relatorio os publica, mas nada no pipeline decide com base neles.
    erro_provavel: bool = False
    condicao_regex: str | None = None


@dataclass
class Correcao:
    passos: list[dict]
    trilha: dict
    cadeia: str


@dataclass
class Trabalho:
    # Preenchido incrementalmente ao longo das etapas do pipeline.
    coluna: Coluna
    amostra: Amostra | None
    detector: Detector | None
    correcao: Correcao | None = None
    medida: dict | None = None
