"""Os dados que viajam entre as etapas do pipeline."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Coluna:
    """Uma coluna da tabela: `sujo`/`limpo` sao pd.Series (a coluna inteira, nao o dataset)."""
    nome: str
    sujo: Any
    limpo: Any
    valores_distintos: list[str]
    contagem: dict


@dataclass
class Tabela:
    """Os dois CSVs completos: `sujo`/`limpo` sao pd.DataFrame (nao pd.Series como em Coluna)."""
    sujo: Any
    limpo: Any
    nome: str
    colunas: list[Coluna]


@dataclass
class Amostra:
    """O que o KMeans mostrou: `representantes` (valores distintos) e `linhas` (indices de linha, viram holdout)."""
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
    # Documentais: o relatorio os publica, mas nada no pipeline decide com base neles.
    erro_provavel: bool = False
    condicao_regex: str | None = None


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
    dependencia: Any = None  # DependenciaFuncional aprovada na deteccao, ou None
    correcao: Correcao | None = None
    medida: dict | None = None
