"""Carga do dataset e particao amostra/holdout.

Regra de ouro do holdout: se o LLM viu um VALOR, todas as linhas que contem
esse valor saem da avaliacao. Testar a regra sobre um valor ja mostrado mede
memorizacao, nao generalizacao -- foi exatamente o furo que a auditoria
encontrou no ZeroDC (o split de validacao e' calculado e nunca usado).
"""
from dataclasses import dataclass, field

import pandas as pd

from . import config


@dataclass
class Coluna:
    """Uma coluna do dataset: valores sujos, limpos e o conjunto de distintos.

    Nao guarda particao -- quem fatia amostra e holdout e' `particionar()`, que
    recebe esta Coluna e devolve os indices. O objeto e' so' a coluna inteira.
    """

    nome: str
    sujo: pd.Series
    limpo: pd.Series
    valores_distintos: list[str] = field(default_factory=list)

    @property
    def total_celulas(self) -> int:
        return len(self.sujo)

    @property
    def celulas_erradas(self) -> int:
        return int((self.sujo != self.limpo).sum())


def carregar(caminho_sujo=None, caminho_limpo=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Le os dois CSVs como texto literal, byte a byte.

    DESVIO DELIBERADO DO ZERODC. O original faz
    `pd.read_csv(dtype=str).fillna('null')`, e o pandas converte "N/A" em NaN por
    default -- entao o "N/A" do dirty e o vazio do clean viram o MESMO token e a
    diferenca some. Na coluna `ibu` do beers isso apaga 1005 erros reais (42% da
    coluna) antes de qualquer algoritmo rodar.

    `keep_default_na=False` mantem "N/A", "NA", "null", "-" como o texto que sao.
    Sentinela de ausencia vira uma coisa que o agente PODE detectar, em vez de um
    artefato de carga que ninguem ve.
    """
    ler = dict(dtype=str, keep_default_na=False, na_values=[])
    sujo = pd.read_csv(caminho_sujo or config.CSV_SUJO, **ler)
    limpo = pd.read_csv(caminho_limpo or config.CSV_LIMPO, **ler)
    if list(sujo.columns) != list(limpo.columns):
        raise ValueError("dirty e clean tem colunas diferentes")
    if len(sujo) != len(limpo):
        raise ValueError(f"dirty tem {len(sujo)} linhas e clean tem {len(limpo)}")
    return sujo, limpo


def montar_coluna(sujo: pd.DataFrame, limpo: pd.DataFrame, nome: str) -> Coluna:
    col = Coluna(nome=nome, sujo=sujo[nome], limpo=limpo[nome])
    col.valores_distintos = sorted(col.sujo.unique().tolist())
    return col


def montar_rotulados(
    col: Coluna, linhas_mostradas: set[int]
) -> tuple[list[dict], pd.Index]:
    """Orcamento de rotulagem por coluna, para o modo --e2e.

    Recebe as linhas efetivamente exibidas (primeira ocorrencia de cada
    representante) e devolve, para cada uma, o par sujo->limpo e se e' erro. Usar
    `clean` AQUI e' o "human labeling budget" do ZeroDC -- legitimo: rotula um
    orcamento pequeno, NAO detecta a coluna inteira.

    Tambem devolve o indice desse conjunto -- e' o HOLDOUT da correcao: as
    celulas cujo valor limpo o portao viu saem da metrica final, senao o numero
    infla (mede memorizacao, nao correcao).
    """
    rows: list[dict] = []
    for idx in sorted(linhas_mostradas):
        sujo = col.sujo.at[idx]
        limpo = col.limpo.at[idx]
        rows.append(
            {"indice": int(idx), "sujo": sujo, "limpo": limpo, "eh_erro": sujo != limpo}
        )
    holdout = pd.Index(sorted(int(i) for i in linhas_mostradas))
    return rows, holdout


def particionar(
    col: Coluna, valores_mostrados: set[str], linhas_mostradas: set[int]
) -> tuple[pd.Index, pd.Index]:
    """Devolve (holdout_por_valor, holdout_por_linha).

    por VALOR  -- linhas cujo valor sujo o agente nunca viu. Mede generalizacao.
                  Pode vir vazio quando uma classe de erro inteira e' um unico
                  valor, e' esse o preco de ser estrito.
    por LINHA  -- tudo menos as linhas efetivamente exibidas. Mede se a regra se
                  aplica ao volume. Sempre populado.

    O primeiro esta contido no segundo.
    """
    por_valor = col.sujo[~col.sujo.isin(valores_mostrados)].index
    por_linha = col.sujo.index.difference(pd.Index(sorted(linhas_mostradas)))
    return por_valor, por_linha
