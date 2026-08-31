"""Carga dos CSVs sujo/limpo e montagem das colunas a processar."""
import pandas as pd

from . import config
from .tipos import Coluna, Tabela


def carregar(caminho_sujo=None, caminho_limpo=None, colunas=None) -> Tabela:
    """Le os dois CSVs como texto literal e devolve a Tabela ja com as Colunas."""
    # keep_default_na=False mantem "N/A", "NA", "null", "-" como o texto que sao:
    # sentinela de ausencia vira algo que o agente pode detectar.
    ler = dict(dtype=str, keep_default_na=False, na_values=[])
    sujo = pd.read_csv(caminho_sujo or config.CSV_SUJO, **ler)
    limpo = pd.read_csv(caminho_limpo or config.CSV_LIMPO, **ler)
    if list(sujo.columns) != list(limpo.columns):
        raise ValueError("dirty e clean tem colunas diferentes")
    if len(sujo) != len(limpo):
        raise ValueError(f"dirty tem {len(sujo)} linhas e clean tem {len(limpo)}")

    nomes = list(colunas) if colunas else [c for c in sujo.columns if c.lower() != "index"]
    faltando = [n for n in nomes if n not in sujo.columns]
    if faltando:
        raise ValueError(f"coluna(s) inexistente(s): {faltando}")
    return Tabela(sujo=sujo, limpo=limpo,
                  colunas=[montar_coluna(sujo, limpo, n) for n in nomes])


def montar_coluna(sujo: pd.DataFrame, limpo: pd.DataFrame, nome: str) -> Coluna:
    """Monta a Coluna com os valores distintos ordenados e a contagem por valor."""
    serie = sujo[nome]
    return Coluna(
        nome=nome,
        sujo=serie,
        limpo=limpo[nome],
        valores_distintos=sorted(serie.unique().tolist()),
        contagem=serie.value_counts().to_dict(),
    )
