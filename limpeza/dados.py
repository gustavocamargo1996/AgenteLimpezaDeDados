"""Carga dos CSVs sujo/limpo e montagem das colunas a processar."""
import re
from pathlib import Path

import pandas as pd

from .tipos import Coluna, Tabela


class DadosInvalidos(ValueError):
    """Dataset que nao da' para processar: caminho, colunas ou linhas desalinhadas."""


def nome_dataset(caminho) -> str:
    """Deriva o nome do dataset do stem do CSV sujo, sem o marcador _dirty/_sujo."""
    stem = Path(caminho).stem
    return re.sub(r"_(dirty|sujo)(?=_|$)", "", stem, flags=re.IGNORECASE)


def carregar(caminho_sujo, caminho_limpo, colunas=None) -> Tabela:
    """Le os dois CSVs como texto literal e devolve a Tabela ja com as Colunas."""
    for rotulo, caminho in (("sujo", caminho_sujo), ("limpo", caminho_limpo)):
        if not Path(caminho).exists():
            raise DadosInvalidos(f"arquivo {rotulo} nao encontrado: {caminho}")

    # keep_default_na=False mantem "N/A", "NA", "null", "-" como o texto que sao:
    # sentinela de ausencia vira algo que o agente pode detectar.
    ler = dict(dtype=str, keep_default_na=False, na_values=[])
    sujo = pd.read_csv(caminho_sujo, **ler)
    limpo = pd.read_csv(caminho_limpo, **ler)
    if list(sujo.columns) != list(limpo.columns):
        raise DadosInvalidos("dirty e clean tem colunas diferentes")
    if len(sujo) != len(limpo):
        raise DadosInvalidos(
            f"dirty tem {len(sujo)} linhas e clean tem {len(limpo)}")

    # Lista vazia e' um pedido explicito de nenhuma coluna; so' None quer dizer
    # "todas as colunas do dataset".
    nomes = (list(colunas) if colunas is not None
             else [c for c in sujo.columns if c.lower() != "index"])
    faltando = [n for n in nomes if n not in sujo.columns]
    if faltando:
        raise DadosInvalidos(f"coluna(s) inexistente(s): {faltando}")
    return Tabela(sujo=sujo, limpo=limpo, nome=nome_dataset(caminho_sujo),
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
