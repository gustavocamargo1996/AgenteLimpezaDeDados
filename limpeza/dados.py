"""Etapa 1 -- carga dos CSVs sujo/limpo e montagem das colunas a processar."""
import re
from pathlib import Path

import pandas as pd

from .tipos import Coluna, Tabela


class DadosInvalidos(ValueError):
    pass


def nome_dataset(caminho) -> str:
    stem = Path(caminho).stem
    return re.sub(r"_(dirty|sujo)(?=_|$)", "", stem, flags=re.IGNORECASE)


def carregar(caminho_sujo, caminho_limpo, colunas=None) -> Tabela:
    """Le os dois CSVs como texto literal e devolve a Tabela ja com as Colunas."""
    if not Path(caminho_sujo).exists():
        raise DadosInvalidos(f"arquivo sujo nao encontrado: {caminho_sujo}")
    if not Path(caminho_limpo).exists():
        raise DadosInvalidos(f"arquivo limpo nao encontrado: {caminho_limpo}")

    # keep_default_na=False: sem isso o pandas converte "N/A" em NaN e apaga
    # 1.005 erros reais de `ibu`. Ver docs/DECISOES.md#carga-literal.
    ler = dict(dtype=str, keep_default_na=False, na_values=[])
    sujo = pd.read_csv(caminho_sujo, **ler)
    limpo = pd.read_csv(caminho_limpo, **ler)
    if list(sujo.columns) != list(limpo.columns):
        raise DadosInvalidos("dirty e clean tem colunas diferentes")
    if len(sujo) != len(limpo):
        raise DadosInvalidos(
            f"dirty tem {len(sujo)} linhas e clean tem {len(limpo)}")

    # Lista vazia = nenhuma coluna; so' None quer dizer "todas".
    nomes = (list(colunas) if colunas is not None
             else [c for c in sujo.columns if c.lower() != "index"])
    faltando = [n for n in nomes if n not in sujo.columns]
    if faltando:
        raise DadosInvalidos(f"coluna(s) inexistente(s): {faltando}")
    return Tabela(sujo=sujo, limpo=limpo, nome=nome_dataset(caminho_sujo),
                  colunas=[montar_coluna(sujo, limpo, n) for n in nomes])


def montar_coluna(sujo: pd.DataFrame, limpo: pd.DataFrame, nome: str) -> Coluna:
    serie = sujo[nome]
    return Coluna(
        nome=nome,
        sujo=serie,
        limpo=limpo[nome],
        valores_distintos=sorted(serie.unique().tolist()),
        contagem=serie.value_counts().to_dict(),
    )
