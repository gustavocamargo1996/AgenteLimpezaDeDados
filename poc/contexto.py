"""Informacao mutua normalizada -- usada SO' na correcao, nunca na deteccao.

Replica detection.py:199-220 do ZeroDC (funcao calc_mi_2):
  - MI bruta por par de colunas via sklearn.metrics.mutual_info_score;
  - pre-filtro `count(alvo, col) >= 2`: pares que aparecem so' uma vez nao
    sustentam dependencia e sujam a estimativa;
  - normaliza dividindo pelo MAXIMO da linha (o maior MI vira 1.0);
  - arredonda para 1 casa.

Serve para eleger colunas candidatas a determinante na FD (camada 2) e a
contexto no fallback (camada 3). NAO entra na deteccao intra-coluna -- ali seria
peso morto, porque a regra ve um escalar, nao a linha.
"""
from sklearn.metrics import mutual_info_score

from . import config


def calc_mi(df, alvo: str) -> dict:
    """MI normalizada de cada coluna de `df` em relacao a `alvo`.

    Devolve {coluna: mi_normalizada_1_casa}. A propria coluna `alvo` aparece
    (tipicamente com 1.0) -- quem chama e' que exclui o alvo ao escolher
    determinantes.
    """

    def mi_par(col: str) -> float:
        if col == alvo:
            return float(mutual_info_score(df[alvo], df[col]))
        # pre-filtro: pares (alvo, col) que ocorrem ao menos 2x
        contagem = df.groupby([alvo, col])[alvo].transform("size")
        filtrado = df[contagem >= 2]
        if filtrado.empty:
            return 0.0
        a = filtrado[alvo].reset_index(drop=True)
        b = filtrado[col].reset_index(drop=True)
        return float(mutual_info_score(a, b))

    brutos = {col: mi_par(col) for col in df.columns}
    maximo = max(brutos.values()) if brutos else 0.0
    if maximo == 0.0:
        return {col: 0.0 for col in brutos}
    return {col: round(v / maximo, 1) for col, v in brutos.items()}


def candidatos_determinantes(df, alvo: str, limiar: float | None = None) -> list[str]:
    """Colunas (exceto o proprio alvo) com MI normalizada >= limiar."""
    limiar = config.MI_THRESHOLD if limiar is None else limiar
    mi = calc_mi(df, alvo)
    return [col for col, v in mi.items() if col != alvo and v >= limiar]
