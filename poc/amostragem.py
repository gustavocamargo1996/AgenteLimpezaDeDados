"""Selecao de representantes por KMeans -- o coracao herdado do ZeroDC.

Ideia: em vez de mandar a coluna inteira ao LLM (caro e ruidoso), agrupa os
valores distintos no espaco de embeddings e mostra, de cada grupo, o mais
atipico e o mais tipico. O atipico expoe a corrupcao; o tipico ancora a norma.

Limite estrutural que a POC existe para demonstrar: quando a corrupcao atinge
100% da coluna, a forma corrompida E' a norma e nao ha atipico para achar.
Nenhuma quantidade de clusters resolve isso.
"""
import numpy as np
from sklearn.cluster import KMeans

from . import config


def selecionar(matriz: np.ndarray, n_clusters=None, maximo=None) -> list[int]:
    """Indices (na lista de valores distintos) escolhidos como representantes."""
    n_clusters = n_clusters or config.N_CLUSTERS
    maximo = maximo or config.MAX_REPRESENTANTES

    total = len(matriz)
    if total == 0:
        return []
    if total <= maximo:
        return list(range(total))

    k = max(2, min(n_clusters, total))
    modelo = KMeans(n_clusters=k, random_state=0, n_init=10).fit(matriz)

    escolhidos: list[int] = []
    for grupo in range(k):
        membros = np.where(modelo.labels_ == grupo)[0]
        if len(membros) == 0:
            continue
        distancias = np.linalg.norm(matriz[membros] - modelo.cluster_centers_[grupo], axis=1)
        escolhidos.append(int(membros[distancias.argmax()]))  # mais atipico
        if len(membros) > 1:
            escolhidos.append(int(membros[distancias.argmin()]))  # mais tipico

    vistos, saida = set(), []
    for i in escolhidos:
        if i not in vistos:
            vistos.add(i)
            saida.append(i)
    return saida[:maximo]
