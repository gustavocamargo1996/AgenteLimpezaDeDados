"""amostragem.selecionar: farthest-point + centroide por grupo, deterministico (random_state=0)."""
import numpy as np

from limpeza import amostragem

# Dois grupos bem separados; dentro de cada um ha um ponto tipico (perto do
# centroide) e um atipico (longe), com distancias inequivocas (sem empate).
_GRUPO_A = [(0, 0), (1, 0), (-1, 0), (0, 5)]      # tipico=(0,0) idx0, atipico=(0,5) idx3
_GRUPO_B = [(1000, 1000), (1005, 1000), (1000, 1001)]  # tipico idx4, atipico idx5
_MATRIZ = np.array(_GRUPO_A + _GRUPO_B, dtype=float)


def test_mesma_entrada_da_os_mesmos_representantes():
    esc1 = amostragem.selecionar(_MATRIZ, n_clusters=2, maximo=6)
    esc2 = amostragem.selecionar(_MATRIZ, n_clusters=2, maximo=6)
    assert esc1 == esc2


def test_selecao_traz_o_mais_tipico_e_o_mais_atipico_de_cada_grupo():
    escolhidos = amostragem.selecionar(_MATRIZ, n_clusters=2, maximo=6)
    # idx0/idx3 = tipico/atipico do grupo A; idx4/idx5 = tipico/atipico do grupo B.
    assert set(escolhidos) == {0, 3, 4, 5}
    assert len(escolhidos) == len(set(escolhidos)), "sem indice repetido"


def test_total_menor_que_maximo_devolve_tudo_sem_clusterizar():
    escolhidos = amostragem.selecionar(_MATRIZ[:3], n_clusters=2, maximo=12)
    assert escolhidos == [0, 1, 2]
