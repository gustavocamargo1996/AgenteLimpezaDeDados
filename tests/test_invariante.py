"""Trava os numeros publicados do limpador de beers."""
from pathlib import Path

from avaliar_limpador import avaliar

FIXTURES = Path(__file__).parent / "fixtures"


def test_f1_reparo_do_limpador_congelado():
    total = avaliar(
        caminho_limpador=FIXTURES / "limpador_beers_congelado.py",
        caminho_sujo=FIXTURES / "beers_dirty_300.csv",
        caminho_limpo=FIXTURES / "beers_clean_300.csv",
    )["total"]
    assert total["erros"] == 495
    assert total["mudancas"] == 121
    assert total["tp"] == 121
    assert total["precisao"] == 1.0
    assert total["recall"] == 0.2444
    assert total["f1"] == 0.3929
    assert total["flags"] == 71
