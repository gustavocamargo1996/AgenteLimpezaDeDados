"""Trava que o corpo estatico de empacotar.py bate com cada fixture congelada."""
from pathlib import Path

import pytest

from limpeza.empacotar import _ESTATICO

FIXTURES = Path(__file__).parent / "fixtures"
CONGELADAS = sorted(FIXTURES.glob("limpador_*_congelado.py"))


def test_a_fixture_do_beers_esta_entre_as_congeladas():
    assert "limpador_beers_congelado.py" in [p.name for p in CONGELADAS]


@pytest.mark.parametrize("fixture", CONGELADAS, ids=lambda p: p.name)
def test_estatico_bate_com_a_fixture_congelada(fixture):
    corpo = _ESTATICO.strip("\n")
    assert corpo, "_ESTATICO vazio -- extracao quebrada, nao um match vacuoso"
    texto = fixture.read_text(encoding="utf-8")
    assert texto.rstrip("\n").endswith(corpo), (
        f"o corpo estatico atual de limpeza/empacotar.py diverge do final de "
        f"{fixture.name} -- o limpador que empacotar.py gera hoje nao e' mais o "
        f"mesmo que o invariante dessa fixture mede"
    )
