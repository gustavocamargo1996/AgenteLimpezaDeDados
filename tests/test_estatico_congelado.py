"""Trava que o corpo estatico de empacotar.py bate com a fixture congelada."""
from pathlib import Path

from limpeza.empacotar import _ESTATICO

FIXTURES = Path(__file__).parent / "fixtures"


def test_estatico_bate_com_a_fixture_congelada():
    corpo = _ESTATICO.strip("\n")
    assert corpo, "_ESTATICO vazio -- extracao quebrada, nao um match vacuoso"
    fixture = (FIXTURES / "limpador_beers_congelado.py").read_text(encoding="utf-8")
    assert fixture.rstrip("\n").endswith(corpo), (
        "o corpo estatico atual de limpeza/empacotar.py diverge do final de "
        "tests/fixtures/limpador_beers_congelado.py -- o limpador que "
        "empacotar.py gera hoje nao e' mais o mesmo que "
        "test_invariante.py::test_f1_reparo_do_limpador_congelado mede"
    )
