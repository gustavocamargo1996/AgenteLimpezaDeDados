"""A deteccao por dependencia funcional, com agente duble (sem rede)."""
from pathlib import Path

import pandas as pd

FIXTURES = Path(__file__).parent / "fixtures"
LER = dict(dtype=str, keep_default_na=False, na_values=[])


def _environment():
    return (pd.read_csv(FIXTURES / "environment_dirty_300.csv", **LER),
            pd.read_csv(FIXTURES / "environment_clean_300.csv", **LER))


def test_fixture_tem_os_erros_cross_column_esperados():
    """Fixa o fenomeno que este projeto existe para atacar."""
    sujo, limpo = _environment()
    assert len(sujo) == 300
    assert int((sujo["State"] != limpo["State"]).sum()) == 54
    assert int((sujo["Climate_Zone"] != limpo["Climate_Zone"]).sum()) == 26


def test_fixture_nenhum_erro_e_alcancavel_intra_coluna():
    """Todo erro dessas duas colunas usa um valor que tambem aparece correto."""
    sujo, limpo = _environment()
    for col in ("State", "Climate_Zone"):
        err = sujo[col] != limpo[col]
        ambiguo = pd.DataFrame({"v": sujo[col], "e": err}).groupby("v")["e"].transform(
            lambda s: s.any() and (~s).any())
        assert int((err & ~ambiguo).sum()) == 0, f"{col} tem erro alcancavel intra-coluna"
