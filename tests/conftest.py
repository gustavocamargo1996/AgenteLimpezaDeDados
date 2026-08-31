"""Fixtures compartilhadas pela suite."""
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
LER = dict(dtype=str, keep_default_na=False, na_values=[])


@pytest.fixture
def beers_sujo():
    return pd.read_csv(FIXTURES / "beers_dirty_300.csv", **LER)


@pytest.fixture
def beers_limpo():
    return pd.read_csv(FIXTURES / "beers_clean_300.csv", **LER)
