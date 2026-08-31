"""Portao AST: aceita codigo valido, rejeita o resto, allowlist aditiva do modo serie."""
import pandas as pd
import pytest

from limpeza import sandbox
from limpeza.deteccao import regra
from limpeza.sandbox import CodigoRejeitado


def test_aceita_funcao_valida():
    sandbox.validar("def corrigir(valor):\n    return valor.strip()\n")


@pytest.mark.parametrize("codigo,trecho", [
    ("def fix(valor):\n    return valor\n", "deve se chamar"),
    ("def corrigir(v, w):\n    return v\n", "exatamente um argumento"),
    ("import re\ndef corrigir(valor):\n    return valor\n", "exatamente uma definicao"),
    ("def corrigir(valor):\n    return eval(valor)\n", "chamada proibida"),
    ("x = 1\ndef corrigir(valor):\n    return valor\n", "exatamente uma definicao"),
])
def test_rejeita(codigo, trecho):
    with pytest.raises(CodigoRejeitado, match=trecho):
        sandbox.validar(codigo)


def test_modo_serie_rejeita_atributo_fora_da_allowlist():
    with pytest.raises(CodigoRejeitado, match="allowlist"):
        sandbox.validar("def detectar(col):\n    return col.duplicated()\n",
                        nome_funcao="detectar", nome_argumento="col", series_mode=True)


def test_modo_serie_aceita_o_idioma_contains():
    sandbox.validar("def detectar(col):\n    return col.astype(str).str.contains('N/A')\n",
                    nome_funcao="detectar", nome_argumento="col", series_mode=True)


def _mat_det(codigo: str):
    """Materializa uma funcao `detectar(col)` no series_mode, como o modulo regra faz."""
    return sandbox.materializar(codigo, nome_funcao="detectar",
                                nome_argumento="col", series_mode=True)


def _rejeita(codigo: str, series_mode: bool) -> bool:
    """True se `validar` rejeita `codigo` no modo pedido (sem executar)."""
    try:
        sandbox.validar(codigo, nome_funcao="detectar", nome_argumento="col",
                        series_mode=series_mode)
        return False
    except CodigoRejeitado:
        return True


def test_idioma_contains_materializa_e_roda():
    f_pct = _mat_det(
        "def detectar(col):\n"
        "    return col.astype(str).str.contains('%', regex=False, na=False)\n"
    )
    res = f_pct(pd.Series(["0.05", "0.05%"]))
    assert isinstance(res, pd.Series) and list(res) == [False, True]


def test_contrato_antigo_corrigir_escalar_intacto():
    f_old = sandbox.materializar("def corrigir(valor):\n    return valor.strip()")
    assert f_old(" x ") == "x"


def test_sob_contrato_detectar_funcao_corrigir_e_rejeitada():
    assert _rejeita("def corrigir(col):\n    return col", series_mode=True)


def test_smoke_serie_rejeita_retorno_escalar():
    f_escalar = _mat_det("def detectar(col):\n    return '%' in col\n")
    with pytest.raises(CodigoRejeitado):
        sandbox.testar_fumaca(f_escalar, pd.Series(regra._AMOSTRAS_FUMACA), series_mode=True)


@pytest.mark.parametrize("codigo", [
    "def detectar(col):\n    return col.to_csv('x')\n",
    "def detectar(col):\n    return col.duplicated()\n",
    "def detectar(col):\n    return pd.read_csv('x')\n",
    "def detectar(col):\n    return pd.io.common.get_handle('x','w')\n",
    "def detectar(col):\n    return col.mode()\n",
])
def test_allowlist_rejeita_io_e_cross_row(codigo):
    assert _rejeita(codigo, series_mode=True)


def test_escalar_nao_aplica_allowlist_aditiva_apenas_series():
    # O mesmo atributo cross-row passa no gate ESCALAR (default): a allowlist e' aditiva.
    assert not _rejeita("def detectar(col):\n    return col.duplicated()\n", series_mode=False)


def test_controle_fronteira_real_e_operacional_nao_a_allowlist():
    """Teste-controle: str.contains permite regex arbitraria e ainda assim passa.

    A fronteira real de seguranca e' OPERACIONAL (dados publicos, offline), nao a
    allowlist ser exaustiva -- limite declarado, nao um bug.
    """
    assert not _rejeita(
        "def detectar(col):\n    return col.str.contains(r'(a+)+$', regex=True, na=False)\n",
        series_mode=True,
    )
