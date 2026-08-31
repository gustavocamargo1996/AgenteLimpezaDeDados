"""construir_mascara/aplicar_detectores: por-coluna, posicional, falha isola a coluna, sem clean."""
import inspect
import io
import tokenize

import pandas as pd

from limpeza import sandbox
from limpeza.deteccao import aplicar_detectores, construir_mascara


def _detector(codigo):
    return sandbox.materializar(codigo, nome_funcao="detectar",
                                nome_argumento="col", series_mode=True)


def test_marca_so_o_marcador():
    # construir_mascara(trabalhos, tabela) exige objetos Trabalho/Tabela completos;
    # aplicar_detectores expoe a mesma logica de marcacao por coluna direto.
    df = pd.DataFrame({"ibu": ["N/A", "42", "N/A"]})
    funcao = _detector("def detectar(col):\n    return col == 'N/A'\n")
    mascara = aplicar_detectores(df, {"ibu": funcao})
    assert list(mascara["ibu"]) == [1, 0, 1]


def test_funcao_que_devolve_escalar_zera_a_coluna_sem_derrubar_o_run():
    df = pd.DataFrame({"ibu": ["N/A", "42"]})
    funcao = _detector("def detectar(col):\n    return True\n")
    mascara = aplicar_detectores(df, {"ibu": funcao})
    assert list(mascara["ibu"]) == [0, 0]


def test_sem_log_quando_detectar_devolve_serie_booleana_valida():
    df = pd.DataFrame({"c": ["0.05", "0.06%", "0.07"]})
    f_pct = _detector(
        "def detectar(col):\n"
        "    return col.astype(str).str.contains('%', regex=False, na=False)\n"
    )
    log = []
    mask = aplicar_detectores(df, {"c": f_pct}, log)
    assert list(mask["c"]) == [0, 1, 0]
    assert log == []


def test_retorno_nao_serie_zera_coluna_e_loga():
    df = pd.DataFrame({"c": ["0.05", "0.06%", "0.07"]})
    f_escalar = _detector("def detectar(col):\n    return True\n")
    log = []
    mask = aplicar_detectores(df, {"c": f_escalar}, log)
    assert list(mask["c"]) == [0, 0, 0]
    assert len(log) == 1 and log[0]["coluna"] == "c" and "nao-Series" in log[0]["motivo"]


def test_tamanho_divergente_zera_coluna_e_loga_sem_boolean_index():
    df = pd.DataFrame({"c": ["0.05", "0.06%", "0.07"]})

    def _f_tam_errado(coluna):
        return pd.Series([True])

    log = []
    mask = aplicar_detectores(df, {"c": _f_tam_errado}, log)
    assert list(mask["c"]) == [0, 0, 0]
    assert len(log) == 1 and "tamanho divergente" in log[0]["motivo"]


def test_detecta_nada_marca_nada_sem_log():
    from limpeza.deteccao import DETECTA_NADA

    df = pd.DataFrame({"c": ["0.05", "0.06%", "0.07"]})
    f_nada = _detector(DETECTA_NADA)
    log = []
    mask = aplicar_detectores(df, {"c": f_nada}, log)
    assert list(mask["c"]) == [0, 0, 0]
    assert log == []


def _nomes_executaveis(*funcoes) -> set:
    """Nomes que o CODIGO das funcoes usa, ignorando docstring e comentario."""
    fonte = "".join(inspect.getsource(f) for f in funcoes)
    return {tok.string
            for tok in tokenize.generate_tokens(io.StringIO(fonte).readline)
            if tok.type == tokenize.NAME}


def test_construir_mascara_delega_por_trabalho():
    """construir_mascara(trabalhos, tabela) e' o wrapper publico usado pelo pipeline."""
    import types

    df = pd.DataFrame({"ibu": ["N/A", "42", "N/A"]})
    funcao = _detector("def detectar(col):\n    return col == 'N/A'\n")
    trabalho = types.SimpleNamespace(
        coluna=types.SimpleNamespace(nome="ibu"),
        detector=types.SimpleNamespace(funcao=funcao),
    )
    tabela = types.SimpleNamespace(sujo=df)
    mascara = construir_mascara([trabalho], tabela)
    assert list(mascara["ibu"]) == [1, 0, 1]


def test_construir_mascara_nao_le_clean():
    """Auditoria: a assinatura nem o codigo executavel referenciam clean/limpo."""
    params = set(inspect.signature(aplicar_detectores).parameters)
    params |= set(inspect.signature(construir_mascara).parameters)
    assert "clean" not in params and "limpo" not in params

    nomes = _nomes_executaveis(aplicar_detectores, construir_mascara)
    assert {"clean", "limpo", "read_csv"}.isdisjoint(nomes)
