"""escrever_limpador gera um .py autonomo que compila, importa e aplica corretamente."""
import importlib.util
import py_compile
import tempfile
from pathlib import Path

import pandas as pd

from limpeza import empacotar


def _gerar_e_importar(nome, detectores, plano, colunas):
    """Gera um limpador em dir temporario, py_compila e importa como modulo."""
    pasta = Path(tempfile.mkdtemp(prefix="limpador_teste_"))
    caminho = pasta / f"{nome}.py"
    empacotar.escrever_limpador(
        caminho=caminho, dataset="teste",
        detectores_codigo=detectores, plano_correcao=plano, colunas=colunas,
    )
    py_compile.compile(str(caminho), doraise=True)  # levanta PyCompileError se nao compilar
    spec = importlib.util.spec_from_file_location(nome, caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, caminho


def test_gerar_limpador_compila_importa_e_aplica_codigo_e_fd():
    detectores = {
        "ounces": "def detectar(col):\n    return col.str.endswith('oz')\n",
        "abv": "def detectar(col):\n    return col.str.contains('%', regex=False, na=False)\n",
        "state": "def detectar(col):\n    return col.eq('')\n",
    }
    plano = {
        "ounces": [{"tipo": "codigo",
                    "codigo": "def corrigir(valor):\n    return re.sub(r'\\s*oz$', '', valor)\n"}],
        "abv": [],  # detecta mas nao tem corretor/FD -> vira flag
        "state": [{"tipo": "fd", "determinante": "brewery", "dependente": "state"}],
    }
    colunas = ["ounces", "abv", "state"]
    mod, caminho = _gerar_e_importar("limpador_basico", detectores, plano, colunas)
    assert caminho.exists()

    df = pd.DataFrame({
        "brewery": ["A", "A", "B", "B"],
        "ounces": ["12 oz", "16", "10 oz", "20"],
        "abv": ["0.05", "0.06%", "0.07", "0.08%"],
        "state": ["", "CA", "NY", ""],
    })
    corrigido, flags = mod.aplicar(df)

    assert list(corrigido["ounces"]) == ["12", "16", "10", "20"]
    assert list(corrigido["state"]) == ["CA", "CA", "NY", "NY"]
    assert list(corrigido["abv"]) == ["0.05", "0.06%", "0.07", "0.08%"]
    assert list(flags["abv"]) == [False, True, False, True]
    assert not flags["ounces"].any() and not flags["state"].any()
    assert list(corrigido["brewery"]) == ["A", "A", "B", "B"], "sem detector/plano fica intacto"


def test_fd_duplo_filtro_moda_exclui_o_dependente_marcado():
    detectores = {"state": "def detectar(col):\n    return col.eq('')\n"}
    plano = {"state": [{"tipo": "fd", "determinante": "brewery", "dependente": "state"}]}
    mod, _ = _gerar_e_importar("limpador_duplo", detectores, plano, ["state"])

    # brewery unica 'A'. state: 3 vazios MARCADOS + 2 'CA' limpos. SEM o filtro do
    # dependente a moda do grupo seria '' (3 vs 2) e nada seria corrigido; COM o
    # duplo-filtro o pool e' ['CA','CA'] -> 'CA'.
    df = pd.DataFrame({
        "brewery": ["A", "A", "A", "A", "A"],
        "state":   ["", "", "", "CA", "CA"],
    })
    corrigido, flags = mod.aplicar(df)
    assert list(corrigido["state"]) == ["CA", "CA", "CA", "CA", "CA"]
    assert not flags["state"].any()


def test_coocorrencia_codigo_e_fd_na_mesma_coluna_ambos_na_ordem():
    detectores = {
        "city": "def detectar(col):\n"
                "    return col.str.contains(r' [A-Z]{2}$', regex=True, na=False) | col.eq('')\n",
    }
    plano = {
        "city": [
            {"tipo": "codigo",
             "codigo": "def corrigir(valor):\n    return re.sub(r'\\s*[A-Z]{2}$', '', valor)\n"},
            {"tipo": "fd", "determinante": "brewery", "dependente": "city"},
        ],
    }
    mod, _ = _gerar_e_importar("limpador_coocorr", detectores, plano, ["city"])

    df = pd.DataFrame({
        "brewery": ["A", "A", "A", "B", "B"],
        "city":    ["Portland OR", "Salem", "", "Austin", ""],
    })
    corrigido, flags = mod.aplicar(df)
    # idx0 'Portland OR' -> CODIGO strip ' OR' -> 'Portland' (a FD sozinha daria
    # 'Salem'; prova que o codigo rodou). idx2/idx4 vazios escalam -> FD.
    assert corrigido["city"].iloc[0] == "Portland"
    assert corrigido["city"].iloc[2] == "Salem" and corrigido["city"].iloc[4] == "Austin"
    assert list(corrigido["city"]) == ["Portland", "Salem", "Salem", "Austin", "Austin"]
    assert not flags["city"].any()


def test_fd_le_o_df_de_entrada_nao_a_copia_ja_corrigida():
    """ORDEM: a FD agrupa sobre o df ORIGINAL mesmo apos outra coluna ja corrigida na copia."""
    detectores = {
        "city": "def detectar(col):\n    return col.str.contains(r' [A-Z]{2}$', regex=True, na=False)\n",
        "state": "def detectar(col):\n    return col.eq('')\n",
    }
    plano = {
        "city": [{"tipo": "codigo",
                  "codigo": "def corrigir(valor):\n    return re.sub(r'\\s*[A-Z]{2}$', '', valor)\n"}],
        "state": [{"tipo": "fd", "determinante": "city", "dependente": "state"}],
    }
    # 'city' processada ANTES de 'state'; a FD de state usa city como determinante.
    mod, _ = _gerar_e_importar("limpador_ordem", detectores, plano, ["city", "state"])

    df = pd.DataFrame({
        "city":  ["Boston MA", "Boston", "Boston", "Boston"],
        "state": ["", "MA", "MA", "MA"],
    })
    corrigido, flags = mod.aplicar(df)
    # city[0] corrigida -> 'Boston'. Se a FD lesse a copia (city ja 'Boston'), o
    # pool seria idx1-3 -> moda 'MA' -> bug. Lendo o df ORIGINAL ('Boston MA'),
    # nenhuma linha nao-marcada bate -> pool vazio -> state[0] fica FLAG.
    assert corrigido["city"].iloc[0] == "Boston"
    assert corrigido["state"].iloc[0] == "" and bool(flags["state"].iloc[0]) is True


def test_determinante_sem_detector_mascara_tudo_false_fd_aplica():
    detectores = {"state": "def detectar(col):\n    return col.eq('')\n"}
    plano = {"state": [{"tipo": "fd", "determinante": "brewery", "dependente": "state"}]}
    # 'brewery' NAO esta em colunas -> sem detector -> mask[brewery] tudo-False.
    mod, _ = _gerar_e_importar("limpador_det_sem_det", detectores, plano, ["state"])

    df = pd.DataFrame({
        "brewery": ["A", "A", "B", "B"],
        "state":   ["", "CA", "NY", ""],
    })
    corrigido, flags = mod.aplicar(df)
    assert "brewery" not in mod._DETECTORES
    assert list(corrigido["state"]) == ["CA", "CA", "NY", "NY"]
    assert not flags["state"].any()
