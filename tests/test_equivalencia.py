"""A mascara do limpador gerado e' a mesma da POC, celula por celula."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from limpeza import deteccao, empacotar
from limpeza.correcao import fd as fd_mod
from limpeza.esquemas import DependenciaFuncional
from limpeza.tipos import Amostra, Coluna, Detector, Tabela, Trabalho

FIXTURES = Path(__file__).parent / "fixtures"
LER = dict(dtype=str, keep_default_na=False, na_values=[])
NADA = "def detectar(col):\n    return col.isin([])\n"
ENV_FDS = {"State": ("City", "State"), "Climate_Zone": ("City", "Climate_Zone")}


def _environment():
    return pd.read_csv(FIXTURES / "environment_dirty_300.csv", **LER)


def _encadeado():
    # City -> State marca as linhas 5-7; em State -> Country elas mudam a moda de S1.
    return pd.DataFrame({
        "City": ["X", "Y", "Y", "Y", "Y", "Y", "Y", "Y"],
        "State": ["S1", "S3", "S3", "S3", "S3", "S1", "S1", "S1"],
        "Country": ["C1", "C3", "C3", "C3", "C3", "C2", "C2", "C2"],
    })


def _empate():
    return pd.DataFrame({"City": ["Bangalore", "Bangalore"], "State": ["KA", "AA"]})


CENARIOS = {
    "intra_vazia": (_environment, {}, ENV_FDS),
    "intra_realista": (_environment,
                       {"City": "def detectar(col):\n    return col.eq('Hamburg')\n"},
                       ENV_FDS),
    "empate": (_empate, {}, {"State": ("City", "State")}),
    "encadeadas": (_encadeado, {},
                   {"State": ("City", "State"), "Country": ("State", "Country")}),
}


def _trabalhos(df, detectores, fds):
    saida = []
    for nome in df.columns:
        codigo = detectores.get(nome, NADA)
        det = fds.get(nome)
        saida.append(Trabalho(
            coluna=Coluna(nome=nome, sujo=df[nome], limpo=df[nome],
                          valores_distintos=sorted(df[nome].unique()),
                          contagem=df[nome].value_counts().to_dict()),
            amostra=Amostra(representantes=[], linhas=set(), rotulados=[],
                            total_linhas=len(df), total_distintos=0),
            detector=Detector(codigo=codigo, funcao=deteccao.materializar(codigo),
                              cadeia=""),
            dependencia=DependenciaFuncional(determinante=det[0], dependente=nome,
                                             justificativa="teste") if det else None,
        ))
    return saida


def _mascara_poc(df, detectores, fds, monkeypatch):
    """O caminho do pipeline: intra, depois FD sobre a intra, depois OU."""
    monkeypatch.setattr(fd_mod, "candidatos_determinantes",
                        lambda _df, alvo, limiar=None: [fds[alvo][0]] if alvo in fds else [])
    monkeypatch.setattr(fd_mod, "propor_fd", lambda dep, *_a, **_k: DependenciaFuncional(
        determinante=fds[dep][0], dependente=dep, justificativa="teste"))
    monkeypatch.setattr(fd_mod, "validar_fd", lambda *_a, **_k: True)
    trabalhos = _trabalhos(df, detectores, fds)
    tabela = Tabela(sujo=df, limpo=df, nome="teste", colunas=[t.coluna for t in trabalhos])
    intra = deteccao.construir_mascara(trabalhos, tabela)
    fd = deteccao.detectar_dependencia(trabalhos, tabela, intra, None)
    return (intra + fd) > 0


def _mascara_limpador(df, detectores, fds, pasta):
    """O caminho do produto: gera o limpador de verdade e chama a _mascara dele."""
    caminho = empacotar.gerar_limpador(
        _trabalhos(df, detectores, fds), pasta / "limpador_eq.py", "teste")
    spec = importlib.util.spec_from_file_location("limpador_eq", caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._mascara(df)


@pytest.mark.parametrize("nome", list(CENARIOS))
def test_mascara_do_limpador_e_a_da_poc(nome, monkeypatch, tmp_path):
    fabrica, detectores, fds = CENARIOS[nome]
    df = fabrica()
    poc = _mascara_poc(df, detectores, fds, monkeypatch)
    limpador = _mascara_limpador(df, detectores, fds, tmp_path)
    pd.testing.assert_frame_equal(limpador, poc, check_dtype=False)


def test_cenarios_discriminam(monkeypatch):
    """Sem isto os cenarios podem passar por coincidencia, nao por equivalencia."""
    vazia = _mascara_poc(_environment(), {}, ENV_FDS, monkeypatch)
    realista = _mascara_poc(_environment(), CENARIOS["intra_realista"][1], ENV_FDS,
                            monkeypatch)
    assert int(vazia["State"].sum()) == 54
    assert int(realista["State"].sum()) == 42
    empate = _mascara_poc(_empate(), {}, {"State": ("City", "State")}, monkeypatch)
    assert empate["State"].tolist() == [True, False]
    encadeada = _mascara_poc(_encadeado(), {}, CENARIOS["encadeadas"][2], monkeypatch)
    assert encadeada["Country"].tolist() == [True] + [False] * 7
