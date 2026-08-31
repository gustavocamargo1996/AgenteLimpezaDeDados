"""A espinha monta um Trabalho por coluna, na ordem das etapas."""
import py_compile
import types
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.runnables import RunnableLambda

from limpeza import amostragem, config, deteccao, pipeline
from limpeza.correcao import cascata, fd as fd_mod
from limpeza.esquemas import RegraDeteccao
from limpeza.tipos import Amostra, Coluna, Trabalho

FIXTURES = Path(__file__).parent / "fixtures"

DETECTA_OZ = "def detectar(col):\n    return col.str.endswith('oz')\n"
CORRIGE_OZ = "def corrigir(valor):\n    return valor.split()[0].removesuffix('.0')\n"


def _coluna(nome="abv"):
    serie = pd.Series(["0.05", "0.06%"])
    return Coluna(nome=nome, sujo=serie, limpo=serie,
                  valores_distintos=["0.05", "0.06%"], contagem={"0.05": 1, "0.06%": 1})


def _amostra():
    return Amostra(representantes=["0.05"], linhas={0}, rotulados=[],
                   total_linhas=2, total_distintos=2)


class _EmbedderFalso:
    """Vetores deterministicos por hash: dispensa o ONNX nos testes."""

    def codificar(self, textos, lote=64):
        matriz = np.zeros((len(textos), 8), dtype=np.float32)
        for i, texto in enumerate(textos):
            rng = np.random.default_rng(zlib.crc32(texto.encode()))
            matriz[i] = rng.random(8)
        normas = np.clip(np.linalg.norm(matriz, axis=1, keepdims=True), 1e-9, None)
        return matriz / normas


def test_trabalho_nasce_sem_correcao_nem_medida():
    t = Trabalho(coluna=_coluna(), amostra=None, detector=None)
    assert t.correcao is None
    assert t.medida is None


def test_coluna_que_falha_na_deteccao_vira_detector_nulo(monkeypatch):
    """A unica fronteira de resiliencia: a coluna nao derruba o run."""
    def explodir(*_a, **_k):
        raise RuntimeError("agente fora do ar")

    monkeypatch.setattr(amostragem, "representantes", lambda coluna: _amostra())
    monkeypatch.setattr(deteccao, "gerar_regra_deteccao", explodir)

    trabalho = pipeline._processar_coluna(_coluna(), {"deteccao": None})

    assert trabalho.detector.codigo == deteccao.DETECTA_NADA
    assert trabalho.detector.funcao(_coluna().sujo).tolist() == [False, False]
    assert trabalho.correcao is None


def _fingir_agentes(monkeypatch):
    """Troca todo LLM da espinha por resposta fixa, para rodar o run offline."""
    def regra_oz(_entrada):
        return RegraDeteccao(erro_provavel=True, condicao_regex=r"\s*oz$",
                             codigo=DETECTA_OZ, cadeia="o sufixo ' oz' marca o erro")

    def camada_codigo(_coluna, _rotulados, _agentes):
        regra = types.SimpleNamespace(
            transformacao=types.SimpleNamespace(tipo="remover_sufixo"),
            descricao_padrao="sufixo de unidade colado no numero")
        return regra, lambda valor: valor.split()[0].removesuffix(".0"), CORRIGE_OZ

    monkeypatch.setattr(amostragem, "embedder", _EmbedderFalso)
    monkeypatch.setattr(pipeline, "_construir_agentes",
                        lambda: {"deteccao": RunnableLambda(regra_oz)})
    monkeypatch.setattr(cascata, "_camada_codigo", camada_codigo)
    monkeypatch.setattr(fd_mod, "candidatos_determinantes",
                        lambda *_a, **_k: [])
    monkeypatch.setattr(config, "ITERACOES_DETECCAO", 1)


def test_espinha_percorre_as_etapas_e_escreve_os_artefatos(tmp_path, monkeypatch):
    """Run offline completo: da carga ao limpador empacotado."""
    _fingir_agentes(monkeypatch)

    limpador, medida = pipeline.gerar_limpador(
        caminho_sujo=FIXTURES / "beers_dirty_300.csv",
        caminho_limpo=FIXTURES / "beers_clean_300.csv",
        colunas=["ounces"],
        saida=tmp_path,
    )

    esperados = ["mascara.csv", "correcoes.csv", "deteccao_metricas.json",
                 "correcao_metricas.json", "cascata.md", "cadeias_deteccao.md"]
    assert [n for n in esperados if not (tmp_path / n).exists()] == []
    py_compile.compile(str(limpador), doraise=True)

    # O sufixo " oz" so' aparece em celula errada, e o corretor so' o remove:
    # a marcacao e' precisa e a correcao nao estraga celula ja limpa.
    assert medida["deteccao"]["ounces"]["precisao"] == 1.0
    assert medida["deteccao"]["ounces"]["recall"] > 0
    assert medida["correcao"]["ounces"]["taxa_acerto"] > 0
    assert medida["correcao"]["ounces"]["taxa_dano"] == 0.0


def test_espinha_absorve_os_dicionarios_paralelos_num_trabalho(tmp_path, monkeypatch):
    """Cada coluna sai com detector, correcao e medida no MESMO objeto."""
    _fingir_agentes(monkeypatch)
    vistos = []
    original = pipeline.correcao.rodar_cascata

    def espiar(trabalhos, tabela, mascara, agentes):
        vistos.extend(trabalhos)
        return original(trabalhos, tabela, mascara, agentes)

    monkeypatch.setattr(pipeline.correcao, "rodar_cascata", espiar)
    pipeline.gerar_limpador(
        caminho_sujo=FIXTURES / "beers_dirty_300.csv",
        caminho_limpo=FIXTURES / "beers_clean_300.csv",
        colunas=["ounces", "city"],
        saida=tmp_path,
    )

    assert [t.coluna.nome for t in vistos] == ["ounces", "city"]
    for trabalho in vistos:
        trilha = trabalho.correcao.trilha
        contagem = trilha["contagem"]
        assert trabalho.detector.codigo == DETECTA_OZ.strip()
        assert trilha["coluna"] == trabalho.coluna.nome
        assert sum(contagem.values()) == trilha["marcadas"]
        assert set(trabalho.medida) == {"deteccao", "correcao"}
