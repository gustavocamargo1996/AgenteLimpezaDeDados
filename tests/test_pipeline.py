"""A espinha monta um Trabalho por coluna, na ordem das etapas."""
import py_compile
import types
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from langchain_core.runnables import RunnableLambda

from limpeza import amostragem, config, deteccao, pipeline
from limpeza.correcao import cascata, fd as fd_mod
from limpeza.esquemas import RegraDeteccao
from limpeza.tipos import Amostra, Coluna, Tabela, Trabalho

import main as cli

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
                        lambda: {"deteccao": RunnableLambda(regra_oz), "fd": None})
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
        # ITERACOES_DETECCAO=1 (via _fingir_agentes): 1-passe, sem refino, codigo sem strip.
        assert trabalho.detector.codigo == DETECTA_OZ
        assert trilha["coluna"] == trabalho.coluna.nome
        assert sum(contagem.values()) == trilha["marcadas"]
        assert set(trabalho.medida) == {"deteccao", "correcao"}


def test_beers_nao_ganha_nenhuma_marca_de_fd():
    """O beers nao tem erro cross-column; uma marca sequer e' vazamento."""
    from limpeza.deteccao import dependencia
    from limpeza.esquemas import DependenciaFuncional

    LER = dict(dtype=str, keep_default_na=False, na_values=[])
    sujo = pd.read_csv(FIXTURES / "beers_dirty_300.csv", **LER)
    limpo = pd.read_csv(FIXTURES / "beers_clean_300.csv", **LER)

    col = Coluna(nome="state", sujo=sujo["state"], limpo=limpo["state"],
                valores_distintos=sorted(sujo["state"].unique()),
                contagem=sujo["state"].value_counts().to_dict())
    tabela = Tabela(sujo=sujo, limpo=limpo, nome="beers", colunas=[col])
    linhas = list(range(10))
    rot = [{"indice": i, "sujo": col.sujo.at[i], "limpo": col.limpo.at[i],
            "eh_erro": col.sujo.at[i] != col.limpo.at[i]} for i in linhas]
    trabalho = Trabalho(coluna=col, detector=None,
                        amostra=Amostra(representantes=[], linhas=set(linhas),
                                        rotulados=rot, total_linhas=len(sujo),
                                        total_distintos=0))

    class _Agente:
        def invoke(self, _a):
            return self(_a)

        # __call__ (alem de invoke) deixa `prompt | agente` coercer este duble,
        # senao o proprio TypeError da coercao mascara o teste (vira "0 marcas" sempre).
        def __call__(self, _a):
            return DependenciaFuncional(determinante="brewery-name", dependente="state",
                                        justificativa="cervejaria fixa o estado")

    zerada = pd.DataFrame(0, index=sujo.index, columns=sujo.columns, dtype=int)
    m = dependencia.detectar_dependencia([trabalho], tabela, zerada, _Agente())
    marcadas = int(m["state"].sum())
    erradas = int((sujo["state"] != limpo["state"]).sum())
    assert marcadas <= erradas, (
        f"marcou {marcadas} celulas em state, mas so' {erradas} estao erradas: "
        "a FD esta marcando celula correta")


def test_iteracoes_deteccao_1_nao_refina(monkeypatch):
    """Gate de _processar_coluna: N=1 e' 1-passe puro, o refino nunca e' chamado."""
    from limpeza.tipos import Detector

    de_1_passe = Detector(codigo=DETECTA_OZ, funcao=lambda col: col.str.endswith("oz"),
                          cadeia="sufixo oz")
    chamadas = []

    monkeypatch.setattr(amostragem, "representantes", lambda coluna: _amostra())
    monkeypatch.setattr(config, "ITERACOES_DETECCAO", 1)
    monkeypatch.setattr(deteccao, "gerar_regra_deteccao", lambda *_a, **_k: de_1_passe)
    monkeypatch.setattr(deteccao, "refinar_regra_deteccao",
                        lambda *_a, **_k: chamadas.append(1) or de_1_passe)

    trabalho = pipeline._processar_coluna(_coluna("ounces"), {"deteccao": None})

    assert chamadas == []
    assert trabalho.detector is de_1_passe
    assert trabalho.detector.codigo == DETECTA_OZ


def test_iteracoes_deteccao_maior_que_1_refina(monkeypatch):
    """Gate de _processar_coluna: N>1 aciona o loop de refino sobre a regra de 1-passe."""
    from limpeza.tipos import Detector

    de_1_passe = Detector(codigo=DETECTA_OZ, funcao=lambda col: col.str.endswith("oz"),
                          cadeia="sufixo oz")
    refinado = Detector(codigo=DETECTA_OZ.strip(), funcao=de_1_passe.funcao,
                        cadeia="sufixo oz, refinado")
    chamadas = []

    def refinar(*_a, **_k):
        chamadas.append(1)
        return refinado

    monkeypatch.setattr(amostragem, "representantes", lambda coluna: _amostra())
    monkeypatch.setattr(config, "ITERACOES_DETECCAO", 2)
    monkeypatch.setattr(deteccao, "gerar_regra_deteccao", lambda *_a, **_k: de_1_passe)
    monkeypatch.setattr(deteccao, "refinar_regra_deteccao", refinar)

    trabalho = pipeline._processar_coluna(_coluna("ounces"), {"deteccao": None})

    assert chamadas == [1]
    assert trabalho.detector is refinado
    assert trabalho.detector.codigo == DETECTA_OZ.strip()


def test_refino_que_falha_mantem_a_regra_de_1_passe(monkeypatch):
    """A segunda politica da fronteira: a regra que ja passou no portao sobrevive."""
    from limpeza.tipos import Detector

    de_1_passe = Detector(codigo=DETECTA_OZ, funcao=lambda col: col.str.endswith("oz"),
                          cadeia="sufixo oz")

    def explodir(*_a, **_k):
        raise RuntimeError("oraculo fora do ar")

    monkeypatch.setattr(amostragem, "representantes", lambda coluna: _amostra())
    monkeypatch.setattr(deteccao, "gerar_regra_deteccao",
                        lambda *_a, **_k: de_1_passe)
    monkeypatch.setattr(deteccao, "refinar_regra_deteccao", explodir)

    trabalho = pipeline._processar_coluna(_coluna("ounces"), {"deteccao": None})

    assert trabalho.detector is de_1_passe
    assert trabalho.detector.codigo == DETECTA_OZ


def test_lista_de_colunas_vazia_nao_vira_o_dataset_inteiro():
    """Lista vazia pede nenhuma coluna; so' None quer dizer 'todas'."""
    from limpeza import dados

    comum = dict(caminho_sujo=FIXTURES / "beers_dirty_300.csv",
                 caminho_limpo=FIXTURES / "beers_clean_300.csv")
    assert dados.carregar(colunas=[], **comum).colunas == []
    assert len(dados.carregar(colunas=None, **comum).colunas) > 1


def test_coluna_inexistente_levanta_erro_de_argumento():
    """O erro de argumento tem tipo proprio, para a CLI nao capturar ValueError largo."""
    from limpeza import dados

    assert issubclass(dados.DadosInvalidos, ValueError)
    with pytest.raises(dados.DadosInvalidos, match="inexistente"):
        dados.carregar(caminho_sujo=FIXTURES / "beers_dirty_300.csv",
                       caminho_limpo=FIXTURES / "beers_clean_300.csv",
                       colunas=["nao_existe"])


def test_carregar_aceita_caminhos_diretos():
    """A entrada e' o caminho do CSV, sem catalogo por nome de dataset."""
    from limpeza import dados

    tabela = dados.carregar(FIXTURES / "beers_dirty_300.csv",
                            FIXTURES / "beers_clean_300.csv")
    assert len(tabela.sujo) == 300
    assert list(tabela.sujo.columns) == list(tabela.limpo.columns)
    assert tabela.nome == "beers_300"


def test_caminho_sujo_inexistente_levanta_dados_invalidos():
    """Caminho que nao existe e' sinalizado com o mesmo tipo de erro, nao um traceback cru."""
    from limpeza import dados

    with pytest.raises(dados.DadosInvalidos, match="nao encontrado"):
        dados.carregar(FIXTURES / "nao_existe.csv", FIXTURES / "beers_clean_300.csv")


def test_sujo_e_limpo_saem_do_ambiente(monkeypatch):
    """As variaveis SUJO e LIMPO suprem os argumentos --sujo e --limpo."""
    monkeypatch.setenv("SUJO", "/dados/a_dirty.csv")
    monkeypatch.setenv("LIMPO", "/dados/a_clean.csv")
    args = cli._argumentos([])
    assert args.sujo == "/dados/a_dirty.csv"
    assert args.limpo == "/dados/a_clean.csv"


def test_argumento_vence_a_variavel(monkeypatch):
    """Argumento explicito supera a variavel de ambiente."""
    monkeypatch.setenv("SUJO", "/dados/do_ambiente.csv")
    args = cli._argumentos(["--sujo", "/dados/do_argumento.csv",
                            "--limpo", "/dados/b.csv"])
    assert args.sujo == "/dados/do_argumento.csv"


def test_sem_argumento_e_sem_variavel_e_erro(monkeypatch):
    """Falta de argumento e variavel gera erro com saida."""
    monkeypatch.delenv("SUJO", raising=False)
    monkeypatch.delenv("LIMPO", raising=False)
    with pytest.raises(SystemExit):
        cli._argumentos([])


def test_saida_sai_do_ambiente(monkeypatch):
    """A variavel SAIDA supre o argumento --saida."""
    monkeypatch.setenv("SUJO", "/d/a.csv")
    monkeypatch.setenv("LIMPO", "/d/b.csv")
    monkeypatch.setenv("SAIDA", "/dados/runs")
    assert cli._argumentos([]).saida == "/dados/runs"


def test_construir_agentes_respeita_a_sobreposicao_por_papel(monkeypatch):
    """MODELO_<PAPEL> chega ao agente daquele papel; os outros ficam no geral."""
    from limpeza import llm

    registrado = {}

    class _ClienteFalso:
        def with_structured_output(self, schema):
            return schema

    def _espiao(provedor, modelo, segundos):
        registrado.setdefault(modelo, 0)
        registrado[modelo] += 1
        return _ClienteFalso()

    monkeypatch.setattr(config, "MODELO_LLM", "geral")
    monkeypatch.setattr(config, "MODELOS_POR_PAPEL", {"deteccao": "X"})
    monkeypatch.setattr(config, "TIMEOUT_LLM", None)
    monkeypatch.setattr(llm, "cliente", _espiao)

    pipeline._construir_agentes()

    assert registrado == {"X": 1, "geral": 3}


def test_sem_modelo_no_cli_a_sobreposicao_por_papel_sobrevive(monkeypatch):
    """--modelo ausente nao pode sobrescrever MODELO_LLM: era o bug do nivel 1."""
    monkeypatch.setattr(config, "MODELO_LLM", "do-ambiente")
    args = cli._argumentos(["--sujo", "/d/a.csv", "--limpo", "/d/b.csv"])
    assert args.modelo is None
    cli._aplicar_config(args)
    assert config.MODELO_LLM == "do-ambiente"


def test_modelo_no_cli_vence_tudo(monkeypatch):
    """Com --modelo explicito, o nivel 1 da precedencia dispara de proposito."""
    monkeypatch.setattr(config, "MODELO_LLM", "do-ambiente")
    args = cli._argumentos(["--sujo", "/d/a.csv", "--limpo", "/d/b.csv",
                            "--modelo", "do-cli"])
    cli._aplicar_config(args)
    assert config.MODELO_LLM == "do-cli"


def test_avisa_quando_nenhuma_coluna_recebeu_regra(capsys):
    """LLM fora do ar termina com exit 0; o aviso e' a unica janela do usuario."""
    trabalhos = [Trabalho(coluna=_coluna("a"), amostra=None,
                          detector=deteccao.detector_nulo("caiu")),
                 Trabalho(coluna=_coluna("b"), amostra=None,
                          detector=deteccao.detector_nulo("caiu"))]
    pipeline._avisar_limpador_vazio(trabalhos)
    saida = capsys.readouterr().out
    assert "AVISO: 0/2 colunas receberam regra de deteccao" in saida


def test_nao_avisa_quando_ao_menos_uma_coluna_tem_regra(capsys):
    """Uma regra viva basta: o limpador nao esta vazio."""
    from limpeza.tipos import Detector

    trabalhos = [Trabalho(coluna=_coluna("a"), amostra=None,
                          detector=deteccao.detector_nulo("caiu")),
                 Trabalho(coluna=_coluna("b"), amostra=None,
                          detector=Detector(codigo=DETECTA_OZ, funcao=None,
                                            cadeia="sufixo oz"))]
    pipeline._avisar_limpador_vazio(trabalhos)
    assert "AVISO" not in capsys.readouterr().out


def test_log_de_excecao_traz_tipo_e_mensagem():
    """Com Ollama, '404' e 'connection refused' nao podem virar a mesma linha."""
    from limpeza import erros

    assert erros.descrever(RuntimeError("404 model not found")) == \
        "RuntimeError: 404 model not found"
    longa = erros.descrever(ValueError("x" * 500))
    assert longa.startswith("ValueError: x") and longa.endswith("...")
    assert len(longa) < 260
    assert erros.descrever(RuntimeError()) == "RuntimeError"


def test_deteccao_que_falha_registra_a_mensagem_na_cadeia(monkeypatch):
    """O motivo do detector_nulo tambem carrega a mensagem, nao so' o tipo."""
    def explodir(*_a, **_k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(amostragem, "representantes", lambda coluna: _amostra())
    monkeypatch.setattr(deteccao, "gerar_regra_deteccao", explodir)

    trabalho = pipeline._processar_coluna(_coluna(), {"deteccao": None})

    assert "connection refused" in trabalho.detector.cadeia
