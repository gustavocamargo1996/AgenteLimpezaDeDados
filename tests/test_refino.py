"""Loop de refinamento da deteccao com oraculo: amostragem, feedback, guarda/piso, e a cegueira ao clean."""
import inspect
import io
import tokenize

import numpy as np
import pandas as pd
import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from limpeza import config, deteccao, sandbox
from limpeza.deteccao import oraculo, refino, regra
from limpeza.esquemas import RegraDeteccao
from limpeza.tipos import Coluna


def _mat_det(codigo: str):
    """Materializa uma funcao `detectar(col)` no series_mode, como o modulo regra faz."""
    return sandbox.materializar(codigo, nome_funcao="detectar",
                                nome_argumento="col", series_mode=True)


class FakeEmbedder:
    """Embedder OFFLINE e deterministico -- nao carrega ONNX.

    Vetor por valor = (comprimento, soma dos ordinais). Basta para o
    farthest-point de `_amostrar_oraculo`/`_selecionar_suspeito` ser estavel.
    """

    def codificar(self, textos):
        return np.array(
            [[float(len(t)), float(sum(ord(ch) for ch in t))] for t in textos]
        )


# --- _classificar (oraculo): aplicacao Series posicional, robusta a falha ---

def test_classificar_aplica_a_funcao_posicionalmente():
    f_oz = _mat_det("def detectar(col):\n    return col.str.endswith('oz')\n")
    marcas = oraculo._classificar(f_oz, ["12", "12 oz", "16", "16 oz"])
    assert marcas == [False, True, False, True]
    assert all(isinstance(x, bool) for x in marcas), "devolve python bool, nao numpy.bool_"


def test_classificar_funcao_none_devolve_falso_para_todos():
    assert oraculo._classificar(None, ["a", "b", "c"]) == [False, False, False]


def test_classificar_retorno_escalar_devolve_falso_para_todos():
    f_escalar = _mat_det("def detectar(col):\n    return True\n")
    assert oraculo._classificar(f_escalar, ["a", "b"]) == [False, False]


def test_classificar_excecao_em_runtime_devolve_falso_para_todos():
    def _f_boom(coluna):
        raise ValueError("boom")
    assert oraculo._classificar(_f_boom, ["a", "b"]) == [False, False]


def test_classificar_lista_vazia_devolve_vazio():
    f_oz = _mat_det("def detectar(col):\n    return col.str.endswith('oz')\n")
    assert oraculo._classificar(f_oz, []) == []


# --- _amostrar_oraculo: 1 previsto-sujo + 1 previsto-limpo, deterministico, respeita usados ---

def test_amostrar_oraculo_escolhe_um_sujo_e_um_limpo():
    emb = FakeEmbedder()
    funcao = _mat_det("def detectar(col):\n    return col.str.endswith('oz')\n")
    valores = ["12", "16", "12.0 oz", "16.0 oz"]
    sujo_col = pd.Series(valores)

    esc = oraculo._amostrar_oraculo(funcao, valores, emb, usados=set(), n=2, sujo_col=sujo_col)
    sujo = [v for v in esc if v.endswith("oz")]
    limpo = [v for v in esc if not v.endswith("oz")]
    assert len(esc) == 2 and len(sujo) == 1 and len(limpo) == 1


def test_amostrar_oraculo_e_deterministico():
    emb = FakeEmbedder()
    funcao = _mat_det("def detectar(col):\n    return col.str.endswith('oz')\n")
    valores = ["12", "16", "12.0 oz", "16.0 oz"]
    sujo_col = pd.Series(valores)

    esc1 = oraculo._amostrar_oraculo(funcao, valores, emb, usados=set(), n=2, sujo_col=sujo_col)
    esc2 = oraculo._amostrar_oraculo(funcao, valores, emb, usados=set(), n=2, sujo_col=sujo_col)
    assert esc1 == esc2


def test_amostrar_oraculo_respeita_usados():
    emb = FakeEmbedder()
    funcao = _mat_det("def detectar(col):\n    return col.str.endswith('oz')\n")
    valores = ["12", "16", "12.0 oz", "16.0 oz"]
    sujo_col = pd.Series(valores)

    esc3 = oraculo._amostrar_oraculo(funcao, valores, emb, usados={"12.0 oz"}, n=2, sujo_col=sujo_col)
    assert "12.0 oz" not in esc3 and any(v.endswith("oz") for v in esc3)


# --- _montar_feedback_det: marca FP/FN e e' cumulativo ---

def test_montar_feedback_marca_falso_positivo_e_falso_negativo():
    funcao = _mat_det("def detectar(col):\n    return col.str.endswith('oz')\n")
    # '12.0 oz': regra marca SUJO, oraculo diz correto -> FALSO POSITIVO.
    # '17'    : regra deixa LIMPO, oraculo diz erro     -> FALSO NEGATIVO.
    rotulos = [
        {"valor": "12.0 oz", "eh_erro_real": False},
        {"valor": "17", "eh_erro_real": True},
    ]
    texto = refino._montar_feedback_det(funcao, rotulos)
    linha_fp = next(l for l in texto.splitlines() if '"12.0 oz"' in l)
    linha_fn = next(l for l in texto.splitlines() if '"17"' in l)
    assert "FALSO POSITIVO" in linha_fp
    assert "FALSO NEGATIVO" in linha_fn


def test_montar_feedback_e_cumulativo():
    funcao = _mat_det("def detectar(col):\n    return col.str.endswith('oz')\n")
    rotulos = [
        {"valor": "12.0 oz", "eh_erro_real": False},
        {"valor": "17", "eh_erro_real": True},
        {"valor": "20.0 oz", "eh_erro_real": True},
    ]
    texto2 = refino._montar_feedback_det(funcao, rotulos)
    assert '"12.0 oz"' in texto2 and '"17"' in texto2 and '"20.0 oz"' in texto2
    assert len(texto2.splitlines()) == 3


def _coluna_teste(nome, sujo_col, limpo_col):
    """Coluna sintetica para os testes do loop de refinamento."""
    return Coluna(nome=nome, sujo=sujo_col, limpo=limpo_col,
                  valores_distintos=sorted(sujo_col.unique().tolist()),
                  contagem=sujo_col.value_counts().to_dict())


def _refinar(coluna, detector, agente, emb, iteracoes=1, amostras=2, monkeypatch=None):
    """Roda o refinamento com o orcamento do teste fixado no config."""
    if monkeypatch is None:
        config.ITERACOES_DETECCAO, config.AMOSTRAS_POR_ITERACAO = iteracoes, amostras
        return deteccao.refinar_regra_deteccao(detector, coluna, agente, emb=emb)
    monkeypatch.setattr(config, "ITERACOES_DETECCAO", iteracoes)
    monkeypatch.setattr(config, "AMOSTRAS_POR_ITERACAO", amostras)
    return deteccao.refinar_regra_deteccao(detector, coluna, agente, emb=emb)


# --- refinar_regra_deteccao com agente FAKE: sucesso troca a funcao, rejeicao mantem ---

def test_refinar_deteccao_sucesso_troca_a_funcao(monkeypatch):
    emb = FakeEmbedder()
    sujo_col = pd.Series(["12.0 oz", "16.0 oz", "12", "16"])
    limpo_col = pd.Series(["12", "16", "12", "16"])
    col = _coluna_teste("ounces", sujo_col, limpo_col)

    funcao0 = _mat_det(deteccao.DETECTA_NADA)
    regra0 = RegraDeteccao(erro_provavel=False, condicao_regex=None,
                           codigo=deteccao.DETECTA_NADA, cadeia="parte de col.isin([])")

    def fake_ok(_prompt_value):
        return RegraDeteccao(
            erro_provavel=True, condicao_regex=r"\s*oz$",
            codigo="def detectar(col):\n    return col.str.endswith('oz')\n",
            cadeia="o feedback disse que '12.0 oz' e' erro; flago o sufixo oz",
        )

    det = _refinar(col, regra.detector_de(regra0, funcao0),
                   RunnableLambda(fake_ok), emb, monkeypatch=monkeypatch)
    f = det.funcao
    assert f is not funcao0
    assert oraculo._classificar(f, ["12.0 oz", "12"]) == [True, False]
    passo = det.historico[0]
    assert len(det.historico) == 1 and passo["mudou"] is True and passo["status"] == "aplicada"
    assert det.orcamento["n_valores"] >= 1 and det.orcamento["n_celulas"] >= 1


def test_refinar_deteccao_rejeicao_mantem_a_funcao_anterior(monkeypatch):
    emb = FakeEmbedder()
    sujo_col = pd.Series(["12.0 oz", "16.0 oz", "12", "16"])
    limpo_col = pd.Series(["12", "16", "12", "16"])
    col = _coluna_teste("ounces", sujo_col, limpo_col)

    funcao0 = _mat_det(deteccao.DETECTA_NADA)
    regra0 = RegraDeteccao(erro_provavel=False, condicao_regex=None,
                           codigo=deteccao.DETECTA_NADA, cadeia="parte de col.isin([])")

    # regex invalida (Series) -> falha no smoke, mantem a regra anterior.
    def fake_ruim(_prompt_value):
        return RegraDeteccao(
            erro_provavel=True, condicao_regex=None,
            codigo="def detectar(col):\n    return col.str.contains('(', regex=True, na=False)\n",
            cadeia="regex proposital invalida (paren sem fechar)",
        )

    det_r = _refinar(col, regra.detector_de(regra0, funcao0),
                     RunnableLambda(fake_ruim), emb, monkeypatch=monkeypatch)
    assert det_r.funcao is funcao0 and det_r.codigo == regra0.codigo
    passo_r = det_r.historico[0]
    assert passo_r["mudou"] is False and passo_r["status"].startswith("rejeitada")


# --- _selecionar_suspeito: raridade alnum, freq-de-celula, bonus x, guardas ---

def test_selecionar_suspeito_rankeia_char_raro_bonus_desligado():
    emb = FakeEmbedder()
    candidatos = ["10", "11", "12", "13", "12a"]
    sujo_col = pd.Series(["10", "11", "12", "13", "10", "11", "12", "13", "12a"])
    pick = oraculo._selecionar_suspeito(
        candidatos, referencia=set(), n=1, embedder=emb, sujo_col=sujo_col,
        usar_bonus_x=False,
    )
    assert pick == ["12a"]


def test_char_counts_pondera_por_frequencia_de_celula():
    # '9' esta em 1 valor distinto mas 10 celulas -> frequente -> raridade BAIXA;
    # '3' em 1 valor distinto e 1 celula -> raro -> raridade ALTA.
    col_freq = pd.Series(["9"] * 10 + ["3"])
    cc = oraculo._contar_chars_alnum(col_freq)
    assert cc.get("9") == 10 and cc.get("3") == 1

    s9 = oraculo._score_artefato("9", cc, usar_bonus_x=False)
    s3 = oraculo._score_artefato("3", cc, usar_bonus_x=False)
    assert s9 < s3 and abs(s9 - 0.1) < 1e-9 and abs(s3 - 1.0) < 1e-9


def test_bonus_x_ligado_sobe_o_score_de_valor_com_x():
    cc_x = oraculo._contar_chars_alnum(pd.Series(["1x2", "10", "20"]))
    on = oraculo._score_artefato("1x2", cc_x, usar_bonus_x=True)
    off = oraculo._score_artefato("1x2", cc_x, usar_bonus_x=False)
    assert on > off


def test_selecionar_suspeito_e_deterministico_e_respeita_referencia():
    emb = FakeEmbedder()
    cand2 = ["10", "20x", "30", "40"]
    col2 = pd.Series(["10", "20x", "30", "40", "99"])
    r1 = oraculo._selecionar_suspeito(cand2, referencia={"99"}, n=2, embedder=emb,
                                       sujo_col=col2, usar_bonus_x=False)
    r2 = oraculo._selecionar_suspeito(cand2, referencia={"99"}, n=2, embedder=emb,
                                       sujo_col=col2, usar_bonus_x=False)
    assert r1 == r2
    assert set(r1) <= set(cand2) and "99" not in r1


def test_selecionar_suspeito_guarda_de_div_zero_em_coluna_degenerada():
    """Coluna sem char alnum (artefato max=0) nao lanca, ainda devolve o pick por diversidade."""
    emb = FakeEmbedder()
    cand_deg = ["--", "++", "**"]
    col_deg = pd.Series(["--", "++", "**"])
    deg = oraculo._selecionar_suspeito(cand_deg, referencia=set(), n=1, embedder=emb,
                                       sujo_col=col_deg, usar_bonus_x=True)
    assert isinstance(deg, list) and len(deg) == 1


# --- _precisao_no_oraculo: precisao no rotulado + Armadilha 1 (cega sem clean) ---

def test_precisao_no_oraculo_marca_so_sujos_da_precisao_1():
    rotulos = [
        {"valor": "A", "eh_erro_real": True},
        {"valor": "B", "eh_erro_real": True},
        {"valor": "C", "eh_erro_real": False},
        {"valor": "D", "eh_erro_real": False},
    ]
    so_sujos = _mat_det("def detectar(col):\n    return col.isin(['A', 'B'])\n")
    assert refino._precisao_no_oraculo(so_sujos, rotulos) == 1.0


def test_precisao_no_oraculo_marca_2_limpos_1_sujo_da_0_333():
    rotulos = [
        {"valor": "A", "eh_erro_real": True},
        {"valor": "B", "eh_erro_real": True},
        {"valor": "C", "eh_erro_real": False},
        {"valor": "D", "eh_erro_real": False},
    ]
    ampla = _mat_det("def detectar(col):\n    return col.isin(['A', 'C', 'D'])\n")
    p2 = refino._precisao_no_oraculo(ampla, rotulos)
    assert p2 is not None and abs(p2 - 1 / 3) < 1e-9
    assert p2 < config.LIMITE_PRECISAO_DETECCAO


def test_precisao_no_oraculo_marca_zero_devolve_none():
    rotulos = [
        {"valor": "A", "eh_erro_real": True},
        {"valor": "B", "eh_erro_real": True},
        {"valor": "C", "eh_erro_real": False},
        {"valor": "D", "eh_erro_real": False},
    ]
    nada = _mat_det(deteccao.DETECTA_NADA)
    assert refino._precisao_no_oraculo(nada, rotulos) is None


def test_precisao_no_oraculo_armadilha_1_rotulos_todos_sujos_regra_ampla_nao_e_punida():
    """Cegueira DECLARADA: sem clean no oraculo, uma regra que marca tudo tem precisao 1.0
    quando todos os rotulos ja sao erro -- a guarda nao pode punir o que nao ve."""
    rotulos_so_sujos = [
        {"valor": "12.0 oz", "eh_erro_real": True},
        {"valor": "16.0 oz", "eh_erro_real": True},
    ]
    marca_tudo = _mat_det("def detectar(col):\n    return col.notna()\n")
    p_arm = refino._precisao_no_oraculo(marca_tudo, rotulos_so_sujos)
    assert p_arm == 1.0 and not (p_arm < config.LIMITE_PRECISAO_DETECCAO)


# --- guarda (A) rejeita ampla / aceita precisa; piso (B); ounces-like nao punido ---

def _novo_nada():
    return _mat_det(deteccao.DETECTA_NADA)


def _regra_nada():
    return RegraDeteccao(erro_provavel=False, condicao_regex=None,
                         codigo=deteccao.DETECTA_NADA, cadeia="col.isin([])")


def test_guarda_rejeita_regra_ampla(monkeypatch):
    emb = FakeEmbedder()
    sujo_col = pd.Series(["A", "B", "C", "D"])
    limpo_col = pd.Series(["A_ok", "B", "C", "D"])
    col = _coluna_teste("col", sujo_col, limpo_col)

    def fake_ampla(_prompt_value):
        return RegraDeteccao(
            erro_provavel=True, condicao_regex=None,
            codigo="def detectar(col):\n    return col.isin(['A', 'B', 'C'])\n",
            cadeia="regra ampla proposital: marca A, B e C (2 sao limpos)",
        )

    funcao0 = _novo_nada()
    regra0 = _regra_nada()
    det_rej = _refinar(col, regra.detector_de(regra0, funcao0),
                       RunnableLambda(fake_ampla), emb, amostras=4, monkeypatch=monkeypatch)
    assert det_rej.funcao is funcao0 and det_rej.codigo == regra0.codigo
    passo_rej = det_rej.historico[0]
    assert passo_rej["mudou"] is False
    assert passo_rej["status"].startswith("rejeitada por precisao")


def test_guarda_aceita_regra_precisa(monkeypatch):
    emb = FakeEmbedder()
    sujo_col = pd.Series(["A", "B", "C", "D"])
    limpo_col = pd.Series(["A_ok", "B", "C", "D"])
    col = _coluna_teste("col", sujo_col, limpo_col)

    def fake_precisa(_prompt_value):
        return RegraDeteccao(
            erro_provavel=True, condicao_regex=None,
            codigo="def detectar(col):\n    return col.eq('A')\n",
            cadeia="marca so' 'A', o unico erro real",
        )

    funcao0b = _novo_nada()
    det_ac = _refinar(col, regra.detector_de(_regra_nada(), funcao0b),
                      RunnableLambda(fake_precisa), emb, amostras=4, monkeypatch=monkeypatch)
    f_ac = det_ac.funcao
    assert f_ac is not funcao0b
    assert oraculo._classificar(f_ac, ["A", "B"]) == [True, False]
    assert det_ac.historico[0]["status"] == "aplicada"


def test_piso_final_derruba_resident_ampla_para_detecta_nada(monkeypatch):
    emb = FakeEmbedder()
    sujo_col = pd.Series(["A", "B", "C", "D"])
    limpo_col = pd.Series(["A_ok", "B", "C", "D"])
    col = _coluna_teste("col", sujo_col, limpo_col)

    def fake_ampla(_prompt_value):
        return RegraDeteccao(
            erro_provavel=True, condicao_regex=None,
            codigo="def detectar(col):\n    return col.isin(['A', 'B', 'C'])\n",
            cadeia="regra ampla proposital: marca A, B e C (2 sao limpos)",
        )

    ampla_cod = "def detectar(col):\n    return col.isin(['A', 'B', 'C'])\n"
    funcao0_ampla = _mat_det(ampla_cod)
    regra0_ampla = RegraDeteccao(erro_provavel=True, condicao_regex=None,
                                 codigo=ampla_cod, cadeia="resident ja ampla")
    det_piso = _refinar(col, regra.detector_de(regra0_ampla, funcao0_ampla),
                        RunnableLambda(fake_ampla), emb, amostras=4, monkeypatch=monkeypatch)
    f_piso = det_piso.funcao
    assert oraculo._classificar(f_piso, ["A", "B", "C"]) == [False, False, False]
    assert det_piso.codigo == deteccao.DETECTA_NADA

    entradas_piso = [p for p in det_piso.historico if p["status"].startswith("piso")]
    assert len(entradas_piso) == 1 and "iteracao" in entradas_piso[0]
    # Mesmas chaves das entradas do laco -- senao relatorio._bloco_refinamento quebraria ao vivo.
    chaves_laco = set(det_piso.historico[0].keys())
    assert set(entradas_piso[0].keys()) == chaves_laco


def test_ounces_like_regra_ampla_nao_punida_quando_todos_os_rotulos_sao_erro(monkeypatch):
    """Rotulos TODOS sujos; regra marca TODOS -> precisao 1.0 -> aceita, piso nao dispara."""
    emb = FakeEmbedder()
    sujo_oz = pd.Series(["12.0 oz", "16.0 oz", "20.0 oz"])
    limpo_oz = pd.Series(["12", "16", "20"])  # 100% erro real
    col_oz = _coluna_teste("ounces", sujo_oz, limpo_oz)

    def fake_oz(_prompt_value):
        return RegraDeteccao(
            erro_provavel=True, condicao_regex=r"\s*oz$",
            codigo="def detectar(col):\n    return col.str.endswith('oz')\n",
            cadeia="marca o sufixo oz (todos os rotulos sao erro)",
        )

    funcao0_oz = _novo_nada()
    det_oz = _refinar(col_oz, regra.detector_de(_regra_nada(), funcao0_oz),
                      RunnableLambda(fake_oz), emb, amostras=3, monkeypatch=monkeypatch)
    f_oz = det_oz.funcao
    assert f_oz is not funcao0_oz
    assert oraculo._classificar(f_oz, ["12.0 oz"]) == [True]
    assert det_oz.historico[0]["status"] == "aplicada"
    assert not any(p["status"].startswith("piso") for p in det_oz.historico)


# --- Regressao: os prompts de deteccao constroem um ChatPromptTemplate REAL sem erro ---

def test_prompts_de_deteccao_constroem_e_formatam():
    """Pega chave literal nao-escapada (ex.: regex `[A-Z]{2}`) que o LangChain leria
    como variavel '2' e crashava toda run ao vivo -- agentes fake nao pegam isso."""
    p1 = ChatPromptTemplate.from_messages(
        [("system", regra.SISTEMA), ("human", regra.HUMANO)])
    p1.format(coluna="city", amostra="x", total_linhas=1, total_distintos=1)
    p2 = ChatPromptTemplate.from_messages(
        [("system", refino.SISTEMA_UPDATE), ("human", refino.HUMANO_UPDATE)])
    p2.format(coluna="abv", codigo_atual="def detectar(col): return col.isin([])",
              feedback="x")


def test_prompt_de_deteccao_mostra_o_regex_correto_ao_llm():
    render = ChatPromptTemplate.from_messages([("system", regra.SISTEMA)]).format()
    assert "[A-Z]" + chr(123) + "2" + chr(125) in render, \
        "chave dobrada des-escapa: o LLM deve ver o regex [A-Z]{2} literal"


# --- Auditoria central: a deteccao e' cega ao clean ---

class _ColunaCega:
    """Duble de Coluna cujo `.limpo` levanta ao ser lido."""

    def __init__(self, nome, sujo):
        self.nome = nome
        self.sujo = sujo
        self.valores_distintos = sorted(sujo.unique().tolist())
        self.contagem = sujo.value_counts().to_dict()

    @property
    def limpo(self):
        raise AssertionError("a deteccao leu coluna.limpo")


class _AmostraCega:
    """Duble de Amostra cujo `.rotulados` (o orcamento com clean) levanta."""

    def __init__(self, representantes, total_linhas, total_distintos):
        self.representantes = representantes
        self.linhas = set()
        self.total_linhas = total_linhas
        self.total_distintos = total_distintos

    @property
    def rotulados(self):
        raise AssertionError("a deteccao leu amostra.rotulados")


def _nomes_executaveis(*funcoes) -> set:
    """Nomes que o CODIGO das funcoes usa, ignorando docstring e comentario."""
    fonte = "".join(inspect.getsource(f) for f in funcoes)
    return {tok.string
            for tok in tokenize.generate_tokens(io.StringIO(fonte).readline)
            if tok.type == tokenize.NAME}


def test_gerar_regra_deteccao_nao_toca_o_gabarito():
    """gerar_regra_deteccao recebe objetos que CARREGAM o clean; prova que nao o le.

    Alegacao central do projeto: a deteccao decide olhando SO' os representantes
    sujos. Os dubles levantam se alguem tocar `coluna.limpo` ou `amostra.rotulados`.
    """
    codigo_oz = "def detectar(col):\n    return col.str.endswith('oz')\n"
    col = _ColunaCega("ounces", pd.Series(["12.0 oz", "12", "16.0 oz"]))
    amostra = _AmostraCega(["12.0 oz", "12"], 3, 3)

    def fake(_entrada):
        return RegraDeteccao(erro_provavel=True, condicao_regex=None,
                             codigo=codigo_oz, cadeia="sufixo oz marca o erro")

    detector = deteccao.gerar_regra_deteccao(col, amostra, RunnableLambda(fake))
    assert detector is not None
    assert detector.codigo.strip() == codigo_oz.strip()


def test_gerar_regra_deteccao_codigo_executavel_nao_referencia_clean():
    nomes = _nomes_executaveis(deteccao.gerar_regra_deteccao, regra._formatar_amostra)
    assert {"clean", "limpo", "rotulados"}.isdisjoint(nomes)
