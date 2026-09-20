"""A fabrica de provedores: quem e' construido, com qual modelo e qual timeout."""
import pytest
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from limpeza import config, llm
from limpeza.esquemas import RegraDeteccao


@pytest.fixture(autouse=True)
def _isola_config(monkeypatch):
    """Cada teste mexe em config; isola para nao vazar entre eles."""
    monkeypatch.setattr(config, "PROVEDOR", "openai")
    monkeypatch.setattr(config, "MODELO_LLM", "gpt-4o-mini")
    monkeypatch.setattr(config, "MODELOS_POR_PAPEL", {})
    monkeypatch.delenv("TIMEOUT_LLM", raising=False)


def test_openai_e_o_default():
    assert isinstance(llm.cliente("openai", "gpt-4o-mini", 180), ChatOpenAI)


def test_ollama_quando_configurado():
    c = llm.cliente("ollama", "qwen2.5:3b", 900)
    assert isinstance(c, ChatOllama)
    assert c.model == "qwen2.5:3b"


def test_timeout_do_ollama_vai_por_client_kwargs():
    # ChatOllama 1.1 nao tem `timeout`; o httpx recebe por client_kwargs.
    assert llm.cliente("ollama", "m", 900).client_kwargs["timeout"] == 900


def test_provedor_desconhecido_falha_alto():
    config.PROVEDOR = "gemini"
    with pytest.raises(llm.ProvedorDesconhecido, match="gemini"):
        llm.construir(RegraDeteccao, "deteccao")


def test_provedor_desconhecido_nao_cai_para_openai(monkeypatch):
    """A falha e' o ponto: cair para openai mandaria dado para fora da rede."""
    config.PROVEDOR = "vazio"
    chamou = []
    monkeypatch.setattr(llm, "ChatOpenAI", lambda **k: chamou.append(k))
    with pytest.raises(llm.ProvedorDesconhecido):
        llm.construir(RegraDeteccao, "deteccao")
    assert chamou == [], "construiu um cliente OpenAI apesar do provedor invalido"


def test_modelo_do_papel_vence_o_geral():
    config.MODELOS_POR_PAPEL = {"deteccao": "qwen2.5-coder:14b"}
    assert llm.modelo_do_papel("deteccao") == "qwen2.5-coder:14b"


def test_sem_modelo_do_papel_cai_no_geral():
    config.MODELOS_POR_PAPEL = {"codigo": "outro"}
    assert llm.modelo_do_papel("deteccao") == "gpt-4o-mini"


def test_modelo_explicito_vence_o_do_papel():
    config.MODELOS_POR_PAPEL = {"deteccao": "do-papel"}
    registrado = {}
    original = llm.cliente

    def _espiao(provedor, modelo, segundos):
        registrado["modelo"] = modelo
        return original(provedor, modelo, segundos)

    llm.cliente = _espiao
    try:
        llm.construir(RegraDeteccao, "deteccao", modelo="explicito")
    finally:
        llm.cliente = original
    assert registrado["modelo"] == "explicito"


def test_timeout_muda_com_o_provedor():
    assert llm.timeout_do_provedor("openai") == 180
    assert llm.timeout_do_provedor("ollama") == 900


def test_timeout_llm_explicito_vence_o_default(monkeypatch):
    monkeypatch.setenv("TIMEOUT_LLM", "42")
    assert llm.timeout_do_provedor("ollama") == 42
