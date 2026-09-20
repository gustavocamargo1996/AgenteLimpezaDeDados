"""O unico lugar do repositorio que sabe qual provedor de LLM existe."""
import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from . import config


class ProvedorDesconhecido(ValueError):
    """PROVEDOR com valor que a fabrica nao sabe construir."""


def modelo_do_papel(papel: str) -> str:
    """Modelo especifico do papel, senao o geral."""
    return config.MODELOS_POR_PAPEL.get(papel) or config.MODELO_LLM


def timeout_do_provedor(provedor: str) -> int:
    """Timeout explicito de TIMEOUT_LLM, senao o default do provedor."""
    if os.getenv("TIMEOUT_LLM"):
        return int(os.getenv("TIMEOUT_LLM"))
    return config.TIMEOUT_POR_PROVEDOR.get(provedor, config.TIMEOUT_LLM)


def cliente(provedor: str, modelo: str, segundos: int):
    """Constroi o cliente cru do provedor, sem amarrar schema."""
    if provedor == "openai":
        return ChatOpenAI(model=modelo, temperature=config.TEMPERATURA,
                          timeout=segundos, max_retries=2)
    if provedor == "ollama":
        # ChatOllama 1.1 nao tem timeout nem max_retries; o httpx recebe por aqui.
        return ChatOllama(model=modelo, base_url=config.OLLAMA_URL,
                          temperature=config.TEMPERATURA,
                          client_kwargs={"timeout": segundos})
    raise ProvedorDesconhecido(
        f"PROVEDOR={provedor!r} nao e' suportado. Use 'openai' ou 'ollama'. "
        "Nao caio para openai em silencio: isso mandaria dado para fora da rede.")


def construir(schema, papel: str, modelo: str | None = None):
    """Devolve um agente que preenche `schema`, no provedor configurado."""
    provedor = config.PROVEDOR.strip().lower()
    escolhido = modelo or modelo_do_papel(papel)
    return cliente(provedor, escolhido,
                   timeout_do_provedor(provedor)).with_structured_output(schema)
