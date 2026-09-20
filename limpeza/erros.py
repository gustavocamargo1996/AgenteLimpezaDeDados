"""Formato unico de excecao no log: o Portainer so' mostra estas linhas."""

LIMITE_MENSAGEM = 200


def descrever(exc: BaseException) -> str:
    """Tipo e mensagem da excecao, truncada no limite."""
    # Sem a mensagem, "404 model not found", "connection refused" e "timeout"
    # do Ollama viram a mesma linha de log.
    texto = " ".join(str(exc).split())
    if len(texto) > LIMITE_MENSAGEM:
        texto = texto[:LIMITE_MENSAGEM] + "..."
    return f"{type(exc).__name__}: {texto}" if texto else type(exc).__name__
