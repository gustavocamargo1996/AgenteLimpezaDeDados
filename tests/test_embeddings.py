"""Alarme do modelo ONNX real: roda so' na maquina de quem tem o modelo baixado."""
import pytest

from limpeza import config
from limpeza.amostragem import Embedder

_MODELO_AUSENTE = not (config.ARQ_ONNX.exists() and config.ARQ_TOKENIZER.exists())


@pytest.mark.skipif(_MODELO_AUSENTE, reason="modelo de embeddings ausente ou incompleto")
def test_embedder_real_aproxima_valores_parecidos():
    """'12.0 oz' deve ficar mais perto de '12.0' do que de 'abacaxi' (MiniLM real)."""
    embedder = Embedder()
    matriz = embedder.codificar(["12.0 oz", "12.0", "N/A", "abacaxi"])
    similar = float(matriz[0] @ matriz[1])
    distante = float(matriz[0] @ matriz[3])
    assert similar > distante, (
        f"'12.0 oz'~'12.0' = {similar:.3f} deveria ser > "
        f"'12.0 oz'~'abacaxi' = {distante:.3f} -- modelo ONNX ausente/corrompido"
    )
