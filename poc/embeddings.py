"""Embeddings locais via all-MiniLM-L6-v2 em ONNX.

Usa tokenizers + onnxruntime direto, sem sentence-transformers -- que puxaria
torch (~2 GB) para fazer mean pooling de 25 strings. O modelo ja esta em disco
no clone do ZeroDC: nada e' baixado, nada sai da maquina.
"""
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from . import config


class Embedder:
    def __init__(self, arq_tokenizer=None, arq_onnx=None, max_tokens: int = 128):
        self.tokenizer = Tokenizer.from_file(str(arq_tokenizer or config.ARQ_TOKENIZER))
        self.tokenizer.enable_truncation(max_length=max_tokens)
        self.tokenizer.enable_padding()

        opcoes = ort.SessionOptions()
        opcoes.log_severity_level = 3  # silencia warnings de otimizacao
        self.sessao = ort.InferenceSession(
            str(arq_onnx or config.ARQ_ONNX),
            sess_options=opcoes,
            providers=["CPUExecutionProvider"],
        )
        self.entradas_esperadas = [e.name for e in self.sessao.get_inputs()]

    def _montar_entrada(self, codificados):
        ids = np.array([c.ids for c in codificados], dtype=np.int64)
        mascara = np.array([c.attention_mask for c in codificados], dtype=np.int64)
        tipos = np.array([c.type_ids for c in codificados], dtype=np.int64)
        disponivel = {
            "input_ids": ids,
            "attention_mask": mascara,
            "token_type_ids": tipos,
        }
        entrada = {n: disponivel[n] for n in self.entradas_esperadas if n in disponivel}
        return entrada, mascara

    def codificar(self, textos: list[str], lote: int = 64) -> np.ndarray:
        """Devolve matriz (n, dim) com vetores normalizados (L2)."""
        blocos = []
        for inicio in range(0, len(textos), lote):
            fatia = [t if t else " " for t in textos[inicio : inicio + lote]]
            codificados = self.tokenizer.encode_batch(fatia)
            entrada, mascara = self._montar_entrada(codificados)
            saidas = self.sessao.run(None, entrada)

            oculto = next((s for s in saidas if getattr(s, "ndim", 0) == 3), None)
            if oculto is None:
                raise RuntimeError(
                    "modelo ONNX nao devolveu tensor 3D (last_hidden_state)"
                )

            peso = mascara[..., None].astype(np.float32)
            vetores = (oculto * peso).sum(axis=1) / np.clip(peso.sum(axis=1), 1e-9, None)
            norma = np.clip(np.linalg.norm(vetores, axis=1, keepdims=True), 1e-9, None)
            blocos.append((vetores / norma).astype(np.float32))

        return np.vstack(blocos)
