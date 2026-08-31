"""Embeddings locais (MiniLM ONNX) e selecao de representantes por KMeans.

Usa tokenizers + onnxruntime direto, sem sentence-transformers -- que puxaria
torch (~2 GB) para fazer mean pooling de 25 strings. O modelo ja esta em disco
no clone do ZeroDC: nada e' baixado, nada sai da maquina.

Ideia da selecao: em vez de mandar a coluna inteira ao LLM (caro e ruidoso),
agrupa os valores distintos no espaco de embeddings e mostra, de cada grupo, o
mais atipico e o mais tipico. O atipico expoe a corrupcao; o tipico ancora a
norma.

Limite estrutural que a POC existe para demonstrar: quando a corrupcao atinge
100% da coluna, a forma corrompida E' a norma e nao ha atipico para achar.
Nenhuma quantidade de clusters resolve isso.
"""
import numpy as np
import onnxruntime as ort
from sklearn.cluster import KMeans
from tokenizers import Tokenizer

from . import config
from .tipos import Amostra, Coluna


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


def selecionar(matriz: np.ndarray, n_clusters=None, maximo=None) -> list[int]:
    """Indices (na lista de valores distintos) escolhidos como representantes."""
    n_clusters = n_clusters or config.N_CLUSTERS
    maximo = maximo or config.MAX_REPRESENTANTES

    total = len(matriz)
    if total == 0:
        return []
    if total <= maximo:
        return list(range(total))

    k = max(2, min(n_clusters, total))
    modelo = KMeans(n_clusters=k, random_state=0, n_init=10).fit(matriz)

    escolhidos: list[int] = []
    for grupo in range(k):
        membros = np.where(modelo.labels_ == grupo)[0]
        if len(membros) == 0:
            continue
        distancias = np.linalg.norm(matriz[membros] - modelo.cluster_centers_[grupo], axis=1)
        escolhidos.append(int(membros[distancias.argmax()]))  # mais atipico
        if len(membros) > 1:
            escolhidos.append(int(membros[distancias.argmin()]))  # mais tipico

    vistos, saida = set(), []
    for i in escolhidos:
        if i not in vistos:
            vistos.add(i)
            saida.append(i)
    return saida[:maximo]


_EMBEDDER: Embedder | None = None


def embedder() -> Embedder:
    """Embedder unico do processo -- o ONNX carrega uma vez por run."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER


def representantes(coluna: Coluna, emb=None) -> Amostra:
    """Escolhe os representantes da coluna e rotula as linhas que serao exibidas."""
    emb = emb or embedder()
    indices = selecionar(emb.codificar(coluna.valores_distintos))
    valores = [coluna.valores_distintos[i] for i in indices]
    primeira = {valor: int(idx) for idx, valor in coluna.sujo.drop_duplicates().items()}
    linhas = {primeira[v] for v in valores if v in primeira}
    return Amostra(
        representantes=valores,
        linhas=linhas,
        rotulados=_rotular(coluna, linhas),
        total_linhas=len(coluna.sujo),
        total_distintos=len(coluna.valores_distintos),
    )


def _rotular(coluna: Coluna, linhas: set) -> list[dict]:
    """Orcamento de rotulos: o par sujo->limpo das linhas efetivamente exibidas."""
    # Estas linhas sao tambem o holdout da metrica: medir sobre um valor ja
    # mostrado ao agente mede memorizacao, nao generalizacao.
    rotulados = []
    for idx in sorted(linhas):
        sujo, limpo = coluna.sujo.at[idx], coluna.limpo.at[idx]
        rotulados.append({"indice": int(idx), "sujo": sujo, "limpo": limpo,
                          "eh_erro": sujo != limpo})
    return rotulados
