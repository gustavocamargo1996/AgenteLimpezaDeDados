"""Configuracao central da POC. Tudo que e' ajustavel mora aqui."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RAIZ = Path(__file__).resolve().parent.parent

# Modelo de embeddings local: nao e' dado do usuario, so' infra.
MODELO_EMBEDDING = Path(os.getenv("MODELO_EMBEDDING", "./all-MiniLM-L6-v2"))
ARQ_ONNX = MODELO_EMBEDDING / "onnx" / "model_O4.onnx"
ARQ_TOKENIZER = MODELO_EMBEDDING / "tokenizer.json"

DIR_RUNS = RAIZ / "runs"

MODELO_LLM = os.getenv("MODELO_LLM", "gpt-4o-mini")
TEMPERATURA = 0.0
TIMEOUT_LLM = 180

# Ver docs/DECISOES.md#kmeans-sobre-distintos.
N_CLUSTERS = 6
MAX_REPRESENTANTES = 12

MAX_TENTATIVAS_CODIGO = 3

# Ver docs/DECISOES.md#loop-de-refinamento-com-oraculo.
ITERACOES_DETECCAO = 5
AMOSTRAS_POR_ITERACAO = 2
# Precisao minima da regra no rotulado cumulativo; abaixo disso cai para DETECTA_NADA.
LIMITE_PRECISAO_DETECCAO = 0.8
# Bonus para artefato de typo 'x'; especifico de benchmark, desligue para dado real.
BONUS_ARTEFATO_X = True
# Corte de MI normalizada para eleger colunas candidatas a determinante da FD.
MI_THRESHOLD = 0.5
