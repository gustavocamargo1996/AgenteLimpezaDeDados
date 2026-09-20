"""Configuracao central da POC. Tudo que e' ajustavel mora aqui."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Privacidade: tracing manda o prompt inteiro (com celulas reais) para a nuvem.
for _VAR in ("LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2",
             "LANGSMITH_TRACING", "LANGCHAIN_TRACING"):
    os.environ[_VAR] = "false"

RAIZ = Path(__file__).resolve().parent.parent

# Modelo de embeddings local: nao e' dado do usuario, so' infra.
MODELO_EMBEDDING = Path(os.getenv("MODELO_EMBEDDING", "./all-MiniLM-L6-v2"))
ARQ_ONNX = MODELO_EMBEDDING / "onnx" / "model_O4.onnx"
ARQ_TOKENIZER = MODELO_EMBEDDING / "tokenizer.json"

DIR_RUNS = RAIZ / "runs"

MODELO_LLM = os.getenv("MODELO_LLM", "gpt-4o-mini")
TEMPERATURA = 0.0


def _segundos(bruto: str | None) -> int | None:
    """Le TIMEOUT_LLM; valor nao-inteiro ou <=0 e' ignorado, sem traceback."""
    try:
        valor = int(str(bruto).strip())
    except (TypeError, ValueError):
        return None
    return valor if valor > 0 else None


# None significa "usa o default do provedor"; ver TIMEOUT_POR_PROVEDOR abaixo.
TIMEOUT_LLM = _segundos(os.getenv("TIMEOUT_LLM"))

PROVEDOR = os.getenv("PROVEDOR", "openai")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Sobrepoe MODELO_LLM num papel so'; vazio significa "usa o geral".
MODELOS_POR_PAPEL = {
    papel: os.getenv(f"MODELO_{papel.upper()}")
    for papel in ("deteccao", "especificador", "codigo", "fd")
}
MODELOS_POR_PAPEL = {p: m for p, m in MODELOS_POR_PAPEL.items() if m}

# Ollama local em CPU ja levou 808s numa chamada; 180 derrubaria o run.
TIMEOUT_POR_PROVEDOR = {"openai": 180, "ollama": 900}

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
