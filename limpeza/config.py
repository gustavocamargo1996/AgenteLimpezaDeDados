"""Configuracao central da POC. Tudo que e' ajustavel mora aqui."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RAIZ = Path(__file__).resolve().parent.parent

# --- Fontes externas (reaproveitadas do clone do ZeroDC, somente leitura) ---
ZERODC_DIR = Path(os.getenv("ZERODC_DIR", r"F:\Projetos\GC\ZeroDC"))
DATASET = os.getenv("DATASET", "beers")
SUFIXO = None  # fatia do dataset: "300" usa {nome}_dirty_300.csv
DIR_DATASET = ZERODC_DIR / "datasets" / DATASET
CSV_SUJO = DIR_DATASET / f"{DATASET}_dirty.csv"
CSV_LIMPO = DIR_DATASET / f"{DATASET}_clean.csv"

DIR_MODELO = ZERODC_DIR / "all-MiniLM-L6-v2"
ARQ_ONNX = DIR_MODELO / "onnx" / "model_O4.onnx"
ARQ_TOKENIZER = DIR_MODELO / "tokenizer.json"

DIR_RUNS = RAIZ / "runs"

# --- LLM ---
MODELO_LLM = os.getenv("MODELO_LLM", "gpt-4o-mini")
TEMPERATURA = 0.0
TIMEOUT_LLM = 180

# --- Amostragem por KMeans ---
# Roda sobre os valores DISTINTOS da coluna, nao sobre todas as celulas:
# 2410 linhas de 'ounces' colapsam em 25 valores, e embedar 25 e' o que
# revela o padrao. Desvio consciente do ZeroDC, que embeda celula a celula.
N_CLUSTERS = 6
MAX_REPRESENTANTES = 12

# --- Geracao de codigo ---
MAX_TENTATIVAS_CODIGO = 3

# --- Loop de refinamento da deteccao (--iteracoes-deteccao N) ---
# 5 iteracoes x 2 amostras = orcamento de ~10 rotulos por coluna.
ITERACOES_DETECCAO = 5
AMOSTRAS_POR_ITERACAO = 2
# Guarda heuristica contra regra AMPLA DEMAIS no loop de deteccao: precisao
# minima que a regra revisada (e a resident final) deve ter no conjunto ROTULADO
# cumulativo do oraculo. Marcar como erro valores que o oraculo confirmou limpos
# derruba a precisao; abaixo deste piso a regra e' rejeitada (guarda por
# iteracao) ou a resident cai para DETECTA_NADA (piso final). Tunavel. Fronteira:
# precisao < LIMITE rejeita; precisao == LIMITE passa.
LIMITE_PRECISAO_DETECCAO = 0.8
# Bonus de artefato 'x' no scorer suspeito da amostragem previsto-limpo do loop
# (deteccao._selecionar_suspeito, portado de detection.py:436-443). O 'x' e'
# marcador de typo dos datasets do ZeroDC -- especifico de BENCHMARK, NAO um
# principio de limpeza. Desligue (False) para dado real, onde 'x' nao sinaliza
# erro. A raridade de caractere alnum fica SEMPRE ativa; so' este bonus e'
# opcional.
BONUS_ARTEFATO_X = True
# Corte da MI normalizada (divide-by-max) para eleger colunas candidatas a
# determinante na FD. Replica o uso do ZeroDC.
MI_THRESHOLD = 0.5

# --- Dados ---
NULO = "null"  # mesmo sentinela do ZeroDC: NaN vira a string 'null'


def caminhos_dataset(dataset: str, sufixo: str | None = None) -> tuple[Path, Path]:
    """Resolve (csv_sujo, csv_limpo) para um dataset, com sufixo opcional.

    Sem sufixo: `{nome}_dirty.csv` / `{nome}_clean.csv` (comportamento antigo).
    Com sufixo '300': `{nome}_dirty_300.csv` / `{nome}_clean_300.csv`.
    """
    base = ZERODC_DIR / "datasets" / dataset
    marca = f"_{sufixo}" if sufixo else ""
    return base / f"{dataset}_dirty{marca}.csv", base / f"{dataset}_clean{marca}.csv"

# Colunas processadas por default: as 5 que sabemos ter erro no beers (vies
# declarado). Use --colunas todas para deixar o pipeline decidir coluna a coluna.
COLUNAS_PADRAO = ["ounces", "ibu", "abv", "city", "state"]
