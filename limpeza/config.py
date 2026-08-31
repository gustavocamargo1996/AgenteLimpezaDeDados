"""Configuracao central da POC. Tudo que e' ajustavel mora aqui."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RAIZ = Path(__file__).resolve().parent.parent

# --- Modelo de embeddings local (nao e' dado do usuario, so' infra) ---
MODELO_EMBEDDING = Path(os.getenv("MODELO_EMBEDDING", "./all-MiniLM-L6-v2"))
ARQ_ONNX = MODELO_EMBEDDING / "onnx" / "model_O4.onnx"
ARQ_TOKENIZER = MODELO_EMBEDDING / "tokenizer.json"

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
NULO = "null"  # sentinela de ausencia: NaN vira a string 'null'
