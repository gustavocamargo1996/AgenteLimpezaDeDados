"""Configuracao central da POC. Tudo que e' ajustavel mora aqui."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RAIZ = Path(__file__).resolve().parent.parent

# --- Fontes externas (reaproveitadas do clone do ZeroDC, somente leitura) ---
ZERODC_DIR = Path(os.getenv("ZERODC_DIR", r"F:\Projetos\GC\ZeroDC"))
DATASET = os.getenv("DATASET", "beers")
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

# --- Modo end-to-end (--e2e) ---
# Liga/desliga a camada 3 da cascata (fallback por celula via LLM). DEFAULT
# False: medido em 7 runs que o fallback resolve ~100% das correcoes de
# ounces/abv mas com acerto ~0-0.5% -- custa milhares de chamadas para ~0 de
# correcao util, e e' a unica camada sem gate. Desligado, as celulas que
# codigo/FD nao cobrem ficam FLAGADAS (trilha 'nao_resolvida', contagem
# fallback==0). Ligue (True) so' para comparacao com o comportamento antigo.
USAR_FALLBACK = False
# Teto de chamadas de LLM na camada 3 (fallback por celula), contadas por VALOR
# distinto enviado, por coluna. Ao atingir, as celulas restantes ficam sujas e
# sao logadas -- o custo nao escapa. So' tem efeito quando USAR_FALLBACK=True.
LIMITE_FALLBACK = 50

# --- Loop de refinamento da deteccao (--e2e --iteracoes-deteccao N) ---
# Default 1 = deteccao de 1 passe (comportamento atual, SEM loop). N>1 ativa o
# active learning com oraculo (clean) que refina `detectar(valor)` iteracao a
# iteracao, para quebrar o blind spot da corrupcao universal (ounces/abv/city).
# AMOSTRAS_POR_ITERACAO = quantos valores distintos o oraculo rotula por iteracao
# (metade previsto-sujo, metade previsto-limpo).
ITERACOES_DETECCAO = 1
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
# determinante na FD e a contexto no fallback. Replica o uso do ZeroDC.
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

# Colunas processadas por default.
# ATENCAO (vies declarado): estas sao exatamente as 5 colunas que sabemos ter
# erro no beers. Restringir a elas vaza um pouco de gabarito para o modo
# 'blind' -- estamos dizendo ao agente onde procurar. Para rodar sem esse
# vies, use --colunas todas, e o agente tera que decidir sozinho, coluna a
# coluna, se ha erro. No beers isso da' 10 colunas (as 11 menos `index`, que o
# main.py descarta): estas 5 mais as 5 limpas.
COLUNAS_PADRAO = ["ounces", "ibu", "abv", "city", "state"]

MODOS = ("blind", "budget")
