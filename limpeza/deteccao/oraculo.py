"""O que perguntar ao oraculo: classificacao e amostragem do loop de refino."""
import re
from collections import Counter

import numpy as np
import pandas as pd

from .. import config


def _classificar(funcao, valores: list[str]) -> list[bool]:
    """Classifica valores distintos pela regra Series viva; qualquer falha devolve tudo False (testado)."""
    n = len(valores)
    if funcao is None or n == 0:
        return [False] * n
    try:
        res = funcao(pd.Series(valores, dtype=object))
        if not isinstance(res, pd.Series) or len(res) != n:
            return [False] * n
        return [bool(x) for x in res.fillna(False).astype(bool)]
    except Exception:  # noqa: BLE001 -- chamada OU coercao: classificacao nao pode derrubar o loop
        return [False] * n


def _selecionar_diverso(candidatos: list[str], referencia, n: int, embedder) -> list[str]:
    # Farthest-point deterministico: maximiza a distancia minima a referencia e aos ja escolhidos.
    candidatos = list(dict.fromkeys(candidatos))  # unico, ordem preservada
    if n <= 0 or not candidatos:
        return []
    if len(candidatos) <= n:
        return candidatos

    referencia = [r for r in dict.fromkeys(referencia) if r not in candidatos]
    universo = candidatos + referencia
    matriz = embedder.codificar(universo)
    emb_cand = matriz[: len(candidatos)]
    ref_emb = matriz[len(candidatos):] if referencia else None

    escolhidos: list[int] = []
    disponiveis = list(range(len(candidatos)))
    while len(escolhidos) < n and disponiveis:
        sub = emb_cand[disponiveis]
        if ref_emb is not None or escolhidos:
            ancoras = []
            if ref_emb is not None:
                ancoras.append(ref_emb)
            if escolhidos:
                ancoras.append(emb_cand[escolhidos])
            anc = np.vstack(ancoras)
            dist = np.linalg.norm(sub[:, None, :] - anc[None, :, :], axis=2)
            score = dist.min(axis=1)
        else:
            centroide = sub.mean(axis=0)
            score = np.linalg.norm(sub - centroide, axis=1)
        melhor = int(np.argmax(score))  # desempate = menor indice
        escolhidos.append(disponiveis.pop(melhor))

    return [candidatos[i] for i in escolhidos]


def _contar_chars_alnum(sujo_col) -> dict:
    chars = (ch for v in sujo_col.astype(str).str.lower() for ch in set(v) if ch.isalnum())
    return dict(Counter(chars))


def _score_artefato(valor: str, char_counts: dict, usar_bonus_x: bool) -> float:
    """Score de artefato de um valor: raridade de caractere alfanumerico, mais bonus opcional para 'x'."""
    v = str(valor).lower()
    chars = [ch for ch in set(v) if ch.isalnum()]
    if chars:
        rarity = max(1 / char_counts.get(ch, 1) for ch in chars)
    else:
        rarity = 0
    bonus = 0
    if usar_bonus_x:
        if "x" in v:
            bonus += 0.5
        if re.search(r"[a-z]x[a-z]", v):
            bonus += 0.5
        if re.search(r"\d+x|x\d", v):
            bonus += 0.5
        if "xx" in v:
            bonus += 0.5
    return rarity + bonus


def _selecionar_suspeito(
    candidatos: list[str], referencia, n: int, embedder, sujo_col, usar_bonus_x: bool
) -> list[str]:
    """Amostra valores combinando diversidade de embeddings com score de artefato, para achar erro escondido."""
    candidatos = list(dict.fromkeys(candidatos))  # unico, ordem preservada
    if n <= 0 or not candidatos:
        return []
    if len(candidatos) <= n:
        return candidatos

    referencia = [r for r in dict.fromkeys(referencia) if r not in candidatos]
    universo = candidatos + referencia
    matriz = embedder.codificar(universo)
    emb_cand = matriz[: len(candidatos)]
    ref_emb = matriz[len(candidatos):] if referencia else None

    char_counts = _contar_chars_alnum(sujo_col)

    escolhidos: list[int] = []
    disponiveis = list(range(len(candidatos)))
    while len(escolhidos) < n and disponiveis:
        sub = emb_cand[disponiveis]
        if ref_emb is not None or escolhidos:
            ancoras = []
            if ref_emb is not None:
                ancoras.append(ref_emb)
            if escolhidos:
                ancoras.append(emb_cand[escolhidos])
            anc = np.vstack(ancoras)
            dist = np.linalg.norm(sub[:, None, :] - anc[None, :, :], axis=2)
            diversidade = dist.min(axis=1)
        else:
            centroide = sub.mean(axis=0)
            diversidade = np.linalg.norm(sub - centroide, axis=1)

        artefato = np.array(
            [_score_artefato(candidatos[i], char_counts, usar_bonus_x) for i in disponiveis]
        )

        if artefato.max() > 0:
            artefato = artefato / artefato.max()
        if diversidade.max() > 0:
            diversidade = diversidade / diversidade.max()

        score = diversidade + artefato
        melhor = int(np.argmax(score))  # desempate = menor indice
        escolhidos.append(disponiveis.pop(melhor))

    return [candidatos[i] for i in escolhidos]


def _amostrar_oraculo(
    funcao, valores: list[str], embedder, usados, n: int, sujo_col
) -> list[str]:
    """Escolhe ate `n` valores novos para o oraculo rotular: metade previsto-sujo, metade previsto-limpo."""
    usados = set(usados or [])
    novos = [v for v in valores if v not in usados]
    marcas = _classificar(funcao, novos)
    pares = list(zip(novos, marcas))
    previsto_sujo = [v for v, marcado in pares if marcado]
    previsto_limpo = [v for v, marcado in pares if not marcado]

    n_sujo = n // 2
    n_limpo = n - n_sujo
    escolhidos = _selecionar_diverso(previsto_sujo, usados, n_sujo, embedder)
    ancora = usados | set(escolhidos)
    escolhidos += _selecionar_suspeito(
        previsto_limpo, ancora, n_limpo, embedder, sujo_col, config.BONUS_ARTEFATO_X
    )

    faltam = n - len(escolhidos)
    if faltam > 0:
        ancora = usados | set(escolhidos)
        resto = [v for v in (previsto_sujo + previsto_limpo) if v not in ancora]
        escolhidos += _selecionar_diverso(resto, ancora, faltam, embedder)
    return escolhidos
