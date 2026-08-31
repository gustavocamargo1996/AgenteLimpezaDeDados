"""O que perguntar ao oraculo: classificacao e amostragem do loop de refino."""
import re

import numpy as np
import pandas as pd

from .. import config


def _classificar(funcao, valores: list[str]) -> list[bool]:
    """Classifica uma lista de VALORES distintos pela regra Series viva.

    Idioma Series: monta `pd.Series(valores, dtype=object)` e chama `funcao` UMA
    vez (a funcao e' `detectar(col) -> pd.Series[bool]`). Le o resultado
    POSICIONALMENTE (`list(res.fillna(False).astype(bool))`), alinhado por
    posicao a `valores` -- nunca por indice. Robusta a falha (mesma politica da
    classificacao por-valor anterior): `funcao` None, retorno nao-Series, tamanho
    divergente, None ou excecao -> `[False]*len(valores)` (nada marcado). E' o
    ponto unico de aplicacao Series no feedback/amostragem/guarda.
    """
    n = len(valores)
    if funcao is None or n == 0:
        return [False] * n
    try:
        res = funcao(pd.Series(valores, dtype=object))
    except Exception:  # noqa: BLE001 -- classificacao nao pode derrubar o loop
        return [False] * n
    if not isinstance(res, pd.Series) or len(res) != n:
        return [False] * n
    try:
        return [bool(x) for x in res.fillna(False).astype(bool)]
    except Exception:  # noqa: BLE001 -- coercao nao pode derrubar o loop
        return [False] * n


def _selecionar_diverso(candidatos: list[str], referencia, n: int, embedder) -> list[str]:
    """Farthest-point deterministico sobre VALORES DISTINTOS (nao celulas).

    Portado de detection.py:355-389 (select_diverse_indices), reduzido ao caso
    intra-coluna: escolhe ate `n` valores de `candidatos` que maximizam a
    distancia minima ao conjunto de `referencia` (valores ja usados) e aos ja
    escolhidos, no espaco de embeddings. Semente (sem referencia nem escolhidos)
    = valor mais distante do centroide dos candidatos. Desempate = menor indice
    (np.argmax devolve a primeira posicao maxima e `candidatos` esta em ordem
    estavel). Reusa SO' embeddings.codificar (via amostragem.Embedder) -- NAO
    toca amostragem.selecionar (que e' KMeans).
    """
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
    """Frequencia de cada caractere ALNUM lowercased sobre TODAS as celulas.

    Portado de detection.py:406-411. Conta por CELULA (nao por valor distinto):
    percorre `sujo_col` inteira e, por celula, incrementa cada char alnum
    presente (usa `set(valor)`, presenca e' 1 por celula, nao multiplicidade
    dentro da celula). Consequencia: um char que so' aparece em 1 valor distinto
    mas em muitas celulas fica FREQUENTE (raridade baixa). Lowercase para alinhar
    com a extracao de chars em `_score_artefato` (detection.py:406,429).
    """
    contagem: dict = {}
    for valor in sujo_col.astype(str).str.lower():
        for ch in set(valor):
            if ch.isalnum():
                contagem[ch] = contagem.get(ch, 0) + 1
    return contagem


def _score_artefato(valor: str, char_counts: dict, usar_bonus_x: bool) -> float:
    """Score de artefato de UM valor: raridade de char alnum (+ bonus 'x').

    Portado de detection.py:428-444.
      - raridade = max(1 / char_counts[ch]) sobre os chars ALNUM do valor,
        LOWERCASED antes (detection.py:429) senao maiuscula desalinha do
        char_counts lowercased. `.get(ch, 1)` defensivo (detection.py:432): char
        ausente conta como 1 (raridade 1.0), nunca div-zero.
      - raridade = 0 quando o valor nao tem NENHUM char alnum (detection.py:433);
        preservado -- e' o que torna corrupcao nao-alnum (ex.: sufixo '%')
        invisivel a este scorer.
      - bonus de 'x' (SO' se `usar_bonus_x`): +0.5 por cada gatilho de
        detection.py:436-443 -- 'x' presente, [a-z]x[a-z], \\d+x|x\\d, 'xx'.
        Atras da flag porque e' artefato de benchmark, nao de dado real.
    """
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
    """Amostrador que PROCURA o erro: diversidade + score de artefato.

    Portado de detection.py:392-459 (select_suspicious_clean_indices), reduzido
    ao caso intra-coluna sobre VALORES DISTINTOS (a unidade da POC), nao indices
    de celula. Espelha `_selecionar_diverso` na parte de diversidade (min-dist a
    referencia + ja-escolhidos no espaco de embeddings; sem ancora -> distancia
    ao centroide) e SOMA um score de artefato (`_score_artefato`) por candidato.
    Por pick: normaliza diversidade e artefato cada um por seu proprio max, COM
    GUARDA max>0 antes de dividir (detection.py:447-450) -- coluna degenerada
    (todos artefato 0, ex.: valores sem char alnum) nao lanca. argmax com
    desempate no menor indice (np.argmax + ordem estavel) -> deterministico.
    Reusa SO' embeddings.codificar.

    DIVERGENCIA DECLARADA do ZeroDC: la o scorer suspeito so' dispara a cada 5a
    checagem do previsto-limpo (detection.py:495, `% 5 == 4`); aqui ele e' usado
    SEMPRE no pick previsto-limpo, porque o loop da POC e' curto (poucas
    iteracoes, 1 previsto-limpo por iteracao) -- e' o objetivo do porte.
    """
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
    """Escolhe ate `n` valores distintos NOVOS para o oraculo rotular.

    Metade previsto-sujo, metade previsto-limpo (plano secao 6, passo 2). O pick
    previsto-SUJO segue farthest-point puro (`_selecionar_diverso`) -- ja encontra
    erro onde a corrupcao e' densa. O pick previsto-LIMPO passa a usar o scorer
    que PROCURA o erro (`_selecionar_suspeito`: diversidade + raridade de char
    alnum + bonus 'x' opcional via config.BONUS_ARTEFATO_X), para expor ao
    oraculo os falsos negativos das colunas de efeito-zero. Se um lado nao tem
    candidato, o outro absorve a cota restante -- nunca lanca (risco 6.6). O ramo
    "resto" (completude de cota) segue `_selecionar_diverso`. Reusa SO'
    embeddings.codificar.
    """
    usados = set(usados or [])
    novos = [v for v in valores if v not in usados]
    marcas = _classificar(funcao, novos)
    previsto_sujo, previsto_limpo = [], []
    for v, marcado in zip(novos, marcas):
        (previsto_sujo if marcado else previsto_limpo).append(v)

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
