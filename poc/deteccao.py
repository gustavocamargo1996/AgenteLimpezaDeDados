"""Deteccao INTRA-COLUNA: constroi a propria mascara de erros, sem ver clean.

Decisao de desenho DECLARADA: a regra e' `detectar(col) -> pd.Series[bool]`;
recebe a COLUNA suja inteira (uma pd.Series de strings) e devolve uma pd.Series
booleana do mesmo tamanho (True por linha corrompida), SEM olhar outras colunas.
O idioma e' `col.str.contains(marcador)` -- o mesmo com que o ZeroDC atinge
abv/city F1=1.0 (regras reais: abv=`df['abv'].str.contains('%')`, city=` [A-Z]{2}$`).
Suficiente para o beers (todo erro e' visivel na propria coluna: sufixo em
ounces, '%' em abv, 'N/A' em ibu, ' XX' final em city, vazio em state). NAO e' o
`fun(df) -> Series` cross-column do ZeroDC (detection.py:147-155,269): so' a
propria coluna -- erros dependentes de contexto passam batido aqui, e essa
limitacao e' aceita para esta POC.

Um unico passe de LLM por coluna (sem loop de active learning): o agente ve so'
os representantes SUJOS e emite a RegraDeteccao ja com o `codigo` de
`detectar(col)`. O codigo passa pelo portao AST do sandbox em modo serie
(`series_mode=True`: allowlist de atributos; `pd` fora do namespace).
`construir_mascara` aplica `detectar(col)` uma vez por coluna e le o resultado
POSICIONALMENTE; NAO importa nem le clean -- auditavel neste arquivo.
"""
import re

import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from . import config, sandbox
from .esquemas import RegraDeteccao

SISTEMA = """Voce escreve regras de DETECCAO de erro para uma coluna de tabela suja.

A regra e' uma funcao `detectar(col) -> pd.Series[bool]`: recebe a COLUNA inteira \
(`col`, uma pd.Series de strings) e devolve uma pd.Series booleana do MESMO \
tamanho -- True na linha corrompida, False na intacta. Ela trabalha a coluna \
elemento a elemento (vetorizado), NAO ve outras colunas.

Sua saida tem a CADEIA (como voce chegou ao criterio, citando valores concretos \
dos representantes) e a REGRA em JSON com o campo `codigo`.

IDIOMA (use este padrao vetorizado):
  - marcador literal ('%', 'oz', 'N/A'): \
`col.astype(str).str.contains('%', regex=False, na=False)`;
  - padrao estrutural (sufixo de 2 letras, barra): \
`col.astype(str).str.contains(r' [A-Z]{{2}}$', regex=True, na=False)`;
  - combinar marcadores com OR dentro da MESMA funcao: \
`col.str.contains('oz', na=False) | col.str.contains('ounce', na=False)`;
  - sentinela exata: `col.astype(str).eq('null')`; vazio: `col.astype(str).eq('')`.
Sempre `na=False` (celula ausente NAO e' marcada por engano).

CONTRATO RIGIDO do `codigo` -- ele passa por validacao AST antes de rodar:
  - exatamente UMA funcao, nada mais (sem import: o modulo `re` JA existe no escopo);
  - a funcao se chama exatamente `detectar` e recebe um unico argumento `col`;
  - devolve uma pd.Series[bool] do tamanho de `col` (True = erro, False = intacto);
  - so' metodos elementwise de `col` (str.contains/startswith/endswith/match, \
astype, eq/ne, isin, isna/notna, fillna, strip, lower, upper). PROIBIDO usar \
operacoes cross-row (duplicated, mode, groupby, shift, value_counts) e \
ler/escrever arquivo. Nada de eval, exec, open, getattr, globals, __import__.

DISCIPLINA:
1. So' marque como erro o que a amostra sustenta. Se os representantes nao mostram
   corrupcao, use `erro_provavel: false` e um `detectar` que devolve tudo False
   (ex.: `col.isin([])`).
2. Detecte pela FORMA/MARCADOR compartilhado (sufixo estranho, sentinela como
   'N/A', vazio, caractere fora do padrao), NAO enumere os valores observados:
   nada de `col.isin(['0.061%', '0.125%'])` nem blacklist -- isso e' memorizar e
   falha em todo valor-erro novo.
3. Prefira nao marcar a marcar demais: um falso positivo vira dano na correcao.

Escreva a cadeia em portugues do Brasil."""

HUMANO = """Coluna: `{coluna}`
Tabela: {total_linhas} linhas, {total_distintos} valores distintos nesta coluna.

Estes sao os valores representativos SUJOS (voce NAO tem o valor correto), \
escolhidos por agrupamento semantico:

{amostra}

Escreva a cadeia e o `codigo` de `detectar(col)`."""

# Fallback de 1 passe: se o unico invoke nao devolve codigo que passe no portao,
# nao marca nada (mascara toda 0 nesta coluna). Nao ha laco de reparo -- a decisao
# foi deteccao em 1 passe. Nunca inventa deteccao. No idioma Series, tudo-False
# alinhado a `col` = `col.isin([])` (usa so' o atributo permitido `isin`, sem pd
# nem `.index`); passa na allowlist do series_mode.
DETECTA_NADA = "def detectar(col):\n    return col.isin([])\n"

_AMOSTRAS_FUMACA = ["12.0 oz", "", "N/A", "17.0", "Portland CA"]


def construir_agente(modelo: str | None = None):
    llm = ChatOpenAI(
        model=modelo or config.MODELO_LLM,
        temperature=config.TEMPERATURA,
        timeout=config.TIMEOUT_LLM,
        max_retries=2,
    )
    return llm.with_structured_output(RegraDeteccao)


def _formatar_amostra(representantes_sujos: list[str], contagem: dict) -> str:
    linhas = []
    for valor in representantes_sujos:
        freq = int(contagem.get(valor, 0))
        linhas.append(f'  - "{valor}"   ({freq}x na coluna)')
    return "\n".join(linhas)


def gerar_regra_deteccao(
    coluna: str,
    representantes_sujos: list[str],
    total_linhas: int,
    total_distintos: int,
    contagem: dict | None = None,
    agente=None,
) -> dict:
    """Um passe de LLM: SO' representantes sujos, sem MI, sem clean.

    Devolve {regra, funcao, erro, tentativas}. `funcao` e' `detectar` viva se o
    codigo passou no portao; se falhou ate o fim, cai para DETECTA_NADA (marca
    zero celulas) -- nunca inventa deteccao.
    """
    agente = agente or construir_agente()
    contagem = contagem or {}
    prompt = ChatPromptTemplate.from_messages([("system", SISTEMA), ("human", HUMANO)])
    regra: RegraDeteccao = (prompt | agente).invoke(
        {
            "coluna": coluna,
            "amostra": _formatar_amostra(representantes_sujos, contagem),
            "total_linhas": total_linhas,
            "total_distintos": total_distintos,
        }
    )

    codigo = (regra.codigo or "").strip()
    funcao = None
    erro = ""
    if codigo:
        try:
            funcao = sandbox.materializar(
                codigo, nome_funcao="detectar", nome_argumento="col",
                series_mode=True,
            )
            sandbox.testar_fumaca(
                funcao, pd.Series(_AMOSTRAS_FUMACA), series_mode=True
            )
        except sandbox.CodigoRejeitado as exc:
            erro = str(exc)
            funcao = None

    if funcao is None:
        funcao = sandbox.materializar(
            DETECTA_NADA, nome_funcao="detectar", nome_argumento="col",
            series_mode=True,
        )
        regra.codigo = DETECTA_NADA

    return {"regra": regra, "funcao": funcao, "erro": erro, "tentativas": 1}


# --------------------------------------------------------------------------- #
# Loop de refinamento (--e2e --iteracoes-deteccao N>1): active learning com
# oraculo (clean). Quebra o ponto cego do 1-passe onde a corrupcao e' universal
# (ounces/abv/city dao F1=0 porque a forma corrompida vira a norma da coluna).
# O 1-passe (N=1) NAO entra aqui -- este bloco so' roda quando iteracoes>1.
# --------------------------------------------------------------------------- #

SISTEMA_UPDATE = """Voce REVISA uma regra de deteccao de erro para uma coluna \
de tabela suja, a partir de FEEDBACK de um oraculo (rotulos verdadeiros de \
algumas celulas).

A regra e' `detectar(col) -> pd.Series[bool]`: recebe a COLUNA inteira (`col`, \
uma pd.Series de strings) e devolve uma pd.Series booleana do mesmo tamanho -- \
True na linha corrompida, False na intacta. Trabalha elemento a elemento \
(vetorizado), NAO ve outras colunas.

IDIOMA (padrao vetorizado): `col.astype(str).str.contains('%', regex=False, \
na=False)`; combinar marcadores com OR na MESMA funcao \
`col.str.contains('oz', na=False) | col.str.contains('ounce', na=False)`; \
sentinela `col.astype(str).eq('null')`; vazio `col.astype(str).eq('')`. \
Sempre `na=False`.

CONTRATO RIGIDO do `codigo` -- ele passa por validacao AST antes de rodar:
  - exatamente UMA funcao, nada mais (sem import: o modulo `re` JA existe no escopo);
  - a funcao se chama exatamente `detectar` e recebe um unico argumento `col`;
  - devolve uma pd.Series[bool] do tamanho de `col` (True = erro, False = intacto);
  - so' metodos elementwise de `col` (str.contains/startswith/endswith/match, \
astype, eq/ne, isin, isna/notna, fillna, strip, lower, upper). PROIBIDO \
operacoes cross-row (duplicated, mode, groupby, shift, value_counts) e I/O;
  - sem eval, exec, open, getattr, globals, __import__ e afins;
  - regex bem formada (flags inline como (?i) so' valem no inicio da expressao).

OBJETIVO da revisao:
1. Reduza FALSOS NEGATIVOS: valores que o oraculo diz ERRO mas a regra deixou \
passar (previu limpo). Aqui e' onde o 1-passe falha quando a corrupcao e' a norma.
2. Reduza FALSOS POSITIVOS: valores corretos que a regra marcou como erro \
(viram dano na correcao). Toda regra retornada DEVE ser False para TODO valor \
que o oraculo confirmou LIMPO neste feedback.
3. Faca NO MAXIMO UMA mudanca comportamental por iteracao. O feedback e' \
CUMULATIVO (traz os rotulos de todas as iteracoes): mudancas bruscas desfazem \
consertos anteriores. Se o feedback ja esta todo OK, devolva o MESMO codigo.

COMO GENERALIZAR (regra critica -- e' o erro mais comum aqui):
Quando varios valores-erro do feedback compartilham um MARCADOR ou uma FORMA \
(um sufixo, um simbolo, uma sentinela, um padrao estrutural), detecte o \
MARCADOR/FORMA com `col.str.contains(...)`, nao os valores especificos. Exemplos \
do mecanismo (adapte ao que o feedback mostrar, NAO copie literalmente):
  - marcador literal como '%' -> `col.str.contains('%', regex=False, na=False)`. \
    NAO liste os valores que tem '%'.
  - sufixo de unidade ('oz', 'ounce') -> detecte o sufixo/marcador, nao cada valor.
  - sentinela ('N/A') ou vazio -> `col.eq('N/A')` / `col.eq('')`.

PROIBIDO (isto e' memorizar, nao detectar):
  - Enumerar ou fixar valores observados: nada de `col.eq('0.061%')`, \
    `col.isin(['0.061%', '0.125%'])`, nem lookahead negativo de valores \
    especificos `(?!0.061%$)(?!0.125%$)...`. Uma blacklist de valores NAO \
    generaliza -- falha em todo valor-erro novo que o oraculo ainda nao mostrou.
  - Inferir whitelist, dicionario, faixa numerica, prefixo obrigatorio ou \
    gramatica completa de formato.
  - Operacoes que olham a coluna como conjunto (duplicated, mode, value_counts, \
    groupby, shift): a deteccao e' por FORMA da propria celula, nao por \
    frequencia/vizinhanca.
Se o feedback nao revela uma forma comum (so' valores avulsos sem marcador \
compartilhado), mantenha a regra atual em vez de inventar blacklist.

Escreva a cadeia (em portugues do Brasil) explicando a UNICA mudanca que fez \
(ou por que manteve) e emita a REGRA em JSON com o campo `codigo`."""

HUMANO_UPDATE = """Coluna: `{coluna}`

CODIGO ATUAL de `detectar(col)`:

```python
{codigo_atual}
```

FEEDBACK do oraculo (CUMULATIVO -- todos os rotulos ate agora, cada linha \
compara a classificacao ATUAL da regra com a verdade do oraculo):

{feedback}

Reescreva o `codigo` de `detectar(col)` corrigindo as discordancias acima \
(uma mudanca comportamental no maximo). Se o feedback nao justifica mudanca, \
devolva exatamente o mesmo codigo."""


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
    estavel). Reusa SO' embeddings.codificar -- NAO toca amostragem.selecionar
    (que e' KMeans).
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


def _montar_feedback_det(funcao, rotulos: list[dict]) -> str:
    """Feedback CUMULATIVO para o LLM de update.

    `rotulos` = lista acumulada de {valor, eh_erro_real} de TODAS as iteracoes
    ate agora. Para cada um recomputa a classificacao ATUAL de `funcao` e
    destaca as discordancias:
      FALSO POSITIVO = a regra marcou SUJO um valor que o oraculo diz LIMPO;
      FALSO NEGATIVO = a regra deixou LIMPO um valor que o oraculo diz ERRO.
    Cumulativo de proposito: sem isso a iteracao 3 poderia desfazer o conserto
    da 1 (a discordancia ja resolvida some do feedback). Devolve o texto do
    prompt.
    """
    marcas = _classificar(funcao, [r["valor"] for r in rotulos])
    linhas = []
    for r, marcado in zip(rotulos, marcas):
        valor = r["valor"]
        eh_erro = bool(r["eh_erro_real"])
        atual = "sujo" if marcado else "limpo"
        verdade = "erro" if eh_erro else "correto"
        if marcado and not eh_erro:
            tag = "FALSO POSITIVO (a regra marcou erro, mas o oraculo diz que esta correto)"
        elif not marcado and eh_erro:
            tag = "FALSO NEGATIVO (a regra deixou passar, mas o oraculo diz que e' erro)"
        else:
            tag = "OK (a regra concorda com o oraculo)"
        linhas.append(
            f'  - valor="{valor}" | regra atual: {atual} | oraculo: {verdade} | {tag}'
        )
    return "\n".join(linhas)


def _precisao_no_oraculo(funcao, rotulos: list[dict]) -> float | None:
    """Precisao da regra VIVA no conjunto ROTULADO cumulativo do oraculo.

    Guarda heuristica contra regra AMPLA DEMAIS (marca como erro valores que o
    oraculo confirmou limpos). O 1o argumento e' o CALLABLE `detectar` vivo, NAO
    o objeto RegraDeteccao: passar o pydantic faria `_classificar` cair na
    excecao -> tudo False -> marcados=0 -> None e a guarda nunca dispararia
    (recurso desligado em silencio).

    `rotulos` = lista {valor, eh_erro_real} acumulada pelo loop. Aplica `funcao`
    aos valores via `_classificar` (uma passada Series); `marcados` = os que a
    regra diz SUJO. Se marca 0 -> None (sem over-flagging mensuravel; guarda
    inerte). Senao devolve |marcados com eh_erro_real True| / |marcados|.

    LIMITACAO DECLARADA (Armadilha 1 do plano): a precisao so' mede o que esta no
    rotulado; e' CEGA a over-flagging em valores SEM representante limpo no
    oraculo (rotulos todos sujos -> precisao 1.0 mesmo para uma regra ampla). NAO
    resolver com fracao de celula (quebra ounces, ~100% erro real -> precisao
    ~1.0, corretamente aceito). O piso final mitiga porque a propria amplitude da
    regra tende a fazer o amostrador colher clean, mas nao e' garantia formal.
    """
    marcas = _classificar(funcao, [r["valor"] for r in rotulos])
    marcados = [r for r, marcado in zip(rotulos, marcas) if marcado]
    if not marcados:
        return None
    acertos = sum(1 for r in marcados if bool(r["eh_erro_real"]))
    return acertos / len(marcados)


def _tentar_update(cadeia_update, coluna: str, codigo_atual: str, feedback: str):
    """Invoca o LLM de update e VALIDA a regra revisada no mesmo portao do 1-passe.

    Devolve (funcao_viva, RegraDeteccao, "") em sucesso; (None, None, motivo) se
    o invoke falhar, o `codigo` vier vazio, ou nao passar em
    sandbox.materializar + testar_fumaca. O chamador entao MANTEM a regra
    anterior (conservador -- risco de regra que quebra em runtime).
    """
    try:
        nova: RegraDeteccao = cadeia_update.invoke(
            {"coluna": coluna, "codigo_atual": codigo_atual, "feedback": feedback}
        )
    except Exception as exc:  # noqa: BLE001 -- update nao pode derrubar o loop
        return None, None, f"invoke falhou ({type(exc).__name__}: {exc})"

    codigo = (nova.codigo or "").strip()
    if not codigo:
        return None, None, "codigo vazio"
    try:
        funcao = sandbox.materializar(
            codigo, nome_funcao="detectar", nome_argumento="col",
            series_mode=True,
        )
        sandbox.testar_fumaca(
            funcao, pd.Series(_AMOSTRAS_FUMACA), series_mode=True
        )
    except sandbox.CodigoRejeitado as exc:
        return None, None, str(exc)
    nova.codigo = codigo
    return funcao, nova, ""


def refinar_regra_deteccao(
    coluna: str,
    funcao,
    regra: RegraDeteccao,
    sujo_col: pd.Series,
    limpo_col: pd.Series,
    valores_distintos: list[str],
    embedder,
    iteracoes: int,
    amostras_por_iter: int,
    agente_update,
) -> dict:
    """Loop de active learning que refina `detectar(col)` com um oraculo.

    Contrato (plano secao 6). Por iteracao:
      1. amostra valores distintos NOVOS (metade previsto-sujo, metade
         previsto-limpo) por farthest-point (`_amostrar_oraculo`);
      2. oraculo: para cada valor amostrado olha as celulas que o contem e decide
         `eh_erro_real = existe >=1 celula com dirty!=clean`; guarda os indices
         dessas celulas em `orcamento` -- SO' para relatorio de custo, NAO para o
         holdout da metrica (plano secao 6.7);
      3. monta o feedback CUMULATIVO (todos os rotulos ate agora);
      4. pede ao LLM (structured output RegraDeteccao) uma regra revisada; ela so'
         substitui a anterior se passar por sandbox.materializar + testar_fumaca,
         senao MANTEM a regra da iteracao anterior.

    A metrica NAO muda: as celulas-oraculo continuam no holdout de correcao, nao
    no da deteccao (invariante 2). Devolve {funcao, regra, historico, orcamento};
    `funcao`/`regra` sao a versao final (refinada ou a ultima valida).
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", SISTEMA_UPDATE), ("human", HUMANO_UPDATE)]
    )
    cadeia_update = prompt | agente_update

    usados: set[str] = set()
    rotulos: list[dict] = []
    historico: list[dict] = []
    orcamento = {"valores_rotulados": [], "indices_celulas": []}

    for i in range(iteracoes):
        amostrados = _amostrar_oraculo(
            funcao, valores_distintos, embedder, usados, amostras_por_iter, sujo_col
        )
        if not amostrados:
            historico.append({
                "iteracao": i + 1,
                "amostrados": [],
                "rotulos_novos": [],
                "feedback": _montar_feedback_det(funcao, rotulos),
                "codigo_antes": regra.codigo or "",
                "codigo_depois": regra.codigo or "",
                "mudou": False,
                "status": "sem_amostra: nenhum valor distinto novo para rotular",
            })
            continue

        marcas_amostrados = _classificar(funcao, list(amostrados))
        rotulos_novos = []
        for v, marcado in zip(amostrados, marcas_amostrados):
            usados.add(v)
            indices = sujo_col.index[sujo_col == v]
            eh_erro_real = bool((limpo_col.loc[indices] != v).any())
            orcamento["valores_rotulados"].append(v)
            orcamento["indices_celulas"].extend(int(x) for x in indices)
            classificacao_atual = "sujo" if marcado else "limpo"
            rot = {"valor": v, "eh_erro_real": eh_erro_real}
            rotulos.append(rot)
            rotulos_novos.append({**rot, "classificacao_atual": classificacao_atual})

        feedback = _montar_feedback_det(funcao, rotulos)
        codigo_antes = regra.codigo or ""

        funcao_nova, regra_nova, erro = _tentar_update(
            cadeia_update, coluna, codigo_antes, feedback
        )
        if funcao_nova is not None:
            # GUARDA (A) por iteracao: a regra revisada ja passou sandbox+fumaca
            # (roda sem quebrar), mas pode ser AMPLA DEMAIS. Avalia o CALLABLE
            # `funcao_nova` (NAO `regra_nova`) no oraculo cumulativo desta
            # iteracao (`rotulos`, ja com o append acima). Precisao < LIMITE ->
            # REJEITA pelo mesmo caminho do fallback (mantem a resident anterior,
            # funcao/regra intactos), com status distinguivel de "rejeitada:
            # {erro}". Precisao None (marca 0) ou == LIMITE -> aceita.
            precisao = _precisao_no_oraculo(funcao_nova, rotulos)
            if precisao is not None and precisao < config.LIMITE_PRECISAO_DETECCAO:
                status = (
                    f"rejeitada por precisao {precisao:.3f}"
                    f"<{config.LIMITE_PRECISAO_DETECCAO}"
                )
                mudou = False
            else:
                funcao, regra = funcao_nova, regra_nova
                status = "aplicada"
                mudou = (regra.codigo or "") != codigo_antes
        else:
            status = f"rejeitada: {erro}"
            mudou = False

        historico.append({
            "iteracao": i + 1,
            "amostrados": list(amostrados),
            "rotulos_novos": rotulos_novos,
            "feedback": feedback,
            "codigo_antes": codigo_antes,
            "codigo_depois": regra.codigo or "",
            "mudou": mudou,
            "status": status,
        })

    # PISO (B) apos o laco -- resolve a Armadilha 2 (resident ampla persiste).
    # Rejeitar (A) mantem a anterior; se a resident FINAL ja e' ampla (ex.: a
    # 1-passe it0 saiu ampla e todas as revisoes foram rejeitadas) o loop
    # entregaria o dano. Avalia o CALLABLE residente `funcao` (NAO `regra`) no
    # oraculo cumulativo: precisao < LIMITE (com >=1 clean marcado) -> substitui
    # a resident por DETECTA_NADA (dano 0), espelhando gerar_regra_deteccao
    # (materializa DETECTA_NADA e faz regra.codigo = DETECTA_NADA). Precisao None
    # (marca 0) ou == LIMITE -> resident preservada.
    precisao_final = _precisao_no_oraculo(funcao, rotulos)
    if precisao_final is not None and precisao_final < config.LIMITE_PRECISAO_DETECCAO:
        codigo_antes_piso = regra.codigo or ""
        feedback_piso = _montar_feedback_det(funcao, rotulos)
        funcao = sandbox.materializar(
            DETECTA_NADA, nome_funcao="detectar", nome_argumento="col",
            series_mode=True,
        )
        regra.codigo = DETECTA_NADA
        # Entry com O MESMO CONJUNTO DE CHAVES das entradas do laco -- o relatorio
        # (relatorio._bloco_historico_deteccao) acessa passo['iteracao'] direto;
        # um entry sem essa chave quebraria o relatorio ao vivo.
        historico.append({
            "iteracao": len(historico) + 1,
            "amostrados": [],
            "rotulos_novos": [],
            "feedback": feedback_piso,
            "codigo_antes": codigo_antes_piso,
            "codigo_depois": DETECTA_NADA,
            "mudou": True,
            "status": (
                f"piso: resident ampla (precisao {precisao_final:.3f}"
                f"<{config.LIMITE_PRECISAO_DETECCAO}) -> DETECTA_NADA"
            ),
        })

    orcamento["indices_celulas"] = sorted(set(orcamento["indices_celulas"]))
    orcamento["n_valores"] = len(orcamento["valores_rotulados"])
    orcamento["n_celulas"] = len(orcamento["indices_celulas"])
    return {
        "funcao": funcao,
        "regra": regra,
        "historico": historico,
        "orcamento": orcamento,
    }


def construir_mascara(
    df: pd.DataFrame, funcoes_por_coluna: dict, log: list | None = None
) -> pd.DataFrame:
    """Aplica `detectar(col)` por COLUNA e devolve mascara 0/1 no shape do df.

    Contrato Series (plano secao 6/7, MANTENDO a assinatura): por coluna chama
    `funcao(df[coluna])` UMA vez e le o resultado POSICIONALMENTE. Exige uma
    pd.Series booleana do MESMO tamanho da coluna; coage o dtype com
    `res.fillna(False).astype(bool)` e atribui a coluna da mascara por POSICAO
    via `.to_numpy()` (nunca boolean-index por indice, que desalinha se a funcao
    devolver indice proprio). Politica estrita, agora com granularidade
    POR-COLUNA (declarada): se a funcao lanca, nao devolve Series ou devolve
    tamanho errado, a coluna INTEIRA fica 0 e uma entrada entra no log. NAO le
    clean -- auditavel: nenhum import de clean, nenhum parametro clean.
    """
    if log is None:
        log = []
    mascara = pd.DataFrame(0, index=df.index, columns=df.columns, dtype=int)
    n = len(df)
    for coluna, funcao in funcoes_por_coluna.items():
        if funcao is None or coluna not in df.columns:
            continue
        try:
            res = funcao(df[coluna])
        except Exception as exc:  # noqa: BLE001 -- deteccao nao pode derrubar o pipeline
            log.append(
                {"coluna": coluna,
                 "motivo": f"excecao {type(exc).__name__}: {exc}"}
            )
            continue
        if not isinstance(res, pd.Series):
            log.append(
                {"coluna": coluna,
                 "motivo": f"retorno nao-Series ({type(res).__name__})"}
            )
            continue
        if len(res) != n:
            log.append(
                {"coluna": coluna,
                 "motivo": f"tamanho divergente: esperado {n}, veio {len(res)}"}
            )
            continue
        try:
            marca = res.fillna(False).astype(bool).to_numpy()
        except Exception as exc:  # noqa: BLE001 -- coercao nao pode derrubar o pipeline
            log.append(
                {"coluna": coluna,
                 "motivo": f"coercao bool falhou ({type(exc).__name__}: {exc})"}
            )
            continue
        mascara[coluna] = marca.astype(int)
    return mascara
