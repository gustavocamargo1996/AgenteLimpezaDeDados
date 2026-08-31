"""O loop de refinamento da deteccao com oraculo (--iteracoes-deteccao N>1)."""
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from .. import config, sandbox
from ..esquemas import RegraDeteccao
from .oraculo import _amostrar_oraculo, _classificar
from .regra import _AMOSTRAS_FUMACA, DETECTA_NADA

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
