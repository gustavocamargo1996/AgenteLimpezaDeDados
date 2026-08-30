"""Orquestracao da cascata de correcao: codigo -> FD -> fallback por celula.

Sobre as celulas com mascara == 1, cada camada tem PORTAO contra o orcamento
rotulado (sujos + limpos) -- exceto a camada 3, que e' ultimo recurso.

Regra de escalonamento (como correction.py:1000-1002 do ZeroDC, onde
`mask = dirty != corrections` zera a deteccao das celulas que mudaram):
  - uma celula so' e' considerada resolvida por uma camada se a camada ALTEROU
    seu valor;
  - celula que a camada deixou intacta ESCALA para a proxima;
  - gate reprovado -> TODAS as celulas marcadas escalam.

A camada 3 e' terminal: uma celula ENVIADA ao fallback dentro do teto conta como
resolvida por fallback (decisao final, mesmo que o valor devolvido coincida com o
sujo); celulas alem do teto ficam nao-resolvidas (sujas) e sao logadas.

Invariante contabil (plano secao 8): por coluna,
    contagem[codigo] + contagem[fd] + contagem[fallback] + contagem[nao_resolvida]
    == numero de celulas marcadas.
"""
from . import config, contexto, fallback_celula, fd as fd_mod, gerador_codigo, identificador_regras


def _gate_codigo(funcao, rows: list[dict]) -> bool:
    """100% no conjunto rotulado (sujos e limpos). funcao None -> reprova."""
    if funcao is None:
        return False
    for r in rows:
        try:
            saida = funcao(r["sujo"])
            saida = "" if saida is None else str(saida)
        except Exception:
            return False
        if saida != r["limpo"]:
            return False
    return True


def _camada_codigo(coluna, rotulados, agentes):
    """Reusa agente 1 (budget) + agente 2. Devolve (regra, funcao, codigo_str).

    O terceiro elemento e' o TEXTO do `corrigir(valor)` gerado (`traducao`
    ["codigo"]), antes descartado. `rodar_cascata` o captura para empacotar o
    limpador autonomo -- e' a copia fiel da correcao de codigo aplicada quando o
    gate passa.
    """
    regra = identificador_regras.especificar(
        coluna=coluna,
        itens=rotulados["itens"],
        modo="budget",
        total_linhas=rotulados["total_linhas"],
        total_distintos=rotulados["total_distintos"],
        agente=agentes.get("especificador"),
    )
    traducao = gerador_codigo.traduzir(regra, agente=agentes.get("codigo"))
    return regra, traducao["funcao"], traducao["codigo"]


def rodar_cascata(
    coluna: str,
    df,
    mascara_col,
    rotulados: dict,
    mascara_completa,
    agentes: dict,
    mi_threshold: float | None = None,
    limite_fallback: int | None = None,
) -> tuple[dict, dict]:
    """Devolve (correcoes_col, trilha).

    correcoes_col: {indice: valor_final} das celulas resolvidas (codigo/fd/
    fallback). Celulas nao resolvidas NAO entram -- ficam sujas no fim.
    trilha: diagnostico por celula e por camada (ver invariante contabil).
    """
    indices_marcados = list(mascara_col[mascara_col == 1].index)
    rows = rotulados["rows"]

    correcoes: dict = {}
    trilha_celula = {idx: "nao_resolvida" for idx in indices_marcados}
    contagem = {"codigo": 0, "fd": 0, "fallback": 0, "nao_resolvida": 0}
    pendentes = list(indices_marcados)
    log: list = []

    # -------- Camada 1: codigo -------------------------------------------------
    regra, funcao, codigo_str = _camada_codigo(coluna, rotulados, agentes)
    gate_codigo = _gate_codigo(funcao, rows)
    # Captura o corretor SEMPRE que o gate passou (mesmo que mude 0 celulas -- e'
    # inocuo e reproduz). None quando reprovou: nao entra no plano do limpador.
    codigo_correcao = codigo_str if gate_codigo else None
    if gate_codigo and pendentes:
        restantes = []
        for idx in pendentes:
            antigo = df.at[idx, coluna]
            try:
                novo = funcao(antigo)
                novo = "" if novo is None else str(novo)
            except Exception:
                novo = antigo
            if novo != antigo:
                correcoes[idx] = novo
                trilha_celula[idx] = "codigo"
                contagem["codigo"] += 1
            else:
                restantes.append(idx)  # intacta escala
        pendentes = restantes

    # -------- Camada 2: FD -----------------------------------------------------
    fd = None
    gate_fd = None
    candidatos = contexto.candidatos_determinantes(df, coluna, limiar=mi_threshold)
    if pendentes and candidatos:
        fd = fd_mod.propor_fd(coluna, rows, candidatos, df, agente=agentes.get("fd"))
        gate_fd = fd_mod.validar_fd(fd, rows, df, mascara_completa, coluna)
        if gate_fd:
            mudancas = fd_mod.aplicar_fd(fd, df, pendentes, mascara_completa, coluna)
            restantes = []
            for idx in pendentes:
                if idx in mudancas:
                    correcoes[idx] = mudancas[idx]
                    trilha_celula[idx] = "fd"
                    contagem["fd"] += 1
                else:
                    restantes.append(idx)  # intacta escala
            pendentes = restantes

    # -------- Camada 3: fallback por celula (terminal) -------------------------
    # So' roda sob config.USAR_FALLBACK (default False). Desligada, os pendentes
    # que codigo/FD nao cobriram permanecem 'nao_resolvida' (flag), contagem
    # 'fallback' fica 0 e nenhuma chamada de LLM da camada 3 acontece.
    fallback_chamadas = 0
    if config.USAR_FALLBACK and pendentes:
        resultado = fallback_celula.aplicar_fallback(
            coluna, df, pendentes, mascara_completa, candidatos,
            agente=agentes.get("fallback"), limite=limite_fallback,
        )
        fallback_chamadas = resultado["chamadas"]
        log.extend(resultado["log"])
        enviados = set(resultado["correcoes"].keys())
        restantes = []
        for idx in pendentes:
            if idx in enviados:
                correcoes[idx] = resultado["correcoes"][idx]
                trilha_celula[idx] = "fallback"
                contagem["fallback"] += 1
            else:
                restantes.append(idx)  # alem do teto: fica suja
        pendentes = restantes

    # -------- Sobra: nao resolvida ---------------------------------------------
    contagem["nao_resolvida"] = len(pendentes)

    trilha = {
        "coluna": coluna,
        "marcadas": len(indices_marcados),
        "contagem": contagem,
        "trilha_celula": trilha_celula,
        "gate_codigo": gate_codigo,
        "codigo_correcao": codigo_correcao,
        "regra_codigo_tipo": regra.transformacao.tipo,
        "gate_fd": gate_fd,
        "fd": fd.model_dump() if fd is not None else None,
        "fallback_chamadas": fallback_chamadas,
        "log": log,
    }
    return correcoes, trilha
