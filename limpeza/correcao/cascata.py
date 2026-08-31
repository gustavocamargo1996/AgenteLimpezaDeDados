"""Orquestracao da cascata de correcao: codigo -> FD.

Sobre as celulas com mascara == 1, cada camada tem PORTAO contra o orcamento
rotulado (sujos + limpos).

Regra de escalonamento (onde `mask = dirty != corrections` zera a deteccao das
celulas que mudaram):
  - uma celula so' e' considerada resolvida por uma camada se a camada ALTEROU
    seu valor;
  - celula que a camada deixou intacta ESCALA para a proxima;
  - gate reprovado -> TODAS as celulas marcadas escalam.

Celula que nenhuma camada resolve fica nao-resolvida (sujo, flagada).

Invariante contabil: por coluna,
    contagem[codigo] + contagem[fd] + contagem[nao_resolvida]
    == numero de celulas marcadas.
"""
from .. import config
from ..tipos import Correcao
from . import fd as fd_mod, regras


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
    regra = regras.especificar(
        coluna=coluna,
        itens=rotulados["itens"],
        total_linhas=rotulados["total_linhas"],
        total_distintos=rotulados["total_distintos"],
        agente=agentes.get("especificador"),
    )
    traducao = regras.traduzir(regra, agente=agentes.get("codigo"))
    return regra, traducao["funcao"], traducao["codigo"]


def rodar_coluna(
    coluna: str,
    df,
    mascara_col,
    rotulados: dict,
    mascara_completa,
    agentes: dict,
    mi_threshold: float | None = None,
) -> tuple[dict, dict]:
    """Devolve (correcoes_col, trilha).

    correcoes_col: {indice: valor_final} das celulas resolvidas (codigo/fd).
    Celulas nao resolvidas NAO entram -- ficam sujas no fim.
    trilha: diagnostico por celula e por camada (ver invariante contabil).
    """
    indices_marcados = list(mascara_col[mascara_col == 1].index)
    rows = rotulados["rows"]

    correcoes: dict = {}
    trilha_celula = {idx: "nao_resolvida" for idx in indices_marcados}
    contagem = {"codigo": 0, "fd": 0, "nao_resolvida": 0}
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
    candidatos = fd_mod.candidatos_determinantes(df, coluna, limiar=mi_threshold)
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
        "cadeia": regra.descricao_padrao,
        "gate_fd": gate_fd,
        "fd": fd.model_dump() if fd is not None else None,
        "log": log,
    }
    return correcoes, trilha


def rodar_cascata(trabalhos: list, tabela, mascara, agentes: dict):
    """Corrige coluna a coluna, preenche `.correcao` e devolve a tabela corrigida."""
    corrigido = tabela.sujo.copy()
    for trabalho in trabalhos:
        nome = trabalho.coluna.nome
        marcadas = int((mascara[nome] == 1).sum())
        print(f"  {nome}: {marcadas} celulas marcadas -> cascata...", flush=True)
        try:
            correcoes, trilha = rodar_coluna(
                coluna=nome,
                df=tabela.sujo,
                mascara_col=mascara[nome],
                rotulados=_rotulados(trabalho),
                mascara_completa=mascara,
                agentes=agentes,
                mi_threshold=config.MI_THRESHOLD,
            )
        except Exception as exc:  # noqa: BLE001 -- 1 coluna ruim nao derruba o run
            print(f"  {nome}: cascata falhou ({type(exc).__name__}) -> celulas "
                  "marcadas viram flag", flush=True)
            correcoes, trilha = {}, _trilha_de_falha(nome, mascara[nome], exc)
        for idx, valor in correcoes.items():
            corrigido.at[idx, nome] = valor
        trabalho.correcao = Correcao(passos=_passos(trilha), trilha=trilha,
                                     cadeia=trilha["cadeia"])
    return corrigido


def _rotulados(trabalho) -> dict:
    """O orcamento rotulado da coluna, no formato que as duas camadas leem."""
    amostra = trabalho.amostra
    return {
        "rows": amostra.rotulados,
        "itens": _itens(trabalho.coluna, amostra),
        "total_linhas": amostra.total_linhas,
        "total_distintos": amostra.total_distintos,
    }


def _itens(coluna, amostra) -> list[dict]:
    """Amostra do especificador: cada representante com seu par limpo e a ambiguidade."""
    # Um mesmo valor sujo pode ter varios limpos (127 celulas vazias de `state`
    # viram 38 estados). Reduzir a moda ensinaria uma regra falsa ao agente.
    itens = []
    for valor in amostra.representantes:
        distintos = coluna.limpo[coluna.sujo == valor].unique().tolist()
        itens.append({
            "sujo": valor,
            "frequencia": int(coluna.contagem.get(valor, 0)),
            "limpo": distintos[0] if distintos else valor,
            "ambiguo": len(distintos) > 1,
            "limpos_distintos": len(distintos),
            "exemplos_limpos": distintos[:4],
        })
    return itens


def _passos(trilha: dict) -> list[dict]:
    """Plano ORDENADO da coluna: o corretor de codigo e depois a FD, se passaram."""
    # Codigo e FD podem coexistir; coluna sem nenhum so' detecta e flaga.
    passos: list[dict] = []
    if trilha.get("codigo_correcao"):
        passos.append({"tipo": "codigo", "codigo": trilha["codigo_correcao"]})
    if trilha.get("gate_fd") and trilha.get("fd"):
        passos.append({
            "tipo": "fd",
            "determinante": trilha["fd"]["determinante"],
            "dependente": trilha["fd"]["dependente"],
        })
    return passos


def _trilha_de_falha(coluna: str, mascara_col, exc: Exception) -> dict:
    """Trilha de uma coluna cuja cascata quebrou: tudo marcado fica nao-resolvido."""
    idx_marcados = list(mascara_col[mascara_col == 1].index)
    motivo = f"cascata falhou: {type(exc).__name__}: {exc}"
    return {
        "coluna": coluna,
        "marcadas": len(idx_marcados),
        "contagem": {"codigo": 0, "fd": 0, "nao_resolvida": len(idx_marcados)},
        "trilha_celula": {idx: "nao_resolvida" for idx in idx_marcados},
        "gate_codigo": False,
        "regra_codigo_tipo": "erro",
        "gate_fd": None,
        "fd": None,
        "codigo_correcao": None,
        "cadeia": motivo,
        "log": [{"coluna": coluna, "motivo": motivo}],
    }
