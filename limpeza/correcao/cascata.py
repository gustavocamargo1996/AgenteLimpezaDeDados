"""Etapa 4: cascata de correcao codigo -> FD, com gate de 100% e escalonamento celula a celula."""
from .. import config, erros
from ..tipos import Correcao
from . import fd as fd_mod, regras


def _gate_codigo(funcao, rows: list[dict]) -> bool:
    """Reprova a regra inteira se errar 1 unico rotulado. Ver docs/DECISOES.md#gates-de-100."""
    # Excecao aqui sobe ate o catch por coluna em rodar_cascata (ja e' a fronteira certa).
    if funcao is None:
        return False
    for r in rows:
        saida = funcao(r["sujo"])
        saida = "" if saida is None else str(saida)
        if saida != r["limpo"]:
            return False
    return True


def _camada_codigo(coluna, rotulados, agentes):
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
    dependencia=None,
    mi_threshold: float | None = None,
) -> tuple[dict, dict]:
    """Roda a cascata numa coluna e devolve (correcoes_por_indice, trilha diagnostica)."""
    indices_marcados = list(mascara_col[mascara_col == 1].index)
    rows = rotulados["rows"]

    correcoes: dict = {}
    # Nenhuma camada resolve -> fica flagada, nunca com valor inventado. Ver docs/DECISOES.md#flag-em-vez-de-chute.
    trilha_celula = {idx: "nao_resolvida" for idx in indices_marcados}
    contagem = {"codigo": 0, "fd": 0, "nao_resolvida": 0}
    pendentes = list(indices_marcados)
    log: list = []

    regra, funcao, codigo_str = _camada_codigo(coluna, rotulados, agentes)
    gate_codigo = _gate_codigo(funcao, rows)
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

    fd = dependencia  # ja proposta e aprovada na deteccao; nao propor de novo aqui
    gate_fd = None
    if fd is None and pendentes:
        candidatos = fd_mod.candidatos_determinantes(df, coluna, limiar=mi_threshold)
        if candidatos:
            fd = fd_mod.propor_fd(coluna, rows, candidatos, df, agente=agentes.get("fd"))
    if fd is not None and pendentes:
        # Revalida mesmo a FD ja aprovada: a mascara aqui e' a combinada, com informacao nova.
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
                dependencia=trabalho.dependencia,
                mi_threshold=config.MI_THRESHOLD,
            )
        except Exception as exc:  # noqa: BLE001 -- 1 coluna ruim nao derruba o run
            print(f"  {nome}: cascata falhou ({erros.descrever(exc)}) -> celulas "
                  "marcadas viram flag", flush=True)
            correcoes, trilha = {}, _trilha_de_falha(nome, mascara[nome], exc)
        for idx, valor in correcoes.items():
            corrigido.at[idx, nome] = valor
        trabalho.correcao = Correcao(passos=_passos(trilha), trilha=trilha,
                                     cadeia=trilha["cadeia"])
    return corrigido


def _rotulados(trabalho) -> dict:
    amostra = trabalho.amostra
    return {
        "rows": amostra.rotulados,
        "itens": _itens(trabalho.coluna, amostra),
        "total_linhas": amostra.total_linhas,
        "total_distintos": amostra.total_distintos,
    }


def _itens(coluna, amostra) -> list[dict]:
    """Amostra do especificador: cada representante com seu par limpo e a ambiguidade."""
    # Um sujo pode mapear para varios limpos; nao reduzir a moda. Ver docs/DECISOES.md#orcamento-vs-gabarito.
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
    # Ordem fixa: codigo, depois FD (podem coexistir; sem nenhum a coluna so' flaga).
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
    idx_marcados = list(mascara_col[mascara_col == 1].index)
    motivo = f"cascata falhou: {erros.descrever(exc)}"
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
