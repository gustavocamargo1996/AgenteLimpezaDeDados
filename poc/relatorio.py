"""Artefatos por execucao: .md para ler, .json para diffar."""
import json
from pathlib import Path

from . import config


def criar_pasta(modo: str, carimbo: str) -> Path:
    pasta = config.DIR_RUNS / f"{carimbo}__{modo}"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def _fmt_pct(v) -> str:
    return "n/d" if v is None else f"{v:.1%}"


def _bloco_historico_deteccao(historico: list, orcamento) -> str:
    """Historico do loop de refinamento de UMA coluna. SO' e' chamado quando N>1.

    Mostra, por iteracao: o que o oraculo rotulou, se a regra mudou (com o
    status) e o codigo resultante. Fecha com o orcamento (custo) do oraculo.
    Com N=1 esta funcao NAO e' chamada, entao a cadeias_deteccao.md fica
    byte-identica ao 1-passe (invariante 7 / plano secao 5).
    """
    linhas = []
    if orcamento:
        linhas.append(
            f"**Orcamento do oraculo:** {orcamento.get('n_valores', 0)} valores "
            f"distintos / {orcamento.get('n_celulas', 0)} celulas rotuladas "
            "(custo do loop -- NAO entra no holdout da metrica).\n"
        )
    linhas.append("### Historico de refinamento\n")
    for passo in historico:
        amostrados = ", ".join(f'"{v}"' for v in passo.get("amostrados", [])) or "(nenhum)"
        linhas.append(
            f"**Iteracao {passo['iteracao']}** -- amostrados: {amostrados} -- "
            f"mudou: {'sim' if passo.get('mudou') else 'nao'} "
            f"({passo.get('status', '')})\n"
        )
        for r in passo.get("rotulos_novos", []):
            linhas.append(
                f"- oraculo: `{r['valor']}` -> "
                f"{'ERRO' if r['eh_erro_real'] else 'correto'} "
                f"(regra previa dizia: {r.get('classificacao_atual', '?')})"
            )
        linhas.append("\n```python\n" + (passo.get("codigo_depois", "") or "") + "\n```\n")
    return "\n".join(linhas)


def escrever_e2e(pasta: Path, run: dict) -> None:
    """Artefatos da geracao do limpador.

    `run` traz: dataset, modelo, sufixo, colunas, mascara (DataFrame), corrigido
    (DataFrame final), deteccao_metricas, correcao_metricas, trilhas,
    regras_deteccao, log_mascara. Quando o loop de refinamento roda (N>1),
    tambem: iteracoes_deteccao (int), historico_deteccao e orcamento_deteccao
    (por coluna). Com N=1 esses campos ficam vazios e a cadeias_deteccao.md sai
    byte-identica ao 1-passe.
    """
    run["mascara"].to_csv(pasta / "mascara.csv", index=False)
    run["corrigido"].to_csv(pasta / "correcoes.csv", index=False)

    (pasta / "deteccao_metricas.json").write_text(
        json.dumps(run["deteccao_metricas"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (pasta / "correcao_metricas.json").write_text(
        json.dumps(run["correcao_metricas"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # -------- cascata.md: qual camada resolveu cada coluna + contagem ----------
    linhas = [
        f"# Cascata de correcao -- `{run['dataset']}`"
        + (f" (sufixo {run['sufixo']})" if run.get("sufixo") else ""),
        f"\nModelo `{run['modelo']}`\n",
        "\n| Coluna | Marcadas | Codigo | FD | Nao resolvida | Soma confere |",
        "|---|---|---|---|---|---|",
    ]
    for nome in run["colunas"]:
        t = run["trilhas"][nome]
        c = t["contagem"]
        soma = c["codigo"] + c["fd"] + c["nao_resolvida"]
        confere = "sim" if soma == t["marcadas"] else f"NAO ({soma}!={t['marcadas']})"
        linhas.append(
            f"| `{nome}` | {t['marcadas']} | {c['codigo']} | {c['fd']} | "
            f"{c['nao_resolvida']} | {confere} |"
        )

    for nome in run["colunas"]:
        t = run["trilhas"][nome]
        linhas.append(f"\n---\n\n## `{nome}`\n")
        linhas.append(
            f"- gate camada 1 (codigo, `{t['regra_codigo_tipo']}`): "
            f"{'PASSOU' if t['gate_codigo'] else 'reprovou'}"
        )
        if t["gate_fd"] is None:
            linhas.append("- gate camada 2 (FD): nao acionado (sem candidato por MI ou nada a escalar)")
        else:
            fd = t["fd"] or {}
            linhas.append(
                f"- gate camada 2 (FD `{fd.get('determinante')}` -> `{fd.get('dependente')}`): "
                f"{'PASSOU' if t['gate_fd'] else 'reprovou (nenhuma celula alterada)'}"
            )
        if t["log"]:
            linhas.append(f"- log ({len(t['log'])} entradas): valores nao enviados / erros registrados")

    (pasta / "cascata.md").write_text("\n".join(linhas), encoding="utf-8")

    # -------- cadeias_deteccao.md ---------------------------------------------
    # O bloco base (por coluna) e' identico ao 1-passe. O historico do loop so'
    # e' acrescentado quando N>1 -- com N=1 a saida fica byte-identica ao atual.
    n_iter = int(run.get("iteracoes_deteccao", 1) or 1)
    historico_det = run.get("historico_deteccao", {}) or {}
    orcamento_det = run.get("orcamento_deteccao", {}) or {}
    partes = [
        f"# Cadeias de deteccao -- `{run['dataset']}`"
        + (f" (sufixo {run['sufixo']})" if run.get("sufixo") else ""),
        f"\nDeteccao INTRA-COLUNA (a regra ve um escalar, nao a linha). "
        f"Modelo `{run['modelo']}`.\n",
    ]
    for nome in run["colunas"]:
        regra = run["regras_deteccao"][nome]
        det = run["deteccao_metricas"].get(nome, {})
        partes.append(f"\n---\n\n## `{nome}`\n")
        partes.append(
            f"**Erro provavel:** {'sim' if regra.erro_provavel else 'nao'} &nbsp;|&nbsp; "
            f"**P:** {_fmt_pct(det.get('precisao'))} &nbsp;|&nbsp; "
            f"**R:** {_fmt_pct(det.get('recall'))} &nbsp;|&nbsp; "
            f"**F1:** {_fmt_pct(det.get('f1'))}\n"
        )
        partes.append(f"**Criterio (regex documental):** `{regra.condicao_regex or '(nenhuma)'}`\n")
        partes.append(f"**Cadeia:**\n\n{regra.cadeia}\n")
        partes.append("**Codigo de deteccao:**\n\n```python\n" + (regra.codigo or "") + "\n```\n")
        if n_iter > 1:
            partes.append(_bloco_historico_deteccao(
                historico_det.get(nome, []), orcamento_det.get(nome)))

    (pasta / "cadeias_deteccao.md").write_text("\n".join(partes), encoding="utf-8")
