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


def _bloco_refinamento(historico: list, orcamento) -> str:
    """Historico do loop de refinamento de UMA coluna, so' emitido quando N>1."""
    # Por iteracao: o que o oraculo rotulou, se a regra mudou e o codigo
    # resultante. Fecha com o orcamento (custo) do oraculo.
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


def escrever(saida: Path, trabalhos: list, mascara, corrigido, medida: dict) -> None:
    """Escreve os artefatos do run: as duas tabelas, os dois .json e os dois .md."""
    mascara.to_csv(saida / "mascara.csv", index=False)
    corrigido.to_csv(saida / "correcoes.csv", index=False)

    (saida / "deteccao_metricas.json").write_text(
        json.dumps(medida["deteccao"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (saida / "correcao_metricas.json").write_text(
        json.dumps(medida["correcao"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (saida / "cascata.md").write_text(_cascata_md(trabalhos), encoding="utf-8")
    (saida / "cadeias_deteccao.md").write_text(_cadeias_md(trabalhos), encoding="utf-8")


def _titulo(assunto: str) -> str:
    """Cabecalho comum dos .md, com o dataset e a fatia do run."""
    marca = f" (sufixo {config.SUFIXO})" if config.SUFIXO else ""
    return f"# {assunto} -- `{config.DATASET}`{marca}"


def _cascata_md(trabalhos: list) -> str:
    """Qual camada resolveu cada coluna, com a contagem que fecha o invariante."""
    linhas = [
        _titulo("Cascata de correcao"),
        f"\nModelo `{config.MODELO_LLM}`\n",
        "\n| Coluna | Marcadas | Codigo | FD | Nao resolvida | Soma confere |",
        "|---|---|---|---|---|---|",
    ]
    for trabalho in trabalhos:
        t = trabalho.correcao.trilha
        c = t["contagem"]
        soma = c["codigo"] + c["fd"] + c["nao_resolvida"]
        confere = "sim" if soma == t["marcadas"] else f"NAO ({soma}!={t['marcadas']})"
        linhas.append(
            f"| `{trabalho.coluna.nome}` | {t['marcadas']} | {c['codigo']} | "
            f"{c['fd']} | {c['nao_resolvida']} | {confere} |"
        )

    for trabalho in trabalhos:
        t = trabalho.correcao.trilha
        linhas.append(f"\n---\n\n## `{trabalho.coluna.nome}`\n")
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

    return "\n".join(linhas)


def _cadeias_md(trabalhos: list) -> str:
    """Cadeia, criterio e codigo de deteccao de cada coluna, com o refinamento."""
    # O bloco por coluna e' o do 1-passe; o historico do loop so' entra com N>1.
    partes = [
        _titulo("Cadeias de deteccao"),
        f"\nDeteccao INTRA-COLUNA (a regra ve um escalar, nao a linha). "
        f"Modelo `{config.MODELO_LLM}`.\n",
    ]
    for trabalho in trabalhos:
        detector = trabalho.detector
        det = (trabalho.medida or {}).get("deteccao", {})
        partes.append(f"\n---\n\n## `{trabalho.coluna.nome}`\n")
        partes.append(
            f"**Erro provavel:** {'sim' if detector.erro_provavel else 'nao'} &nbsp;|&nbsp; "
            f"**P:** {_fmt_pct(det.get('precisao'))} &nbsp;|&nbsp; "
            f"**R:** {_fmt_pct(det.get('recall'))} &nbsp;|&nbsp; "
            f"**F1:** {_fmt_pct(det.get('f1'))}\n"
        )
        partes.append(f"**Criterio (regex documental):** `{detector.condicao_regex or '(nenhuma)'}`\n")
        partes.append(f"**Cadeia:**\n\n{detector.cadeia}\n")
        partes.append("**Codigo de deteccao:**\n\n```python\n" + (detector.codigo or "") + "\n```\n")
        if config.ITERACOES_DETECCAO > 1:
            partes.append(_bloco_refinamento(detector.historico, detector.orcamento))

    return "\n".join(partes)
