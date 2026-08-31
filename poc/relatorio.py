"""Artefatos por execucao: .md para ler, .json para diffar."""
import json
from pathlib import Path

from . import config


def criar_pasta(modo: str, carimbo: str) -> Path:
    pasta = config.DIR_RUNS / f"{carimbo}__{modo}"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def _linha_medida(m) -> str:
    acerto = "n/d" if m.taxa_acerto is None else f"{m.taxa_acerto:.1%} ({m.acertos}/{m.erradas})"
    dano = "n/d" if m.taxa_dano is None else f"{m.taxa_dano:.1%} ({m.dano}/{m.corretas})"
    return f"| {m.escopo} | {m.celulas} | {acerto} | {dano} |"


def _bloco_cadeia(regra, resultado) -> str:
    c = regra.cadeia_de_pensamento
    t = regra.transformacao
    aviso = ""
    if resultado.excecoes:
        aviso += (
            f"\n> **{resultado.excecoes} excecoes lancadas pelo codigo gerado** — nesses "
            "casos o valor foi devolvido intacto, entao contam como nao-corrigido e nao "
            "como dano.\n"
        )
    if resultado.generalizacao.observacao:
        aviso += f"\n> **{resultado.generalizacao.observacao}**\n"
    if resultado.observacao:
        aviso += f"\n> **{resultado.observacao}**\n"

    return f"""## Coluna `{regra.coluna}`

**Erro detectado:** {"sim" if regra.erro_detectado else "nao"} &nbsp;|&nbsp; \
**Tipo:** `{regra.tipo_erro}` &nbsp;|&nbsp; \
**Transformacao:** `{t.tipo}` &nbsp;|&nbsp; \
**Confianca declarada:** {regra.confianca:.2f}

| escopo | celulas | acerto | dano |
|---|---|---|---|
{_linha_medida(resultado.generalizacao)}
{_linha_medida(resultado.cobertura)}

`generalizacao` = so' valores que o agente nunca viu &nbsp;|&nbsp; \
`cobertura` = coluna inteira menos as {resultado.linhas_mostradas} linhas exibidas
{aviso}

### Cadeia de pensamento

**1. Observacao da amostra**
{c.observacao_da_amostra}

**2. Identificacao do padrao**
{c.identificacao_do_padrao}

**3. Formulacao da regra**
{c.formulacao_da_regra}

**4. Limites da regra**
{c.limites_da_regra}

### Regra especificada
{regra.descricao_padrao}

- condicao de aplicacao: `{regra.condicao_regex or "(nenhuma)"}`
- transformacao: `{t.tipo}` -> padrao=`{t.padrao}` substituicao=`{t.substituicao}` \
mapa=`{"sim" if t.mapa else "nao"}` valor=`{t.valor}`

---
"""


def escrever(pasta: Path, modo: str, itens: list[dict]) -> None:
    """`itens` = [{regra, traducao, resultado}, ...] na ordem das colunas."""
    cabecalho = (
        f"# Cadeias de pensamento -- modo `{modo}`\n\n"
        f"Dataset `{config.DATASET}` | modelo `{config.MODELO_LLM}` | "
        f"{len(itens)} colunas\n\n"
        + (
            "Modo **blind**: o agente viu apenas valores sujos e teve que inferir "
            "o defeito sozinho.\n\n"
            if modo == "blind"
            else "Modo **budget**: o agente viu os pares sujo->correto das celulas "
            "representativas.\n\n"
        )
        + "---\n\n"
    )
    corpo = "".join(_bloco_cadeia(i["regra"], i["resultado"]) for i in itens)
    (pasta / "cadeias.md").write_text(cabecalho + corpo, encoding="utf-8")

    (pasta / "regras.json").write_text(
        json.dumps([i["regra"].model_dump() for i in itens], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (pasta / "metricas.json").write_text(
        json.dumps([i["resultado"].dict() for i in itens], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # codigo.json existe para permitir `main.py --reavaliar` sem gastar API:
    # o corretores.py renomeia as funcoes e nao volta limpo para materializar.
    (pasta / "codigo.json").write_text(
        json.dumps(
            {
                i["regra"].coluna: {
                    "codigo": i["traducao"]["codigo"],
                    "nota": i["traducao"]["nota"],
                    "tentativas": i["traducao"]["tentativas"],
                    "rejeicoes": i["traducao"]["rejeicoes"],
                }
                for i in itens
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    partes = [f'"""Corretores gerados -- modo {modo}. Nao editar a mao."""\n']
    for i in itens:
        partes.append(
            f"\n# --- coluna: {i['regra'].coluna} ---\n"
            f"# nota do tradutor: {i['traducao']['nota']}\n"
            f"# tentativas: {i['traducao']['tentativas']}"
            + (f" | rejeicoes: {i['traducao']['rejeicoes']}" if i["traducao"]["rejeicoes"] else "")
            + "\n"
            + i["traducao"]["codigo"].replace("def corrigir(", f"def corrigir_{i['regra'].coluna}(")
            + "\n"
        )
    (pasta / "corretores.py").write_text("\n".join(partes), encoding="utf-8")


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
    """Artefatos do modo --e2e.

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


def comparativo(pasta_saida: Path, por_modo: dict[str, list[dict]]) -> Path:
    """Mesma coluna, os dois modos lado a lado -- e' aqui que a POC entrega."""
    linhas = ["# Comparativo `blind` vs `budget`\n"]

    linhas.append(
        "\n| Coluna | Modo | Erro | Transformacao | Acerto (geral.) | Dano (geral.) "
        "| Acerto (cobert.) | Dano (cobert.) |"
    )
    linhas.append("|---|---|---|---|---|---|---|---|")
    pct = lambda v: "n/d" if v is None else f"{v:.1%}"  # noqa: E731
    colunas = [i["regra"].coluna for i in next(iter(por_modo.values()))]
    for nome in colunas:
        for modo, itens in por_modo.items():
            item = next((x for x in itens if x["regra"].coluna == nome), None)
            if not item:
                continue
            r, res = item["regra"], item["resultado"]
            g, c = res.generalizacao, res.cobertura
            linhas.append(
                f"| `{nome}` | {modo} | {'sim' if r.erro_detectado else 'NAO'} | "
                f"`{r.transformacao.tipo}` | {pct(g.taxa_acerto)} | {pct(g.taxa_dano)} "
                f"| {pct(c.taxa_acerto)} | {pct(c.taxa_dano)} |"
            )

    for nome in colunas:
        linhas.append(f"\n---\n\n## `{nome}`\n")
        for modo, itens in por_modo.items():
            item = next((x for x in itens if x["regra"].coluna == nome), None)
            if not item:
                continue
            c = item["regra"].cadeia_de_pensamento
            linhas.append(f"### modo `{modo}`\n")
            linhas.append(f"**Padrao identificado:** {c.identificacao_do_padrao}\n")
            linhas.append(f"**Limites declarados:** {c.limites_da_regra}\n")

    destino = pasta_saida / "comparativo.md"
    destino.write_text("\n".join(linhas), encoding="utf-8")
    return destino
