"""Orquestrador da POC.

    le o CSV -> KMeans escolhe representantes -> agente 1 especifica a regra
    (+ cadeia de pensamento) -> agente 2 traduz para codigo -> avalia em duas
    escalas de holdout -> escreve artefatos e imprime as cadeias.

Uso:
    python main.py                          # roda os dois modos e compara
    python main.py --modo budget            # so' um modo
    python main.py --colunas ounces,state   # subconjunto
    python main.py --colunas todas          # todas as colunas, sem vies de selecao
    python main.py --reavaliar runs/2026-07-23_1133__budget   # sem gastar API
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Windows abre o stdout em cp1252 e as cadeias saem com mojibake no terminal.
# Os arquivos .md ja saem em utf-8; isto conserta so' a impressao.
for fluxo in (sys.stdout, sys.stderr):
    if hasattr(fluxo, "reconfigure"):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

from poc import (  # noqa: E402
    amostragem, avaliacao, cascata, config, dados, deteccao, empacotar,
    fallback_celula, fd, gerador_codigo, identificador_regras, relatorio, sandbox,
)
from poc.embeddings import Embedder  # noqa: E402
from poc.esquemas import RegraCorrecao, RegraDeteccao  # noqa: E402


def _selecionar_representantes(coluna, embedder):
    """Selecao deterministica (KMeans random_state=0) e independente do modo.

    Por ser deterministica, uma reavaliacao reproduz exatamente os mesmos
    representantes de uma execucao anterior -- e' o que permite recalcular
    metricas sem chamar o LLM de novo.
    """
    matriz = embedder.codificar(coluna.valores_distintos)
    indices = amostragem.selecionar(matriz)
    primeira_ocorrencia = {
        valor: int(idx) for idx, valor in coluna.sujo.drop_duplicates().items()
    }
    valores = [coluna.valores_distintos[i] for i in indices]
    linhas = {primeira_ocorrencia[v] for v in valores if v in primeira_ocorrencia}
    return indices, set(valores), linhas


def _preparar_itens(coluna, indices_repr, contagem, modo):
    """Monta a amostra que vai ao agente 1.

    Cuidado nao-obvio no modo budget: um mesmo valor sujo pode corresponder a
    varios valores limpos diferentes. Em `state`, as 127 celulas vazias viram 38
    estados distintos. Reduzir isso a moda ("vazio -> CO") ensinaria uma regra
    falsa ao agente e ele produziria, com toda a logica do mundo, um corretor que
    escreve CO em tudo. Quando o mapeamento e' ambiguo, dizemos que e' ambiguo.
    """
    itens = []
    for indice in indices_repr:
        valor = coluna.valores_distintos[indice]
        item = {"sujo": valor, "frequencia": int(contagem.get(valor, 0))}
        if modo == "budget":
            distintos = coluna.limpo[coluna.sujo == valor].unique().tolist()
            item["limpo"] = distintos[0] if distintos else valor
            item["ambiguo"] = len(distintos) > 1
            item["limpos_distintos"] = len(distintos)
            item["exemplos_limpos"] = distintos[:4]
        itens.append(item)
    return itens


def _avaliar(coluna, modo, funcao, valores, linhas):
    holdout_valor, holdout_linha = dados.particionar(coluna, valores, linhas)
    return avaliacao.avaliar(
        coluna=coluna.nome,
        modo=modo,
        sujo=coluna.sujo,
        limpo=coluna.limpo,
        holdout_valor=holdout_valor,
        holdout_linha=holdout_linha,
        funcao=funcao,
        valores_mostrados=len(valores),
        linhas_mostradas=len(linhas),
        erradas_na_coluna=coluna.celulas_erradas,
    )


def processar_coluna(coluna, modo, embedder, agente_spec, agente_cod, verboso=True):
    indices_repr, valores, linhas = _selecionar_representantes(coluna, embedder)
    contagem = coluna.sujo.value_counts().to_dict()
    itens = _preparar_itens(coluna, indices_repr, contagem, modo)

    if verboso:
        print(f"  [{modo}] {coluna.nome}: {len(coluna.valores_distintos)} valores distintos, "
              f"{len(itens)} representantes -> agente 1...", flush=True)

    regra = identificador_regras.especificar(
        coluna=coluna.nome, itens=itens, modo=modo,
        total_linhas=coluna.total_celulas,
        total_distintos=len(coluna.valores_distintos),
        agente=agente_spec,
    )

    if verboso:
        print(f"  [{modo}] {coluna.nome}: regra `{regra.transformacao.tipo}` -> agente 2...", flush=True)

    traducao = gerador_codigo.traduzir(regra, agente=agente_cod)
    resultado = _avaliar(coluna, modo, traducao["funcao"], valores, linhas)
    return {"regra": regra, "traducao": traducao, "resultado": resultado}


# --------------------------------------------------------------------------- #
# Reavaliacao: recalcula metricas de uma execucao passada, sem chamar o LLM.
# --------------------------------------------------------------------------- #

_MARCADOR = re.compile(r"^# --- coluna: (.+?) ---$", re.MULTILINE)


def carregar_codigos(pasta: Path) -> dict[str, str]:
    """Le o codigo gerado. Prefere codigo.json; cai para o corretores.py legado.

    O fallback existe porque o `corretores.py` renomeia cada funcao para
    `corrigir_<coluna>` (senao as 5 colidiriam no mesmo arquivo), e o portao AST
    exige o nome exato `corrigir`. Entao o .py nao volta limpo para
    `sandbox.materializar` -- daqui o parse reverso que desfaz o rename. O
    `codigo.json` foi criado justamente para tornar esse caminho desnecessario;
    ele so' serve para execucoes anteriores a esse arquivo existir.
    """
    arquivo = pasta / "codigo.json"
    if arquivo.exists():
        return {k: v["codigo"] for k, v in json.loads(arquivo.read_text(encoding="utf-8")).items()}

    texto = (pasta / "corretores.py").read_text(encoding="utf-8")
    marcas = list(_MARCADOR.finditer(texto))
    codigos = {}
    for i, marca in enumerate(marcas):
        nome = marca.group(1)
        fim = marcas[i + 1].start() if i + 1 < len(marcas) else len(texto)
        trecho = texto[marca.end() : fim]
        inicio = trecho.find(f"def corrigir_{nome}(")
        if inicio == -1:
            continue
        codigos[nome] = trecho[inicio:].replace(f"def corrigir_{nome}(", "def corrigir(", 1).strip()
    return codigos


def reavaliar(pasta: Path, sujo, limpo, embedder) -> list[dict]:
    regras = [RegraCorrecao(**r) for r in json.loads((pasta / "regras.json").read_text(encoding="utf-8"))]
    codigos = carregar_codigos(pasta)
    modo = pasta.name.split("__")[-1]
    itens = []
    for regra in regras:
        coluna = dados.montar_coluna(sujo, limpo, regra.coluna)
        _, valores, linhas = _selecionar_representantes(coluna, embedder)
        codigo = codigos.get(regra.coluna, "")
        try:
            funcao = sandbox.materializar(codigo) if codigo else None
        except sandbox.CodigoRejeitado:
            funcao = None
        resultado = _avaliar(coluna, modo, funcao, valores, linhas)
        itens.append({
            "regra": regra,
            "traducao": {"codigo": codigo, "nota": "(reavaliado)", "tentativas": 0, "rejeicoes": []},
            "resultado": resultado,
        })
    return itens


def imprimir(por_modo: dict[str, list[dict]]) -> None:
    print("=" * 78)
    for modo, itens in por_modo.items():
        print(f"\n### MODO {modo.upper()}\n")
        for item in itens:
            regra, res = item["regra"], item["resultado"]
            g, c = res.generalizacao, res.cobertura
            fmt = lambda v: "n/d" if v is None else f"{v:.1%}"  # noqa: E731
            print(f"--- {regra.coluna} | erro={'sim' if regra.erro_detectado else 'NAO'} "
                  f"| {regra.transformacao.tipo}")
            print(f"      generalizacao: acerto={fmt(g.taxa_acerto):>6s} dano={fmt(g.taxa_dano):>6s}"
                  f"   ({g.erradas} erradas / {g.corretas} corretas em {g.celulas} celulas)")
            print(f"      cobertura    : acerto={fmt(c.taxa_acerto):>6s} dano={fmt(c.taxa_dano):>6s}"
                  f"   ({c.erradas} erradas / {c.corretas} corretas em {c.celulas} celulas)")
            if res.excecoes:
                print(f"      !! {res.excecoes} excecoes lancadas pelo codigo gerado "
                      f"(valor devolvido intacto nesses casos)")
            if g.observacao:
                print(f"      !! {g.observacao}")
            if res.observacao:
                print(f"      !! {res.observacao}")
            print(f"      limites: {regra.cadeia_de_pensamento.limites_da_regra[:150]}\n")


# --------------------------------------------------------------------------- #
# Modo end-to-end (--e2e): deteccao intra-coluna -> cascata -> avaliacao.
# --------------------------------------------------------------------------- #


def rodar_e2e(nomes, sujo, limpo, embedder, args) -> int:
    """Detecta a propria mascara (sem clean) e corrige por cascata.

    Fluxo por coluna: deteccao intra-coluna -> mascara 0/1 -> cascata
    codigo/FD/fallback com portao por camada -> metricas contra clean, excluindo
    o orcamento rotulado (holdout).
    """
    modelo = config.MODELO_LLM
    agentes = {
        "especificador": identificador_regras.construir_agente(modelo),
        "codigo": gerador_codigo.construir_agente(modelo),
        "fd": fd.construir_agente(modelo),
        "fallback": fallback_celula.construir_agente(modelo),
    }
    agente_det = deteccao.construir_agente(modelo)

    print(f"[e2e] dataset={args.dataset}"
          + (f" sufixo={args.sufixo}" if args.sufixo else "")
          + f" | {len(sujo)} linhas | colunas={nomes}", flush=True)
    print(f"[e2e] modelo={modelo} | limite_fallback={args.limite_fallback}"
          + (f" | iteracoes_deteccao={args.iteracoes_deteccao}"
             f" amostras_iter={args.amostras_iter}"
             if args.iteracoes_deteccao > 1 else "")
          + "\n", flush=True)

    # -------- 1. Deteccao intra-coluna (SO' sujo, sem clean) -------------------
    funcoes_deteccao, regras_deteccao = {}, {}
    orcamento_deteccao, historico_deteccao = {}, {}
    contexto_por_coluna = {}
    for nome in nomes:
        coluna = dados.montar_coluna(sujo, limpo, nome)
        indices_repr, valores, linhas = _selecionar_representantes(coluna, embedder)
        contagem = coluna.sujo.value_counts().to_dict()
        representantes_sujos = [coluna.valores_distintos[i] for i in indices_repr]
        print(f"  [e2e] {nome}: deteccao a partir de {len(representantes_sujos)} "
              f"representantes sujos...", flush=True)
        # Resiliencia por coluna: uma falha de LLM (timeout, LengthFinishReason,
        # parse) numa coluna NAO pode derrubar o dataset inteiro. Cai para
        # DETECTA_NADA (coluna nao marcada) e o run continua.
        try:
            saida_det = deteccao.gerar_regra_deteccao(
                coluna=nome,
                representantes_sujos=representantes_sujos,
                total_linhas=coluna.total_celulas,
                total_distintos=len(coluna.valores_distintos),
                contagem=contagem,
                agente=agente_det,
            )
        except Exception as exc:  # noqa: BLE001 -- 1 coluna ruim nao derruba o run
            print(f"  [e2e] {nome}: deteccao FALHOU ({type(exc).__name__}) "
                  "-> DETECTA_NADA (coluna nao marcada)", flush=True)
            saida_det = {
                "funcao": sandbox.materializar(
                    deteccao.DETECTA_NADA, nome_funcao="detectar",
                    nome_argumento="col", series_mode=True),
                "regra": RegraDeteccao(
                    erro_provavel=False, condicao_regex=None,
                    codigo=deteccao.DETECTA_NADA,
                    cadeia=f"deteccao falhou ({type(exc).__name__}); coluna nao marcada"),
            }
        funcoes_deteccao[nome] = saida_det["funcao"]
        regras_deteccao[nome] = saida_det["regra"]

        # Loop de refinamento com oraculo SO' quando iteracoes>1. O 1-passe
        # (default) nao entra aqui -- caminho identico ao atual (invariante 7).
        # A regra refinada SOBRESCREVE funcoes_deteccao/regras_deteccao ANTES de
        # construir_mascara (que roda apos este loop), senao a mascara usaria a
        # regra de 1-passe e o refino nao teria efeito (invariante 6).
        if args.iteracoes_deteccao > 1 and saida_det["regra"].codigo != deteccao.DETECTA_NADA:
            print(f"  [e2e] {nome}: refinando por {args.iteracoes_deteccao} "
                  f"iteracoes com oraculo...", flush=True)
            try:
                saida_ref = deteccao.refinar_regra_deteccao(
                    coluna=nome,
                    funcao=saida_det["funcao"],
                    regra=saida_det["regra"],
                    sujo_col=coluna.sujo,
                    limpo_col=coluna.limpo,
                    valores_distintos=coluna.valores_distintos,
                    embedder=embedder,
                    iteracoes=args.iteracoes_deteccao,
                    amostras_por_iter=args.amostras_iter,
                    agente_update=agente_det,
                )
                funcoes_deteccao[nome] = saida_ref["funcao"]
                regras_deteccao[nome] = saida_ref["regra"]
                orcamento_deteccao[nome] = saida_ref["orcamento"]
                historico_deteccao[nome] = saida_ref["historico"]
            except Exception as exc:  # noqa: BLE001 -- refino falhou: mantem a regra de 1-passe
                print(f"  [e2e] {nome}: refino FALHOU ({type(exc).__name__}) "
                      "-> mantem a regra de 1-passe", flush=True)

        itens = _preparar_itens(coluna, indices_repr, contagem, "budget")
        rows, holdout = dados.montar_rotulados(coluna, linhas)
        contexto_por_coluna[nome] = {
            "rotulados": {
                "rows": rows,
                "itens": itens,
                "total_linhas": coluna.total_celulas,
                "total_distintos": len(coluna.valores_distintos),
            },
            "holdout": holdout,
        }

    log_mascara: list = []
    mascara_completa = deteccao.construir_mascara(sujo, funcoes_deteccao, log_mascara)

    # -------- 2. Cascata por coluna -------------------------------------------
    corrigido = sujo.copy()
    trilhas = {}
    for nome in nomes:
        marcadas = int((mascara_completa[nome] == 1).sum())
        print(f"  [e2e] {nome}: {marcadas} celulas marcadas -> cascata...", flush=True)
        try:
            correcoes_col, trilha = cascata.rodar_cascata(
                coluna=nome,
                df=sujo,
                mascara_col=mascara_completa[nome],
                rotulados=contexto_por_coluna[nome]["rotulados"],
                mascara_completa=mascara_completa,
                agentes=agentes,
                mi_threshold=config.MI_THRESHOLD,
                limite_fallback=args.limite_fallback,
            )
        except Exception as exc:  # noqa: BLE001 -- 1 coluna ruim nao derruba o run
            print(f"  [e2e] {nome}: cascata FALHOU ({type(exc).__name__}) "
                  "-> coluna sem correcao (celulas marcadas viram flag)", flush=True)
            idx_marc = list(mascara_completa[nome][mascara_completa[nome] == 1].index)
            correcoes_col = {}
            trilha = {
                "coluna": nome, "marcadas": len(idx_marc),
                "contagem": {"codigo": 0, "fd": 0, "fallback": 0,
                             "nao_resolvida": len(idx_marc)},
                "trilha_celula": {idx: "nao_resolvida" for idx in idx_marc},
                "gate_codigo": False, "regra_codigo_tipo": "erro",
                "gate_fd": None, "fd": None, "codigo_correcao": None,
                "fallback_chamadas": 0,
                "log": [{"coluna": nome,
                         "motivo": f"cascata falhou: {type(exc).__name__}: {exc}"}],
            }
        for idx, valor in correcoes_col.items():
            corrigido.at[idx, nome] = valor
        trilhas[nome] = trilha

    # -------- 3. Avaliacao (exclui holdout do orcamento rotulado) -------------
    deteccao_metricas, correcao_metricas = {}, {}
    for nome in nomes:
        holdout = contexto_por_coluna[nome]["holdout"]
        deteccao_metricas[nome] = avaliacao.metricas_deteccao(
            mascara_completa[nome], sujo[nome], limpo[nome], holdout
        )
        correcao_metricas[nome] = avaliacao.metricas_correcao(
            corrigido[nome], sujo[nome], limpo[nome], holdout
        )

    carimbo = datetime.now().strftime("%Y-%m-%d_%H%M")
    pasta = relatorio.criar_pasta("e2e", carimbo)
    relatorio.escrever_e2e(pasta, {
        "dataset": args.dataset,
        "modelo": modelo,
        "sufixo": args.sufixo,
        "limite_fallback": args.limite_fallback,
        "colunas": nomes,
        "mascara": mascara_completa,
        "corrigido": corrigido,
        "deteccao_metricas": deteccao_metricas,
        "correcao_metricas": correcao_metricas,
        "trilhas": trilhas,
        "regras_deteccao": regras_deteccao,
        "log_mascara": log_mascara,
        "iteracoes_deteccao": args.iteracoes_deteccao,
        "orcamento_deteccao": orcamento_deteccao,
        "historico_deteccao": historico_deteccao,
    })

    # -------- 4. Empacotar o limpador autonomo --------------------------------
    # Por coluna: o codigo de deteccao (regras_deteccao[col].codigo) e um plano
    # de correcao ORDENADO -- [corrigir-codigo se gate_codigo] + [FD (det,dep) se
    # gate_fd]. Codigo e FD PODEM COEXISTIR (a trilha traz os dois). Coluna sem
    # nenhum -> so' detecta e flaga.
    deteccao_codigo = {nome: regras_deteccao[nome].codigo for nome in nomes}
    plano_correcao: dict = {}
    for nome in nomes:
        t = trilhas[nome]
        passos: list = []
        if t.get("codigo_correcao"):  # gate_codigo passou
            passos.append({"tipo": "codigo", "codigo": t["codigo_correcao"]})
        if t.get("gate_fd") and t.get("fd"):  # gate_fd passou
            passos.append({
                "tipo": "fd",
                "determinante": t["fd"]["determinante"],
                "dependente": t["fd"]["dependente"],
            })
        plano_correcao[nome] = passos
    caminho_limpador = empacotar.gerar_limpador(
        caminho=pasta / f"limpador_{args.dataset}_{carimbo}.py",
        dataset=args.dataset,
        detectores_codigo=deteccao_codigo,
        plano_correcao=plano_correcao,
        colunas=nomes,
    )

    print(f"\n  [e2e] artefatos em {pasta}")
    print(f"  [e2e] limpador autonomo: {caminho_limpador}\n")
    for nome in nomes:
        d, c = deteccao_metricas[nome], correcao_metricas[nome]
        fmt = lambda v: "n/d" if v is None else f"{v:.1%}"  # noqa: E731
        # deteccao pode ser n/d (mensuravel:false) quando nao ha erro real no
        # conjunto medido -- P/R/F1 vem None e nao formatam com :.2f.
        fdet = lambda v: " n/d" if v is None else f"{v:.2f}"  # noqa: E731
        orc = orcamento_deteccao.get(nome)
        extra = (f" | oraculo: {orc['n_valores']}v/{orc['n_celulas']}c" if orc else "")
        print(f"  {nome:10s} | deteccao P={fdet(d['precisao'])} R={fdet(d['recall'])} "
              f"F1={fdet(d['f1'])} | correcao acerto={fmt(c['taxa_acerto'])} "
              f"dano={fmt(c['taxa_dano'])}" + extra)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="POC: cadeias de pensamento para regras de correcao")
    ap.add_argument("--modo", choices=["blind", "budget", "ambos"], default="ambos")
    ap.add_argument("--dataset", default=config.DATASET)
    ap.add_argument("--colunas", default=",".join(config.COLUNAS_PADRAO),
                    help="lista separada por virgula, ou 'todas'")
    ap.add_argument("--modelo", default=config.MODELO_LLM)
    ap.add_argument("--reavaliar", metavar="PASTA", default=None,
                    help="recalcula metricas de uma execucao passada, sem chamar o LLM")
    ap.add_argument("--e2e", action="store_true",
                    help="modo end-to-end: deteccao intra-coluna + cascata de correcao")
    ap.add_argument("--sufixo", default=None,
                    help="fatia do dataset: '300' usa {nome}_dirty_300.csv / _clean_300.csv")
    ap.add_argument("--limite-fallback", type=int, default=config.LIMITE_FALLBACK,
                    dest="limite_fallback",
                    help="teto de chamadas de LLM na camada 3, por coluna (default config.LIMITE_FALLBACK)")
    ap.add_argument("--iteracoes-deteccao", type=int, default=config.ITERACOES_DETECCAO,
                    dest="iteracoes_deteccao",
                    help="(--e2e) iteracoes de refinamento da deteccao com oraculo. "
                         "1 (default) = 1-passe, sem loop; N>1 = active learning")
    ap.add_argument("--amostras-iter", type=int, default=config.AMOSTRAS_POR_ITERACAO,
                    dest="amostras_iter",
                    help="(--e2e) valores distintos que o oraculo rotula por iteracao "
                         "(metade previsto-sujo, metade previsto-limpo)")
    args = ap.parse_args()

    config.DATASET = args.dataset
    config.MODELO_LLM = args.modelo
    config.DIR_DATASET = config.ZERODC_DIR / "datasets" / args.dataset
    config.CSV_SUJO, config.CSV_LIMPO = config.caminhos_dataset(args.dataset, args.sufixo)

    if not config.CSV_SUJO.exists():
        print(f"ERRO: nao encontrei {config.CSV_SUJO}", file=sys.stderr)
        return 1

    sujo, limpo = dados.carregar()
    embedder = Embedder()

    if args.reavaliar:
        pastas = [Path(p) for p in args.reavaliar.split(",")]
        por_modo = {}
        for pasta in pastas:
            if not (pasta / "regras.json").exists():
                print(f"ERRO: {pasta}/regras.json nao existe", file=sys.stderr)
                return 1
            itens = reavaliar(pasta, sujo, limpo, embedder)
            modo = pasta.name.split("__")[-1]
            relatorio.escrever(pasta, modo, itens)
            por_modo[modo] = itens
            print(f"  reavaliado (sem API): {pasta}")
        if len(por_modo) > 1:
            print(f"  comparativo em {relatorio.comparativo(pastas[0].parent, por_modo)}")
        imprimir(por_modo)
        return 0

    if args.colunas.strip().lower() == "todas":
        nomes = [c for c in sujo.columns if c.lower() != "index"]
    else:
        nomes = [c.strip() for c in args.colunas.split(",") if c.strip()]
    faltando = [n for n in nomes if n not in sujo.columns]
    if faltando:
        print(f"ERRO: coluna(s) inexistente(s): {faltando}", file=sys.stderr)
        return 1

    if args.e2e:
        return rodar_e2e(nomes, sujo, limpo, embedder, args)

    modos = list(config.MODOS) if args.modo == "ambos" else [args.modo]
    print(f"dataset={args.dataset} | {len(sujo)} linhas | colunas={nomes} | modos={modos}")
    print(f"modelo={config.MODELO_LLM} | ~{len(nomes) * len(modos) * 2} chamadas de LLM\n")

    agente_spec = identificador_regras.construir_agente(config.MODELO_LLM)
    agente_cod = gerador_codigo.construir_agente(config.MODELO_LLM)

    carimbo = datetime.now().strftime("%Y-%m-%d_%H%M")
    por_modo, pastas = {}, []

    for modo in modos:
        itens_modo = []
        for nome in nomes:
            coluna = dados.montar_coluna(sujo, limpo, nome)
            itens_modo.append(processar_coluna(coluna, modo, embedder, agente_spec, agente_cod))
        pasta = relatorio.criar_pasta(modo, carimbo)
        relatorio.escrever(pasta, modo, itens_modo)
        por_modo[modo] = itens_modo
        pastas.append(pasta)
        print(f"\n  artefatos de `{modo}` em {pasta}\n")

    if len(por_modo) > 1:
        print(f"  comparativo em {relatorio.comparativo(pastas[0].parent, por_modo)}\n")

    imprimir(por_modo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
