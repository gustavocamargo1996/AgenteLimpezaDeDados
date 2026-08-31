"""CLI da POC: le os argumentos, chama a espinha e imprime o resultado.

Uso:
    python main.py                          # gera o limpador do dataset padrao
    python main.py --colunas ounces,state   # subconjunto
    python main.py --colunas todas          # todas as colunas, sem vies de selecao
"""
import argparse
import sys

# Windows abre o stdout em cp1252 e as cadeias saem com mojibake no terminal.
# Os arquivos .md ja saem em utf-8; isto conserta so' a impressao.
for fluxo in (sys.stdout, sys.stderr):
    if hasattr(fluxo, "reconfigure"):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

from limpeza import config, dados, pipeline  # noqa: E402


def _argumentos(argv=None):
    """Le a linha de comando."""
    ap = argparse.ArgumentParser(
        description="POC: geracao de limpador autonomo por dataset")
    ap.add_argument("--dataset", default=config.DATASET)
    ap.add_argument("--colunas", default=",".join(config.COLUNAS_PADRAO),
                    help="lista separada por virgula, ou 'todas'")
    ap.add_argument("--modelo", default=config.MODELO_LLM)
    ap.add_argument("--sufixo", default=None,
                    help="fatia do dataset: '300' usa {nome}_dirty_300.csv / _clean_300.csv")
    ap.add_argument("--iteracoes-deteccao", type=int, default=config.ITERACOES_DETECCAO,
                    dest="iteracoes_deteccao",
                    help="iteracoes de refinamento da deteccao com oraculo. "
                         "1 = 1-passe, sem loop; N>1 = active learning")
    ap.add_argument("--amostras-iter", type=int, default=config.AMOSTRAS_POR_ITERACAO,
                    dest="amostras_iter",
                    help="valores distintos que o oraculo rotula por iteracao "
                         "(metade previsto-sujo, metade previsto-limpo)")
    return ap.parse_args(argv)


def _aplicar_config(args) -> None:
    """Passa as escolhas da linha de comando para o config, que o pipeline le."""
    config.DATASET = args.dataset
    config.MODELO_LLM = args.modelo
    config.SUFIXO = args.sufixo
    config.ITERACOES_DETECCAO = args.iteracoes_deteccao
    config.AMOSTRAS_POR_ITERACAO = args.amostras_iter
    config.DIR_DATASET = config.ZERODC_DIR / "datasets" / args.dataset
    config.CSV_SUJO, config.CSV_LIMPO = config.caminhos_dataset(
        args.dataset, args.sufixo)


def _imprimir(limpador, medida: dict) -> None:
    """Resumo por coluna: P/R/F1 da deteccao e acerto/dano da correcao."""
    print(f"\n  [e2e] artefatos em {limpador.parent}")
    print(f"  [e2e] limpador autonomo: {limpador}\n")
    for nome, d in medida["deteccao"].items():
        c = medida["correcao"][nome]
        # P/R/F1 saem None quando nao ha erro real no conjunto medido.
        num = lambda v: " n/d" if v is None else f"{v:.2f}"  # noqa: E731
        pct = lambda v: "n/d" if v is None else f"{v:.1%}"  # noqa: E731
        print(f"  {nome:10s} | deteccao P={num(d['precisao'])} R={num(d['recall'])} "
              f"F1={num(d['f1'])} | correcao acerto={pct(c['taxa_acerto'])} "
              f"dano={pct(c['taxa_dano'])}")


def main(argv=None) -> int:
    args = _argumentos(argv)
    _aplicar_config(args)

    if not config.CSV_SUJO.exists():
        print(f"ERRO: nao encontrei {config.CSV_SUJO}", file=sys.stderr)
        return 1

    escolha = args.colunas.strip()
    colunas = None if escolha.lower() == "todas" else [
        c.strip() for c in escolha.split(",") if c.strip()]
    try:
        limpador, medida = pipeline.gerar_limpador(colunas=colunas)
    except dados.DadosInvalidos as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    _imprimir(limpador, medida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
