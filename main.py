"""CLI da POC: le os argumentos, chama a espinha do pipeline e imprime o resultado."""
import argparse
import sys
from pathlib import Path

# Windows abre o stdout em cp1252 (mojibake); isto forca utf-8 so' na impressao.
for fluxo in (sys.stdout, sys.stderr):
    if hasattr(fluxo, "reconfigure"):
        fluxo.reconfigure(encoding="utf-8", errors="replace")

from limpeza import config, dados, pipeline  # noqa: E402


def _argumentos(argv=None):
    ap = argparse.ArgumentParser(
        description="POC: geracao de limpador autonomo a partir do dirty e do clean")
    ap.add_argument("--sujo", required=True, help="CSV com os dados sujos")
    ap.add_argument("--limpo", required=True, help="CSV de referencia (orcamento de rotulos)")
    ap.add_argument("--colunas", default="todas",
                    help="lista separada por virgula, ou 'todas'")
    ap.add_argument("--modelo", default=config.MODELO_LLM)
    ap.add_argument("--saida", default="runs", help="pasta base onde o run e' escrito")
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
    config.MODELO_LLM = args.modelo
    config.ITERACOES_DETECCAO = args.iteracoes_deteccao
    config.AMOSTRAS_POR_ITERACAO = args.amostras_iter
    config.DIR_RUNS = Path(args.saida)


def _imprimir(limpador, medida: dict) -> None:
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

    escolha = args.colunas.strip()
    colunas = None if escolha.lower() == "todas" else [
        c.strip() for c in escolha.split(",") if c.strip()]
    try:
        limpador, medida = pipeline.gerar_limpador(
            caminho_sujo=args.sujo, caminho_limpo=args.limpo, colunas=colunas)
    except dados.DadosInvalidos as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    _imprimir(limpador, medida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
