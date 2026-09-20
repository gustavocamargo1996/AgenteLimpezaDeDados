"""CLI da POC: le os argumentos, chama a espinha do pipeline e imprime o resultado."""
import argparse
import os
import sys
from pathlib import Path

# Windows abre o stdout em cp1252 (mojibake); isto forca utf-8 so' na impressao.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from limpeza import config, dados, pipeline  # noqa: E402


def _argumentos(argv=None):
    ap = argparse.ArgumentParser(
        description="POC: geracao de limpador autonomo a partir do dirty e do clean")
    ap.add_argument("--sujo", default=os.getenv("SUJO"),
                    help="CSV com os dados sujos (ou a variavel SUJO)")
    ap.add_argument("--limpo", default=os.getenv("LIMPO"),
                    help="CSV de referencia (ou a variavel LIMPO)")
    ap.add_argument("--colunas", default="todas",
                    help="lista separada por virgula, ou 'todas'")
    ap.add_argument("--modelo", default=None,
                    help="modelo para os quatro papeis. Sem ele vale MODELO_LLM "
                         "ou a sobreposicao por papel (MODELO_DETECCAO etc.)")
    ap.add_argument("--saida", default=os.getenv("SAIDA", "runs"),
                    help="pasta base onde o run e' escrito (ou a variavel SAIDA)")
    ap.add_argument("--iteracoes-deteccao", type=int, default=config.ITERACOES_DETECCAO,
                    dest="iteracoes_deteccao",
                    help="iteracoes de refinamento da deteccao com oraculo. "
                         "1 = 1-passe, sem loop; N>1 = active learning")
    ap.add_argument("--amostras-iter", type=int, default=config.AMOSTRAS_POR_ITERACAO,
                    dest="amostras_iter",
                    help="valores distintos que o oraculo rotula por iteracao "
                         "(metade previsto-sujo, metade previsto-limpo)")
    args = ap.parse_args(argv)
    faltando = [n for n in ("sujo", "limpo") if not getattr(args, n)]
    if faltando:
        ap.error(f"faltam --{' e --'.join(faltando)} (ou as variaveis "
                 f"{' e '.join(n.upper() for n in faltando)})")
    return args


def _aplicar_config(args) -> None:
    # --modelo ausente e' None: preserva MODELO_LLM e a sobreposicao por papel.
    if args.modelo is not None:
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
    except dados.DadosInvalidos as exc:  # erro de dados e' do usuario: mensagem curta, sem traceback
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    _imprimir(limpador, medida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
