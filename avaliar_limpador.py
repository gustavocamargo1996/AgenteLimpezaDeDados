"""Mede a QUALIDADE de um limpador gerado: aplica-o ao dirty e compara com clean.

F1 de REPARO (padrao de data-cleaning, alinhado ao calc_p_r_f do ZeroDC). Por celula:
  erro    = dirty != clean           (erro real que existe)
  mudanca = corrigido != dirty       (o que o limpador alterou)
  TP      = mudanca E corrigido==clean   (alterou E ficou certo)
  Precisao = TP / |mudanca|   (das alteracoes, quantas ficaram certas; None se 0 mudancas)
  Recall   = TP / |erro|      (dos erros, quantos foram consertados; None se 0 erros)
  F1       = 2*P*R/(P+R)

Degenerado (coluna sem erro real) -> mensuravel:false, nao 0.0 silencioso (mesmo padrao
do resto da POC). Reporta tambem celulas FLAGADAS (detectadas e nao corrigidas).

Uso:
    python avaliar_limpador.py --limpador <path.py> --dataset beers [--sufixo 300]
"""
import argparse
import importlib.util
import warnings
from pathlib import Path

import pandas as pd

from poc import config

LER = dict(dtype=str, keep_default_na=False, na_values=[])


def carregar_limpador(caminho: str):
    spec = importlib.util.spec_from_file_location("limpador_gerado", caminho)
    modulo = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # SyntaxWarning de regex nao-raw do LLM
        spec.loader.exec_module(modulo)
    return modulo


def caminhos(dataset: str, sufixo: str | None):
    base = config.ZERODC_DIR / "datasets" / dataset
    suf = f"_{sufixo}" if sufixo else ""
    return base / f"{dataset}_dirty{suf}.csv", base / f"{dataset}_clean{suf}.csv"


def f1_reparo(sujo: pd.Series, corrigido: pd.Series, limpo: pd.Series) -> dict:
    erro = (sujo != limpo)
    mudanca = (corrigido != sujo)
    tp = int((mudanca & (corrigido == limpo)).sum())
    n_erro = int(erro.sum())
    n_mud = int(mudanca.sum())
    if n_erro == 0:
        return {"mensuravel": False, "erros": 0, "mudancas": n_mud, "tp": tp,
                "precisao": None, "recall": None, "f1": None,
                "observacao": "sem erro real (dirty==clean) nesta coluna"}
    p = tp / n_mud if n_mud else None
    r = tp / n_erro
    f1 = 2 * p * r / (p + r) if (p and (p + r)) else 0.0
    return {"mensuravel": True, "erros": n_erro, "mudancas": n_mud, "tp": tp,
            "precisao": round(p, 4) if p is not None else None,
            "recall": round(r, 4), "f1": round(f1, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description="F1 de reparo de um limpador gerado")
    ap.add_argument("--limpador", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sufixo", default=None)
    args = ap.parse_args()

    d_sujo, d_limpo = caminhos(args.dataset, args.sufixo)
    sujo = pd.read_csv(d_sujo, **LER)
    limpo = pd.read_csv(d_limpo, **LER)
    limp = carregar_limpador(args.limpador)

    corrigido, flags = limp.aplicar(sujo)

    print(f"limpador: {Path(args.limpador).name}")
    print(f"dataset:  {args.dataset}{'  sufixo=' + args.sufixo if args.sufixo else ''} "
          f"({len(sujo)} linhas)\n")
    print(f"{'coluna':16s}| {'erros':>6s} | {'mudou':>6s} | {'TP':>5s} | "
          f"{'precisao':>8s} | {'recall':>6s} | {'F1':>6s} | flags")
    print("-" * 82)

    total_tp = total_erro = total_mud = total_flag = 0
    for col in sujo.columns:
        if col not in corrigido.columns:
            continue
        m = f1_reparo(sujo[col], corrigido[col], limpo[col])
        nflag = int(flags[col].sum()) if col in flags.columns else 0
        total_flag += nflag
        if m["mensuravel"]:
            total_tp += m["tp"]; total_erro += m["erros"]; total_mud += m["mudancas"]
            print(f"{col:16s}| {m['erros']:6d} | {m['mudancas']:6d} | {m['tp']:5d} | "
                  f"{_pct(m['precisao']):>8s} | {_pct(m['recall']):>6s} | "
                  f"{_pct(m['f1']):>6s} | {nflag}")
        else:
            print(f"{col:16s}|      0 | {m['mudancas']:6d} |     - |      n/d |    n/d |"
                  f"    n/d | {nflag}   (sem erro real)")

    P = total_tp / total_mud if total_mud else None
    R = total_tp / total_erro if total_erro else 0.0
    F1 = 2 * P * R / (P + R) if (P and (P + R)) else 0.0
    print("-" * 82)
    print(f"{'TOTAL':16s}| {total_erro:6d} | {total_mud:6d} | {total_tp:5d} | "
          f"{_pct(round(P,4) if P else None):>8s} | {_pct(round(R,4)):>6s} | "
          f"{_pct(round(F1,4)):>6s} | {total_flag}")
    return 0


def _pct(v) -> str:
    return "n/d" if v is None else f"{v:.1%}"


if __name__ == "__main__":
    raise SystemExit(main())
