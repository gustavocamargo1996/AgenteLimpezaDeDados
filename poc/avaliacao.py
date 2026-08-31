"""Metricas do caminho e2e: deteccao (P/R/F1) e correcao (acerto/dano).

Em ambas: acerto = das celulas ERRADAS, quantas consertou. dano = das que JA
ESTAVAM CERTAS, quantas estragou. Sem `dano` a POC mente, porque numa coluna
100% corrompida qualquer regra agressiva marca 100% de acerto. As duas
funcoes excluem do calculo o orcamento rotulado (holdout) que o agente viu.
"""
import pandas as pd


def metricas_deteccao(
    mascara_col: pd.Series,
    sujo: pd.Series,
    limpo: pd.Series,
    holdout=None,
) -> dict:
    """P/R/F1 de UMA coluna: mascara detectada vs verdade `sujo != limpo`.

    Verdade de deteccao = `dirty != clean` (divergencia declarada do
    `*_error_detection.csv` do repo, por transparencia). Exclui o holdout do
    orcamento rotulado.

    Guarda de DEGENERACAO (plano secao 6.7): quando nao ha erro real no conjunto
    medido (tp+fn==0) a metrica nao e' definivel -- o recall seria 0/0. Nesse
    caso devolve `mensuravel: False` e P/R/F1 = None (n/d), NUNCA 0.0 silencioso.
    Isso separa "coluna sem erro a medir" de "detector que errou tudo" (este tem
    tp+fn>0 e recall=0 legitimo). Reusa o mesmo padrao "NAO MENSURAVEL" da
    correcao.
    """
    indices = sujo.index if holdout is None else sujo.index.difference(pd.Index(list(holdout)))
    verdade = (sujo.loc[indices] != limpo.loc[indices])
    predito = (mascara_col.loc[indices] == 1)

    tp = int((predito & verdade).sum())
    fp = int((predito & ~verdade).sum())
    fn = int((~predito & verdade).sum())
    tn = int((~predito & ~verdade).sum())

    base = {
        "celulas_avaliadas": int(len(indices)),
        "verdadeiros_positivos": tp,
        "falsos_positivos": fp,
        "falsos_negativos": fn,
        "verdadeiros_negativos": tn,
    }

    if (tp + fn) == 0:
        base.update({
            "precisao": None,
            "recall": None,
            "f1": None,
            "mensuravel": False,
            "observacao": (
                "NAO MENSURAVEL: nao ha erro real (dirty != clean) no conjunto "
                "medido, entao P/R/F1 nao sao definiveis (o recall seria 0/0). "
                f"Falsos positivos contabilizados: {fp}."
            ),
        })
        return base

    precisao = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) else 0.0

    base.update({
        "precisao": round(precisao, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mensuravel": True,
    })
    return base


def metricas_correcao(
    corrigido_col: pd.Series,
    sujo: pd.Series,
    limpo: pd.Series,
    holdout=None,
) -> dict:
    """Acerto/dano de UMA coluna apos a cascata, contra `clean`.

    acerto = das celulas ERRADAS (sujo != limpo), quantas ficaram == limpo.
    dano   = das celulas JA CERTAS (sujo == limpo), quantas sairam de limpo.
    Exclui as celulas do orcamento rotulado (holdout) -- senao o numero infla.
    """
    indices = sujo.index if holdout is None else sujo.index.difference(pd.Index(list(holdout)))
    sub_sujo = sujo.loc[indices]
    sub_limpo = limpo.loc[indices]
    sub_final = corrigido_col.loc[indices]

    erradas = corretas = acertos = dano = 0
    for bruto, esperado, final in zip(sub_sujo, sub_limpo, sub_final):
        if bruto != esperado:
            erradas += 1
            if final == esperado:
                acertos += 1
        else:
            corretas += 1
            if final != esperado:
                dano += 1

    return {
        "celulas_avaliadas": int(len(indices)),
        "erradas": erradas,
        "corretas": corretas,
        "acertos": acertos,
        "dano": dano,
        "taxa_acerto": round(acertos / erradas, 4) if erradas else None,
        "taxa_dano": round(dano / corretas, 4) if corretas else None,
    }
