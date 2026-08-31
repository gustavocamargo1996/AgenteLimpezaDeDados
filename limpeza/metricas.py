"""Etapa 6: mede deteccao (P/R/F1) e correcao (acerto/dano) contra o clean."""
import pandas as pd


def metricas_deteccao(
    mascara_col: pd.Series,
    sujo: pd.Series,
    limpo: pd.Series,
    holdout=None,
) -> dict:
    """P/R/F1 de uma coluna: mascara detectada vs verdade `sujo != limpo`, fora do holdout."""
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
        # Sem erro real no conjunto medido o recall seria 0/0: devolve None, nunca 0.0.
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
    indices = sujo.index if holdout is None else sujo.index.difference(pd.Index(list(holdout)))
    sub_sujo = sujo.loc[indices]
    sub_limpo = limpo.loc[indices]
    sub_final = corrigido_col.loc[indices]

    # Vetorizado: mesma contagem que iterar celula a celula, sem o loop.
    era_erro = sub_sujo != sub_limpo
    era_correta = ~era_erro
    erradas = int(era_erro.sum())
    corretas = int(era_correta.sum())
    acertos = int((era_erro & (sub_final == sub_limpo)).sum())
    dano = int((era_correta & (sub_final != sub_limpo)).sum())

    return {
        "celulas_avaliadas": int(len(indices)),
        "erradas": erradas,
        "corretas": corretas,
        "acertos": acertos,
        "dano": dano,
        "taxa_acerto": round(acertos / erradas, 4) if erradas else None,
        "taxa_dano": round(dano / corretas, 4) if corretas else None,
    }


def avaliar(trabalhos: list, tabela, mascara, corrigido) -> dict:
    """Mede deteccao e correcao de cada coluna fora do holdout e preenche `.medida`."""
    saida: dict = {"deteccao": {}, "correcao": {}}
    for trabalho in trabalhos:
        coluna = trabalho.coluna
        holdout = sorted(trabalho.amostra.linhas) if trabalho.amostra else None
        deteccao = metricas_deteccao(
            mascara[coluna.nome], coluna.sujo, coluna.limpo, holdout
        )
        correcao = metricas_correcao(
            corrigido[coluna.nome], coluna.sujo, coluna.limpo, holdout
        )
        trabalho.medida = {"deteccao": deteccao, "correcao": correcao}
        saida["deteccao"][coluna.nome] = deteccao
        saida["correcao"][coluna.nome] = correcao
    return saida
