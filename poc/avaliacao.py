"""Duas escalas de holdout, porque uma so' nao consegue medir tudo.

GENERALIZACAO (holdout por VALOR) -- so' as linhas cujo valor sujo o agente
nunca viu. Responde: a regra vale para formas que nao estavam na amostra?
E' a medida honesta, mas degenera: quando uma classe de erro inteira e' um
unico valor (ibu='N/A' em 1005 celulas, state='' em 127), mostrar esse valor
remove a classe toda e nao sobra nada para medir.

COBERTURA (holdout por LINHA) -- a coluna inteira menos as ~12 linhas
efetivamente exibidas. Responde: a regra, uma vez formulada, se aplica
corretamente ao resto da coluna? Nao prova generalizacao -- o valor pode ser o
mesmo que o agente viu -- mas prova que a regra foi bem formulada, bem
traduzida e nao quebra no volume. Sempre mensuravel.

As duas juntas separam tres falhas que um numero so' confunde: regra que nao
generaliza, regra mal traduzida, e regra que nao pode ser avaliada.

Em ambas: acerto = das celulas ERRADAS, quantas consertou. dano = das que JA
ESTAVAM CERTAS, quantas estragou. Sem `dano` a POC mente, porque numa coluna
100% corrompida qualquer regra agressiva marca 100% de acerto.
"""
from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass
class Medida:
    escopo: str
    celulas: int
    erradas: int
    corretas: int
    acertos: int
    dano: int
    taxa_acerto: float | None
    taxa_dano: float | None
    mensuravel: bool
    observacao: str = ""


@dataclass
class Resultado:
    coluna: str
    modo: str
    valores_mostrados: int
    linhas_mostradas: int
    erradas_na_coluna: int
    excecoes: int
    generalizacao: Medida
    cobertura: Medida
    observacao: str = ""
    _campos_planos: tuple = field(default=("coluna", "modo"), repr=False)

    def dict(self) -> dict:
        d = asdict(self)
        d.pop("_campos_planos", None)
        return d

    # atalhos usados pelo relatorio e pela impressao
    @property
    def taxa_acerto(self):
        return self.generalizacao.taxa_acerto

    @property
    def taxa_dano(self):
        return self.generalizacao.taxa_dano

    @property
    def celulas_holdout(self):
        return self.generalizacao.celulas


def _medir(escopo: str, sujo: pd.Series, limpo: pd.Series, indices: pd.Index, funcao):
    sub_sujo = sujo.loc[indices]
    sub_limpo = limpo.loc[indices]
    erradas = corretas = acertos = dano = excecoes = 0

    for bruto, esperado in zip(sub_sujo, sub_limpo):
        if funcao is None:
            saida = bruto  # traducao falhou: nenhuma regra existe, nada muda
        else:
            try:
                resultado = funcao(bruto)
                saida = "" if resultado is None else str(resultado)
            except Exception:
                excecoes += 1
                saida = bruto

        if bruto != esperado:
            erradas += 1
            if saida == esperado:
                acertos += 1
        else:
            corretas += 1
            if saida != esperado:
                dano += 1

    medida = Medida(
        escopo=escopo,
        celulas=len(indices),
        erradas=erradas,
        corretas=corretas,
        acertos=acertos,
        dano=dano,
        taxa_acerto=round(acertos / erradas, 4) if erradas else None,
        taxa_dano=round(dano / corretas, 4) if corretas else None,
        mensuravel=erradas > 0,
    )
    return medida, excecoes


def avaliar(
    coluna: str,
    modo: str,
    sujo: pd.Series,
    limpo: pd.Series,
    holdout_valor: pd.Index,
    holdout_linha: pd.Index,
    funcao,
    valores_mostrados: int,
    linhas_mostradas: int,
    erradas_na_coluna: int = 0,
) -> Resultado:
    generalizacao, exc1 = _medir("generalizacao", sujo, limpo, holdout_valor, funcao)
    cobertura, exc2 = _medir("cobertura", sujo, limpo, holdout_linha, funcao)

    if not generalizacao.mensuravel and erradas_na_coluna > 0:
        generalizacao.observacao = (
            f"NAO MENSURAVEL: as {erradas_na_coluna} celulas erradas desta coluna usam "
            "apenas valores que foram mostrados ao agente, entao nenhuma sobrou no "
            "holdout por valor. Use a cobertura para julgar esta coluna."
        )

    observacao = ""
    if funcao is None:
        observacao = "traducao falhou: nenhuma regra aplicada"

    return Resultado(
        coluna=coluna,
        modo=modo,
        valores_mostrados=valores_mostrados,
        linhas_mostradas=linhas_mostradas,
        erradas_na_coluna=erradas_na_coluna,
        excecoes=exc1 + exc2,
        generalizacao=generalizacao,
        cobertura=cobertura,
        observacao=observacao,
    )


# --------------------------------------------------------------------------- #
# Modo end-to-end (--e2e): metricas de DETECCAO e de CORRECAO.
# --------------------------------------------------------------------------- #


def _sem_holdout(indice: pd.Index, holdout) -> pd.Index:
    """Remove do indice as celulas do orcamento rotulado (holdout)."""
    if holdout is None:
        return indice
    return indice.difference(pd.Index(list(holdout)))


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
    indices = _sem_holdout(sujo.index, holdout)
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
    indices = _sem_holdout(sujo.index, holdout)
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
