"""Aplicacao de `detectar(col)` por coluna, gerando a mascara 0/1 do df."""
import pandas as pd


def aplicar_detectores(
    df: pd.DataFrame, funcoes_por_coluna: dict, log: list | None = None
) -> pd.DataFrame:
    """Aplica `detectar(col)` por COLUNA e devolve mascara 0/1 no shape do df.

    Contrato Series (plano secao 6/7, MANTENDO a assinatura): por coluna chama
    `funcao(df[coluna])` UMA vez e le o resultado POSICIONALMENTE. Exige uma
    pd.Series booleana do MESMO tamanho da coluna; coage o dtype com
    `res.fillna(False).astype(bool)` e atribui a coluna da mascara por POSICAO
    via `.to_numpy()` (nunca boolean-index por indice, que desalinha se a funcao
    devolver indice proprio). Politica estrita, agora com granularidade
    POR-COLUNA (declarada): se a funcao lanca, nao devolve Series ou devolve
    tamanho errado, a coluna INTEIRA fica 0 e uma entrada entra no log. NAO le
    clean -- auditavel: nenhum import de clean, nenhum parametro clean.
    """
    if log is None:
        log = []
    mascara = pd.DataFrame(0, index=df.index, columns=df.columns, dtype=int)
    n = len(df)
    for coluna, funcao in funcoes_por_coluna.items():
        if funcao is None or coluna not in df.columns:
            continue
        try:
            res = funcao(df[coluna])
        except Exception as exc:  # noqa: BLE001 -- deteccao nao pode derrubar o pipeline
            log.append(
                {"coluna": coluna,
                 "motivo": f"excecao {type(exc).__name__}: {exc}"}
            )
            continue
        if not isinstance(res, pd.Series):
            log.append(
                {"coluna": coluna,
                 "motivo": f"retorno nao-Series ({type(res).__name__})"}
            )
            continue
        if len(res) != n:
            log.append(
                {"coluna": coluna,
                 "motivo": f"tamanho divergente: esperado {n}, veio {len(res)}"}
            )
            continue
        try:
            marca = res.fillna(False).astype(bool).to_numpy()
        except Exception as exc:  # noqa: BLE001 -- coercao nao pode derrubar o pipeline
            log.append(
                {"coluna": coluna,
                 "motivo": f"coercao bool falhou ({type(exc).__name__}: {exc})"}
            )
            continue
        mascara[coluna] = marca.astype(int)
    return mascara


def construir_mascara(trabalhos: list, tabela) -> pd.DataFrame:
    """Aplica o detector de cada trabalho e devolve a mascara 0/1 da tabela suja."""
    log: list = []
    mascara = aplicar_detectores(
        tabela.sujo, {t.coluna.nome: t.detector.funcao for t in trabalhos}, log
    )
    for entrada in log:
        print(f"  {entrada['coluna']}: deteccao descartada ({entrada['motivo']})",
              flush=True)
    return mascara
