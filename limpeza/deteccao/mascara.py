"""Aplicacao de `detectar(col)` por coluna, gerando a mascara 0/1 do df."""
import pandas as pd


def aplicar_detectores(
    df: pd.DataFrame, funcoes_por_coluna: dict, log: list | None = None
) -> pd.DataFrame:
    """Aplica `detectar(col)` por coluna; se a funcao falhar, a coluna inteira fica 0 e entra no log."""
    if log is None:
        log = []
    mascara = pd.DataFrame(0, index=df.index, columns=df.columns, dtype=int)
    n = len(df)
    for coluna, funcao in funcoes_por_coluna.items():
        if funcao is None or coluna not in df.columns:
            continue
        try:
            res = funcao(df[coluna])
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
            marca = res.fillna(False).astype(bool).to_numpy()
        except Exception as exc:  # noqa: BLE001 -- chamada OU coercao: deteccao nao pode derrubar o pipeline
            log.append(
                {"coluna": coluna,
                 "motivo": f"excecao {type(exc).__name__}: {exc}"}
            )
            continue
        mascara[coluna] = marca.astype(int)
    return mascara


def construir_mascara(trabalhos: list, tabela) -> pd.DataFrame:
    log: list = []
    mascara = aplicar_detectores(
        tabela.sujo, {t.coluna.nome: t.detector.funcao for t in trabalhos}, log
    )
    for entrada in log:
        print(f"  {entrada['coluna']}: deteccao descartada ({entrada['motivo']})",
              flush=True)
    return mascara
