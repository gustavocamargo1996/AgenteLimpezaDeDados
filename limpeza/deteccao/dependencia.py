"""Etapa 3b: marca celulas que violam uma dependencia funcional entre colunas."""
import pandas as pd

from .. import config, erros
from ..correcao import fd as fd_mod


def detectar_dependencia(trabalhos: list, tabela, mascara_intra, agente) -> pd.DataFrame:
    """Propoe uma FD por coluna e marca quem desvia da moda condicionada."""
    df = tabela.sujo
    mascara = pd.DataFrame(0, index=df.index, columns=df.columns, dtype=int)
    for trabalho in trabalhos:
        nome = trabalho.coluna.nome
        candidatos = fd_mod.candidatos_determinantes(df, nome, limiar=config.MI_THRESHOLD)
        if not candidatos:
            continue
        try:
            marca = _marcar_coluna(trabalho, df, mascara_intra, candidatos, agente)
        except Exception as exc:  # noqa: BLE001 -- 1 coluna ruim nao derruba a etapa
            print(f"  {nome}: FD falhou ({erros.descrever(exc)}) -> sem marca", flush=True)
            continue
        if marca is not None:
            mascara[nome] = marca.astype(int)
    return mascara


def _marcar_coluna(trabalho, df, mascara_intra, candidatos, agente):
    """Devolve a marca booleana da coluna, ou None quando o gate reprova."""
    nome = trabalho.coluna.nome
    rotulados = trabalho.amostra.rotulados
    fd = fd_mod.propor_fd(nome, rotulados, candidatos, df, agente=agente)
    if not fd_mod.validar_fd(fd, rotulados, df, mascara_intra, nome):
        print(f"  {nome}: FD `{fd.determinante}` reprovada no gate -> sem marca", flush=True)
        return None
    trabalho.dependencia = fd
    print(f"  {nome}: FD `{fd.determinante}` -> `{nome}` aprovada; marcando desvios...",
          flush=True)
    return _desvios_da_moda(df, mascara_intra, fd.determinante, nome)


def _desvios_da_moda(df, mascara, determinante, dependente) -> pd.Series:
    """True onde o valor difere da moda do grupo; grupo sem moda nao marca ninguem."""
    marca = pd.Series(False, index=df.index)
    for valor_det in df[determinante].unique():
        moda = fd_mod.moda_condicionada(df, mascara, determinante, dependente, valor_det)
        if moda is None:
            continue
        no_grupo = df[determinante] == valor_det
        marca |= no_grupo & (df[dependente] != moda)
    return marca
