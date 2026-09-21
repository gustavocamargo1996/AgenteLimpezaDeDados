"""Limpador autonomo gerado pela POC de limpeza de dados.

Origem: limpador_beers_2026-08-20_0927.py | dataset: beers
Colunas: id, beer-name, style, ounces, abv, ibu, brewery_id, brewery-name, city, state

AVISO -- codigo gerado por LLM. As funcoes de deteccao/correcao foram
escritas por um LLM e passaram por sandbox (portao AST + allowlist) NA
GERACAO; este arquivo final e' Python legivel que roda sem sandbox.
REVISE antes de usar em producao.

As regras generalizam a FORMA do erro (o mesmo padrao/marcador visto na
amostra), NAO qualquer erro na mesma estrutura de tabela. Um erro de
forma diferente passa batido.

CO-OCORRENCIA codigo+FD nestas colunas (ambas as camadas se aplicam,
codigo depois FD): city.
"""
import re  # noqa: F401 -- disponivel para o codigo gerado

import pandas as pd


# --- detectores por coluna (codigo de deteccao gerado, renomeado) ---
def _detectar_id(col):
    return col.isin([])


def _detectar_beer_name(col):
    return col.isin([])


def _detectar_style(col):
    return col.isin([])


def _detectar_ounces(col):
    return col.astype(str).str.contains('(?i)\b(ounce|oz)\b', regex=True, na=False) | col.astype(str).str.contains('(?i)\bSilo Can\b', regex=True, na=False) | col.astype(str).str.contains('(?i)\bAlumi-Tek\b', regex=True, na=False) | col.astype(str).str.contains('(?i)\b\d+(\.\d+)?\s*(ounce|oz)\b', regex=True, na=False)


def _detectar_abv(col):
    return col.astype(str).str.contains('%', regex=False, na=False)


def _detectar_ibu(col):
    return col.eq('N/A')


def _detectar_brewery_id(col):
    return col.isin([])


def _detectar_brewery_name(col):
    return col.isin([])


def _detectar_city(col):
    return col.str.contains(' TX', na=False)


def _detectar_state(col):
    return col.astype(str).eq('') | col.astype(str).eq('N/A')



# --- corretores por coluna (codigo de correcao gerado, renomeado) ---
def _corrigir_id(valor):
    """Regra 'nenhuma': nada a corrigir localmente. Devolve o valor intacto."""
    return valor


def _corrigir_beer_name(valor):
    """Regra 'nenhuma': nada a corrigir localmente. Devolve o valor intacto."""
    return valor


def _corrigir_style(valor):
    """Regra 'nenhuma': nada a corrigir localmente. Devolve o valor intacto."""
    return valor


def _corrigir_ibu(valor):
    if re.match(r'^(N/A|)$', valor):
        return ''
    return valor


def _corrigir_brewery_id(valor):
    """Regra 'nenhuma': nada a corrigir localmente. Devolve o valor intacto."""
    return valor


def _corrigir_brewery_name(valor):
    """Regra 'nenhuma': nada a corrigir localmente. Devolve o valor intacto."""
    return valor


def _corrigir_city(valor):
    """Regra 'nenhuma': nada a corrigir localmente. Devolve o valor intacto."""
    return valor



_DETECTORES = {
    'id': _detectar_id,
    'beer-name': _detectar_beer_name,
    'style': _detectar_style,
    'ounces': _detectar_ounces,
    'abv': _detectar_abv,
    'ibu': _detectar_ibu,
    'brewery_id': _detectar_brewery_id,
    'brewery-name': _detectar_brewery_name,
    'city': _detectar_city,
    'state': _detectar_state,
}

_CORRETORES = {
    'id': _corrigir_id,
    'beer-name': _corrigir_beer_name,
    'style': _corrigir_style,
    'ibu': _corrigir_ibu,
    'brewery_id': _corrigir_brewery_id,
    'brewery-name': _corrigir_brewery_name,
    'city': _corrigir_city,
}

DEPENDENCIAS = {}

FDS = {
    'city': ('brewery-name', 'city'),
    'state': ('brewery_id', 'state'),
}

COLUNAS = ['id', 'beer-name', 'style', 'ounces', 'abv', 'ibu', 'brewery_id', 'brewery-name', 'city', 'state']

def _mascara(df):
    """Espelho de deteccao.construir_mascara + detectar_dependencia: mascara BOOL.

    1. Intra-coluna: inicializa tudo-False no shape do df. Por coluna COM
       detector, chama `_detectar_<col>(df[col])` UMA vez e coage
       `res.fillna(False).astype(bool)`, atribuindo por POSICAO via `.to_numpy()`
       (nunca boolean-index por indice). Se o detector lanca, nao devolve Series
       ou devolve tamanho errado, a coluna INTEIRA fica nao-marcada.
    2. Dependencia funcional: por coluna em DEPENDENCIAS, soma (OU) as celulas que
       desviam da moda condicionada. Toda FD le a mascara INTRA do passo 1, nunca
       a que esta sendo acumulada -- a mesma ordem da POC, que evita circularidade.
    """
    mascara = pd.DataFrame(False, index=df.index, columns=df.columns)
    n = len(df)
    for coluna, detector in _DETECTORES.items():
        if detector is None or coluna not in df.columns:
            continue
        try:
            res = detector(df[coluna])
        except Exception:
            continue
        if not isinstance(res, pd.Series) or len(res) != n:
            continue
        try:
            marca = res.fillna(False).astype(bool).to_numpy()
        except Exception:
            continue
        mascara[coluna] = marca
    intra = mascara.copy()
    for coluna, (det, dep) in DEPENDENCIAS.items():
        if det not in df.columns or dep not in df.columns:
            continue
        mascara[coluna] = mascara[coluna] | _desvios_da_moda(df, intra, det, dep)
    return mascara


def _desvios_da_moda(df, mascara, det, dep):
    """Espelho de deteccao/dependencia.py: True onde `dep` difere da moda do grupo.

    A moda de `dep` em cada grupo `df[det] == valor` usa so' as linhas com
    `mascara[det]` e `mascara[dep]` nao-marcados (DUPLO FILTRO, igual a
    fd.moda_condicionada). Empate: `mode().iloc[0]`, o menor valor. Grupo sem
    linha compativel nao marca ninguem.
    """
    marca = pd.Series(False, index=df.index)
    for valor_det in df[det].unique():
        no_grupo = df[det] == valor_det
        condicao = (
            no_grupo
            & (mascara[det] == False)  # noqa: E712 -- espelha mascara[det]==0 da POC
            & (mascara[dep] == False)  # noqa: E712
        )
        valores = df.loc[condicao, dep]
        if valores.empty:
            continue
        marca |= no_grupo & (df[dep] != valores.mode().iloc[0])
    return marca


def _aplicar_fd(df, mascara, col, det, dep):
    """Espelho de fd.aplicar_fd/_moda_condicionada: moda com DUPLO-FILTRO.

    Para cada celula MARCADA em `col`, a correcao e' a moda de `dep` entre as
    linhas com `df[det] == valor_det` E `mascara[det]` nao-marcado E
    `mascara[dep]` nao-marcado, computada no df ORIGINAL. A POC usa mascara int
    0/1 e filtra `== 0`; aqui a mascara e' bool e o filtro e' `== False` --
    equivalente (`False == 0`). `det` sem detector -> `mascara[det]` tudo-False
    (nao filtra por det errado). Sem linha compativel -> mantem (vira flag).
    Devolve {indice: novo_valor} SO' das celulas cujo valor mudou.
    """
    if det not in df.columns or dep not in df.columns:
        return {}
    mudancas = {}
    marcados = list(df.index[mascara[col].to_numpy()])
    for idx in marcados:
        valor_det = df.at[idx, det]
        condicao = (
            (df[det] == valor_det)
            & (mascara[det] == False)  # noqa: E712 -- espelha mascara[det]==0 da POC
            & (mascara[dep] == False)  # noqa: E712
        )
        valores = df.loc[condicao, dep]
        if valores.empty:
            continue
        corrigido_valor = valores.mode().iloc[0]
        if corrigido_valor != df.at[idx, dep]:
            mudancas[idx] = corrigido_valor
    return mudancas


def aplicar(df):
    """Reproduz a limpeza do run: (df_corrigido, df_flags).

    1. `m = _mascara(df)` sobre o df de entrada INTACTO (todas as mascaras ANTES
       de corrigir); 2. `corrigido = df.copy()`, `flags` tudo-False no shape do
       df; 3. por coluna, na ORDEM do plano: aplica o corretor de codigo lendo o
       VALOR ORIGINAL de `df` nas celulas marcadas (igual a camada 1, que le
       `df.at[idx, col]`, nunca a copia parcial), depois a FD lendo o df ORIGINAL
       e escrevendo em `corrigido`; celula marcada e nao-alterada por nenhuma
       camada -> `flags[col] = True`; 4. devolve `(corrigido, flags)`.
    """
    m = _mascara(df)
    corrigido = df.copy()
    flags = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in COLUNAS:
        if col not in df.columns:
            continue
        marcados = list(df.index[m[col].to_numpy()])
        corretor = _CORRETORES.get(col)
        pendentes = []
        for idx in marcados:
            if corretor is not None:
                antigo = df.at[idx, col]
                try:
                    novo = corretor(antigo)
                    novo = "" if novo is None else str(novo)
                except Exception:
                    novo = antigo
                if novo != antigo:
                    corrigido.at[idx, col] = novo
                    continue
            pendentes.append(idx)  # intacta escala para a FD
        fd = FDS.get(col)
        if fd is not None and pendentes:
            det, dep = fd
            mudancas = _aplicar_fd(df, m, col, det, dep)
            restantes = []
            for idx in pendentes:
                if idx in mudancas:
                    corrigido.at[idx, col] = mudancas[idx]
                else:
                    restantes.append(idx)
            pendentes = restantes
        for idx in pendentes:
            flags.at[idx, col] = True
    return corrigido, flags


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("uso: python <este_limpador>.py <entrada.csv>")
        raise SystemExit(2)
    entrada = sys.argv[1]
    # Mesma carga da POC (dados.carregar): texto literal, sem NA magico -- senao
    # 'N/A'/'null' virariam NaN e a reproducao divergiria.
    df_in = pd.read_csv(entrada, dtype=str, keep_default_na=False, na_values=[])
    df_corrigido, df_flags = aplicar(df_in)
    base = entrada[:-4] if entrada.lower().endswith(".csv") else entrada
    df_corrigido.to_csv(base + "_corrigido.csv", index=False)
    df_flags.to_csv(base + "_flags.csv", index=False)
    print("corrigido:", base + "_corrigido.csv")
    print("flags:", base + "_flags.csv")
