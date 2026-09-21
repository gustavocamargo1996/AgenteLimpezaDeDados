"""Limpador autonomo gerado pela POC de limpeza de dados.

Origem: limpador_environment_300_2026-09-21_2021.py | dataset: environment_300
Colunas: City, State, Country, Climate_Zone, Monitoring_Station_ID, PM2.5, Temperature, Measurement_Date, Project_Name, Eco_Status

AVISO -- codigo gerado por LLM. As funcoes de deteccao/correcao foram
escritas por um LLM e passaram por sandbox (portao AST + allowlist) NA
GERACAO; este arquivo final e' Python legivel que roda sem sandbox.
REVISE antes de usar em producao.

As regras generalizam a FORMA do erro (o mesmo padrao/marcador visto na
amostra), NAO qualquer erro na mesma estrutura de tabela. Um erro de
forma diferente passa batido.

CO-OCORRENCIA codigo+FD nestas colunas (ambas as camadas se aplicam,
codigo depois FD): Climate_Zone.
"""
import re  # noqa: F401 -- disponivel para o codigo gerado

import pandas as pd


# --- detectores por coluna (codigo de deteccao gerado, renomeado) ---
def _detectar_City(col):
    return col.isin([])


def _detectar_State(col):
    return col.astype(str).str.contains(r'(?i)^[A-Z]{2}$', regex=True, na=False)


def _detectar_Country(col):
    return col.isin([])


def _detectar_Climate_Zone(col):
    return col.str.contains('Tropical|Temperate', regex=True, na=False)


def _detectar_Monitoring_Station_ID(col):
    return col.astype(str).str.contains(r'^[A-Za-z]+[0-9]{3}$', regex=True, na=False)


def _detectar_PM2_5(col):
    return col.isin([])


def _detectar_Temperature(col):
    return col.isin([])


def _detectar_Measurement_Date(col):
    return col.astype(str).str.contains('\d{1,2}/\d{1,2}/\d{4}', regex=True, na=False)


def _detectar_Project_Name(col):
    return col.astype(str).str.contains('(?i)Initiatixe|Inxtiative|Initxative|Initiativxx|Ixitiatixxx', regex=True, na=False)


def _detectar_Eco_Status(col):
    return col.isin([])



# --- corretores por coluna (codigo de correcao gerado, renomeado) ---
def _corrigir_City(valor):
    """Regra 'nenhuma': nada a corrigir localmente. Devolve o valor intacto."""
    return valor


def _corrigir_Country(valor):
    """Regra 'nenhuma': nada a corrigir localmente. Devolve o valor intacto."""
    return valor


def _corrigir_Climate_Zone(valor):
    """Regra 'nenhuma': nada a corrigir localmente. Devolve o valor intacto."""
    return valor


def _corrigir_Monitoring_Station_ID(valor):
    if re.match(r'[A-Za-z]+\d{3}', valor):
        return re.sub(r'([A-Za-z]+)(\d{3})', r'\1_\2', valor)
    return valor


def _corrigir_PM2_5(valor):
    """Regra 'nenhuma': nada a corrigir localmente. Devolve o valor intacto."""
    return valor



_DETECTORES = {
    'City': _detectar_City,
    'State': _detectar_State,
    'Country': _detectar_Country,
    'Climate_Zone': _detectar_Climate_Zone,
    'Monitoring_Station_ID': _detectar_Monitoring_Station_ID,
    'PM2.5': _detectar_PM2_5,
    'Temperature': _detectar_Temperature,
    'Measurement_Date': _detectar_Measurement_Date,
    'Project_Name': _detectar_Project_Name,
    'Eco_Status': _detectar_Eco_Status,
}

_CORRETORES = {
    'City': _corrigir_City,
    'Country': _corrigir_Country,
    'Climate_Zone': _corrigir_Climate_Zone,
    'Monitoring_Station_ID': _corrigir_Monitoring_Station_ID,
    'PM2.5': _corrigir_PM2_5,
}

DEPENDENCIAS = {
    'Country': ('City', 'Country'),
    'Climate_Zone': ('City', 'Climate_Zone'),
}

FDS = {
    'Climate_Zone': ('City', 'Climate_Zone'),
}

COLUNAS = ['City', 'State', 'Country', 'Climate_Zone', 'Monitoring_Station_ID', 'PM2.5', 'Temperature', 'Measurement_Date', 'Project_Name', 'Eco_Status']

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
