"""Empacota as regras geradas num limpador.py AUTONOMO e reutilizavel.

O limpador NAO e' "inlinar funcoes" -- e' uma COPIA CONGELADA da logica de
aplicacao da cascata (camadas 1 e 2: codigo e FD) mais as regras geradas. Ele
reproduz a semantica exata do run que o gerou:

  - a mascara e' construida sobre o df ORIGINAL, por coluna, POSICIONALMENTE
    (espelho de deteccao.construir_mascara);
  - a correcao aplica primeiro o corretor de CODIGO e depois a FD (duplo-filtro
    deteccao==0 nas duas colunas), na ordem da cascata, lendo SEMPRE o df de
    entrada intacto e escrevendo numa copia;
  - celula marcada que nenhuma camada alterou vira FLAG.

O .py gerado nao importa a POC nem usa o sandbox em runtime: e' Python legivel
(re + pandas) que o usuario le e roda. O codigo de deteccao/correcao ja passou
pelo portao AST+allowlist do sandbox NA GERACAO; o cabecalho do limpador declara
isso e pede revisao antes de producao.

Contrato de `gerar_limpador`:
  detectores_codigo: {coluna: codigo_str de detectar(col)} -- uma entrada por
    coluna processada (mesmo as que nao acharam erro: cai em DETECTA_NADA).
  plano_correcao: {coluna: [passo, ...]} onde cada passo e' um dict:
    {"tipo": "codigo", "codigo": <corrigir(valor) str>} e/ou
    {"tipo": "fd", "determinante": <col>, "dependente": <col>}.
    Codigo e FD PODEM COEXISTIR numa coluna (na ordem: codigo depois FD).
  colunas: ordem de processamento (as chaves de dominio ficam SO' aqui, como
    dados -- nenhuma logica hardcoda nome de coluna).
"""
import re
from pathlib import Path

from . import config

# Idioma Series alinhado a `col`, tudo-False -- espelha deteccao.DETECTA_NADA.
# Usado quando uma coluna nao trouxe codigo de deteccao (defensivo).
_DETECTA_NADA = "def detectar(col):\n    return col.isin([])\n"


def _sufixos_seguros(colunas: list[str]) -> dict:
    """Mapeia cada coluna a um sufixo de identificador Python UNICO.

    O nome de coluna vira nome de funcao (`_detectar_<sufixo>`), entao precisa
    ser identificador valido: nao-alnum vira '_', e um prefixo entra se comeca
    por digito ou fica vazio. Colisao (duas colunas -> mesmo sufixo) e' desfeita
    com um contador. O nome de coluna REAL segue sendo a chave dos dicts do
    limpador -- este sufixo e' so' cosmetico para o nome da funcao.
    """
    usados: set = set()
    mapa: dict = {}
    for i, col in enumerate(colunas):
        base = re.sub(r"\W", "_", str(col))
        if not base or base[0].isdigit():
            base = f"col_{base}" if base else f"col_{i}"
        cand = base
        j = 1
        while cand in usados:
            cand = f"{base}_{j}"
            j += 1
        usados.add(cand)
        mapa[col] = cand
    return mapa


def _renomear(codigo: str, de: str, para: str) -> str:
    """Renomeia a UNICA def `de` para `para`, preservando o corpo intacto.

    Regex tolerante a espacos (`def  detectar (`). O sandbox ja garantiu que ha
    exatamente uma funcao com esse nome, entao count=1 e' suficiente.
    """
    texto = re.sub(rf"\bdef\s+{re.escape(de)}\s*\(", f"def {para}(", codigo.strip(), count=1)
    return texto.rstrip() + "\n"


def _passos_de(plano_correcao: dict, coluna: str) -> tuple:
    """Extrai (codigo_correcao_str | None, (det, dep) | None) do plano da coluna."""
    corretor = None
    fd = None
    for passo in plano_correcao.get(coluna, []) or []:
        if passo.get("tipo") == "codigo" and passo.get("codigo"):
            corretor = passo["codigo"]
        elif passo.get("tipo") == "fd":
            fd = (passo["determinante"], passo["dependente"])
    return corretor, fd


# --------------------------------------------------------------------------- #
# Corpo ESTATICO do limpador: mascara + FD (duplo-filtro) + aplicar + __main__.
# Copia congelada da cascata (camadas 1 e 2). NAO simplificar a coercao da
# mascara nem o duplo-filtro: sao o que garante a reproducao do run.
# --------------------------------------------------------------------------- #
_ESTATICO = '''
def _mascara(df):
    """Espelho de deteccao.construir_mascara: mascara BOOL por coluna, posicional.

    Inicializa tudo-False no shape do df. Por coluna COM detector, chama
    `_detectar_<col>(df[col])` UMA vez e coage `res.fillna(False).astype(bool)`,
    atribuindo por POSICAO via `.to_numpy()` (nunca boolean-index por indice). Se
    o detector lanca, nao devolve Series ou devolve tamanho errado, a coluna
    INTEIRA fica nao-marcada. Coluna SEM detector (ex.: o determinante da FD) ->
    tudo-False.
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
    return mascara


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
'''


def _montar_cabecalho(origem: str, dataset: str, colunas: list[str],
                      coocorrencia: list[str]) -> str:
    """Docstring de cabecalho do limpador (ASCII-only)."""
    linhas = [
        '"""Limpador autonomo gerado pela POC de limpeza de dados.',
        "",
        f"Origem: {origem} | dataset: {dataset}",
        f"Colunas: {', '.join(str(c) for c in colunas)}",
        "",
        "AVISO -- codigo gerado por LLM. As funcoes de deteccao/correcao foram",
        "escritas por um LLM e passaram por sandbox (portao AST + allowlist) NA",
        "GERACAO; este arquivo final e' Python legivel que roda sem sandbox.",
        "REVISE antes de usar em producao.",
        "",
        "As regras generalizam a FORMA do erro (o mesmo padrao/marcador visto na",
        "amostra), NAO qualquer erro na mesma estrutura de tabela. Um erro de",
        "forma diferente passa batido.",
    ]
    if coocorrencia:
        linhas += [
            "",
            "CO-OCORRENCIA codigo+FD nestas colunas (ambas as camadas se aplicam,",
            f"codigo depois FD): {', '.join(coocorrencia)}.",
        ]
    linhas += ['"""']
    return "\n".join(linhas)


def escrever_limpador(
    caminho,
    dataset: str,
    detectores_codigo: dict,
    plano_correcao: dict,
    colunas: list,
) -> Path:
    """Escreve um limpador .py autonomo e devolve o Path.

    Emite: cabecalho + imports; `_detectar_<col>` por coluna; `_corrigir_<col>`
    por coluna com corretor; os dicts `_DETECTORES`/`_CORRETORES`/`FDS`/`COLUNAS`
    (as chaves sao os nomes de coluna reais, unico lugar onde eles aparecem);
    e o corpo estatico (`_mascara`, `_aplicar_fd`, `aplicar`, `__main__`).
    """
    caminho = Path(caminho)
    colunas = list(colunas)
    sufixos = _sufixos_seguros(colunas)

    partes: list = []
    detectores_por_col: dict = {}
    for col in colunas:
        codigo = (detectores_codigo.get(col) or "").strip() or _DETECTA_NADA
        nome = f"_detectar_{sufixos[col]}"
        partes.append(_renomear(codigo, "detectar", nome))
        detectores_por_col[col] = nome

    corretores_por_col: dict = {}
    fds_por_col: dict = {}
    coocorrencia: list = []
    for col in colunas:
        corretor, fd = _passos_de(plano_correcao, col)
        if corretor:
            nome = f"_corrigir_{sufixos[col]}"
            partes.append(_renomear(corretor, "corrigir", nome))
            corretores_por_col[col] = nome
        if fd is not None:
            fds_por_col[col] = fd
        if corretor and fd is not None:
            coocorrencia.append(str(col))

    # Dicts que ligam nome-de-coluna REAL -> funcao. As chaves (repr) sao o unico
    # ponto onde nome de coluna de dominio aparece; nenhuma logica os hardcoda.
    def _dict_funcs(mapa: dict) -> str:
        if not mapa:
            return "{}"
        itens = ",\n".join(f"    {col!r}: {nome}" for col, nome in mapa.items())
        return "{\n" + itens + ",\n}"

    def _dict_fds(mapa: dict) -> str:
        if not mapa:
            return "{}"
        itens = ",\n".join(
            f"    {col!r}: ({det!r}, {dep!r})" for col, (det, dep) in mapa.items()
        )
        return "{\n" + itens + ",\n}"

    cabecalho = _montar_cabecalho(caminho.name, dataset, colunas, coocorrencia)

    blocos = [
        cabecalho,
        "import re  # noqa: F401 -- disponivel para o codigo gerado",
        "",
        "import pandas as pd",
        "",
        "",
        "# --- detectores por coluna (codigo de deteccao gerado, renomeado) ---",
        "\n\n".join(partes[: len(colunas)]) if colunas else "",
    ]

    # corretores (as partes apos os detectores)
    corretor_partes = partes[len(colunas):]
    if corretor_partes:
        blocos += [
            "",
            "",
            "# --- corretores por coluna (codigo de correcao gerado, renomeado) ---",
            "\n\n".join(corretor_partes),
        ]

    blocos += [
        "",
        "",
        f"_DETECTORES = {_dict_funcs(detectores_por_col)}",
        "",
        f"_CORRETORES = {_dict_funcs(corretores_por_col)}",
        "",
        f"FDS = {_dict_fds(fds_por_col)}",
        "",
        f"COLUNAS = {colunas!r}",
        _ESTATICO,
    ]

    texto = "\n".join(blocos).rstrip() + "\n"
    caminho.write_text(texto, encoding="utf-8")
    return caminho


def gerar_limpador(trabalhos: list, saida: Path) -> Path:
    """Empacota os detectores e os planos de correcao dos trabalhos num limpador .py."""
    return escrever_limpador(
        caminho=saida,
        dataset=config.DATASET,
        detectores_codigo={t.coluna.nome: t.detector.codigo for t in trabalhos},
        plano_correcao={t.coluna.nome: (t.correcao.passos if t.correcao else [])
                        for t in trabalhos},
        colunas=[t.coluna.nome for t in trabalhos],
    )
