"""Portao AST + execucao restrita do codigo gerado pelo LLM.

Adaptado de detection.py:125-161 do ZeroDC, com dois endurecimentos que o
original nao tem:
  1. NAO executa em globals() -- o correction.py:790 do ZeroDC faz
     exec(code, globals()), o que permite ao codigo gerado redefinir qualquer
     nome do modulo. Aqui o namespace e' descartavel.
  2. builtins entram por allowlist explicita, em vez de virem inteiros.

MODO SERIE (series_mode, ADITIVO -- usado so' pela deteccao `detectar(col)`):
a deteccao passou a receber a COLUNA inteira (uma pd.Series) e devolver uma
pd.Series[bool]. Nesse modo o portao ganha uma ALLOWLIST DE ATRIBUTOS
(`_ATRIBUTOS_SERIE`): todo `ast.Attribute` do corpo deve ter `.attr` num conjunto
pequeno e fechado (o idioma-alvo: str.contains e comparacoes elementwise).
Qualquer atributo fora disso REJEITA -- isso bloqueia I/O (`to_csv`/`to_pickle`) e
operacoes cross-row (`duplicated`/`mode`/`groupby`/`shift`/`values`/`to_numpy`/
`apply`/`pipe`). `pd` NAO entra no namespace do exec (segue `{re, __builtins__}`),
entao `pd.read_csv`/`pd.io.common.get_handle`/`pd.eval` ficam inalcancaveis
(NameError) alem de ja serem barrados pela allowlist. O modo escalar (default,
series_mode=False) e' o caminho de `corrigir(valor)` e fica INTACTO.

HONESTIDADE SOBRE O LIMITE: isto reduz superficie, nao e' sandbox de verdade.
Codigo determinado ainda escapa de allowlist de builtins em CPython, e a allowlist
de atributos do modo serie NAO e' prova de contencao total. A fronteira REAL de
seguranca e' OPERACIONAL: a POC roda sobre dados publicos (beers), offline, e nada
mais deve ser executado aqui.
"""
import ast
import re

NOME_FUNCAO = "corrigir"
NOME_ARGUMENTO = "valor"

_CHAMADAS_PROIBIDAS = {
    "__import__", "eval", "exec", "compile", "open", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "breakpoint", "memoryview", "exit", "quit",
}

_BUILTINS_PERMITIDOS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "range": range, "round": round, "set": set, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "isinstance": isinstance, "ValueError": ValueError, "TypeError": TypeError,
    "AttributeError": AttributeError, "IndexError": IndexError,
    "KeyError": KeyError, "Exception": Exception, "None": None,
}

# ALLOWLIST de atributos do modo serie (deteccao `detectar(col) -> pd.Series`).
# Conjunto pequeno e FECHADO = o idioma-alvo (str.contains e comparacoes
# elementwise). No series_mode, todo `ast.Attribute` do corpo deve ter `.attr`
# aqui; qualquer outro atributo REJEITA. Isso barra I/O em `col` (to_csv/
# to_pickle), cross-row (duplicated/mode/groupby/shift/values/to_numpy/apply/
# pipe) e travessia de submodulo de `pd` (read_csv/io/common/get_handle/eval).
# NAO e' contencao total -- a fronteira real e' operacional (ver docstring).
_ATRIBUTOS_SERIE = {
    "str", "contains", "startswith", "endswith", "match", "fullmatch",
    "astype", "eq", "ne", "isin", "isna", "notna", "fillna", "strip",
    "lower", "upper", "len",
}


class CodigoRejeitado(Exception):
    """O codigo gerado nao passou no portao. Nunca foi executado."""


def validar(
    codigo: str,
    nome_funcao: str = NOME_FUNCAO,
    nome_argumento: str = NOME_ARGUMENTO,
    series_mode: bool = False,
) -> None:
    """Portao AST. Defaults preservam o contrato antigo (`corrigir(valor)`).

    Parametrizar `nome_funcao`/`nome_argumento` permite reusar o mesmo portao
    para a funcao de deteccao (`detectar(col)`) sem duplicar a logica.

    `series_mode=True` (ADITIVO): alem de todo o gate escalar, exige que TODO
    `ast.Attribute` do corpo tenha `.attr` em `_ATRIBUTOS_SERIE`. Qualquer
    atributo fora da allowlist REJEITA (bloqueia I/O e cross-row em `col` e
    travessia de submodulo de `pd`). Com series_mode=False (default) o portao e'
    EXATAMENTE o de antes.
    """
    try:
        arvore = ast.parse(codigo)
    except SyntaxError as erro:
        raise CodigoRejeitado(f"sintaxe invalida: {erro}") from erro

    if len(arvore.body) != 1 or not isinstance(arvore.body[0], ast.FunctionDef):
        raise CodigoRejeitado(
            "o codigo deve conter exatamente uma definicao de funcao e mais nada "
            "(sem imports, sem constantes de modulo, sem chamadas no topo)"
        )

    funcao = arvore.body[0]
    if funcao.name != nome_funcao:
        raise CodigoRejeitado(f"a funcao deve se chamar '{nome_funcao}', veio '{funcao.name}'")

    args = funcao.args
    if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
        raise CodigoRejeitado("a funcao nao pode ter *args, **kwargs nem argumentos so-nomeados")
    if len(args.args) != 1 or args.args[0].arg != nome_argumento:
        raise CodigoRejeitado(
            f"a funcao deve receber exatamente um argumento chamado '{nome_argumento}'"
        )

    for no in ast.walk(funcao):
        if isinstance(no, (ast.Import, ast.ImportFrom)):
            raise CodigoRejeitado("import proibido -- o modulo 're' ja esta disponivel")
        if isinstance(no, ast.Call) and isinstance(no.func, ast.Name):
            if no.func.id in _CHAMADAS_PROIBIDAS:
                raise CodigoRejeitado(f"chamada proibida: {no.func.id}()")
        if isinstance(no, ast.Attribute) and no.attr.startswith("__"):
            raise CodigoRejeitado(f"acesso a atributo dunder proibido: {no.attr}")
        if series_mode and isinstance(no, ast.Attribute):
            if no.attr not in _ATRIBUTOS_SERIE:
                raise CodigoRejeitado(
                    f"atributo fora da allowlist do modo serie: .{no.attr} "
                    "-- so' e' permitido o idioma str.contains/comparacoes "
                    "elementwise (sem I/O nem operacoes cross-row/travessia de "
                    "submodulo)"
                )


def materializar(
    codigo: str,
    nome_funcao: str = NOME_FUNCAO,
    nome_argumento: str = NOME_ARGUMENTO,
    series_mode: bool = False,
):
    """Valida e devolve a funcao viva. Levanta CodigoRejeitado sem executar nada.

    Defaults preservam `corrigir(valor)`. Passe `nome_funcao='detectar'`,
    `nome_argumento='col'` e `series_mode=True` para materializar a funcao de
    deteccao Series (`detectar(col) -> pd.Series`).

    O namespace do exec e' o MESMO nos dois modos (`{re, __builtins__}`): `pd`
    NAO entra. Em series_mode a diferenca esta so' no gate (allowlist de
    atributos em `validar`); `col` chega como argumento em runtime, nao pelo
    namespace.
    """
    validar(
        codigo,
        nome_funcao=nome_funcao,
        nome_argumento=nome_argumento,
        series_mode=series_mode,
    )
    namespace = {"re": re, "__builtins__": dict(_BUILTINS_PERMITIDOS)}
    exec(compile(codigo, "<regra-gerada>", "exec"), namespace, namespace)
    funcao = namespace.get(nome_funcao)
    if not callable(funcao):
        raise CodigoRejeitado("a funcao nao ficou definida apos a execucao")
    return funcao


def testar_fumaca(funcao, amostras, series_mode: bool = False) -> None:
    """Chama a funcao e levanta CodigoRejeitado se ela explodir em runtime.

    O portao AST valida ESTRUTURA e nao pega defeito que so' aparece em runtime.
    Caso real: o agente 1 emitiu `(?i)...|(?i)...` como condicao_regex, o agente 2
    copiou fiel, o AST aprovou (e Python estruturalmente valido) e a funcao lancou
    `re.error` em TODAS as 3977 celulas -- reportando 0% de acerto e 0% de dano,
    visualmente identico a uma regra so' ineficaz. Uma chamada de verdade custa
    microssegundos e transforma falha silenciosa em rejeicao com feedback.

    Escalar (default): `amostras` e' list[str]; chama `funcao(amostra)` por item.

    series_mode=True: `amostras` chega como uma pd.Series ja montada pelo CHAMADOR
    (o sandbox NAO importa pandas -- pd fica fora do namespace do exec). Aplica
    `funcao` a ela UMA vez e exige o contrato Series: o retorno deve ser uma
    pd.Series do MESMO tamanho da entrada. Retorno escalar (ex.: `'%' in col`),
    array cru ou tamanho divergente sao REJEITADOS. O teste de tipo usa
    `type(entrada)` (a propria Series de entrada) como referencia da classe
    pd.Series, evitando importar pandas aqui.
    """
    if not series_mode:
        for amostra in amostras:
            try:
                funcao(amostra)
            except Exception as erro:
                raise CodigoRejeitado(
                    f"a funcao lancou {type(erro).__name__} ao ser chamada com "
                    f'"{amostra}": {erro}. Corrija a causa -- atencao a regex mal '
                    "formada (flags inline como (?i) so' valem no inicio da expressao)."
                ) from erro
        return

    entrada = amostras
    try:
        resultado = funcao(entrada)
    except Exception as erro:
        raise CodigoRejeitado(
            f"a funcao de deteccao lancou {type(erro).__name__} ao ser chamada "
            f"com a coluna de fumaca: {erro}. Corrija a causa -- atencao a regex "
            "mal formada e a metodos que exigem tipos que a coluna nao tem."
        ) from erro
    if not isinstance(resultado, type(entrada)):
        raise CodigoRejeitado(
            "detectar(col) deve devolver uma pd.Series[bool] alinhada a coluna; "
            f"veio {type(resultado).__name__} (retorno escalar ou array cru e' "
            "rejeitado -- use, por ex., col.str.contains(marcador))."
        )
    if len(resultado) != len(entrada):
        raise CodigoRejeitado(
            "detectar(col) deve devolver uma Series do MESMO tamanho da coluna: "
            f"entrada tem {len(entrada)} linhas, saida tem {len(resultado)}."
        )
