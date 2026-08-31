"""Portao AST e execucao restrita do codigo Python gerado pelo LLM."""
import ast
import re

NOME_FUNCAO = "corrigir"
NOME_ARGUMENTO = "valor"

# Namespace do exec e' sempre descartavel, nunca globals(). Ver docs/DECISOES.md#portao-ast.
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

# Allowlist fechada de atributos do modo serie (detectar(col) -> pd.Series).
# Ver docs/DECISOES.md#modo-serie.
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
    """Valida a AST do codigo gerado; levanta CodigoRejeitado sem executar nada."""
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
    """Valida o codigo e devolve a funcao viva; nao materializa se `validar` rejeitar."""
    validar(
        codigo,
        nome_funcao=nome_funcao,
        nome_argumento=nome_argumento,
        series_mode=series_mode,
    )
    # pd fica fora do namespace do exec nos dois modos. Ver docs/DECISOES.md#modo-serie.
    namespace = {"re": re, "__builtins__": dict(_BUILTINS_PERMITIDOS)}
    exec(compile(codigo, "<regra-gerada>", "exec"), namespace, namespace)
    funcao = namespace.get(nome_funcao)
    if not callable(funcao):
        raise CodigoRejeitado("a funcao nao ficou definida apos a execucao")
    return funcao


def testar_fumaca(funcao, amostras, series_mode: bool = False) -> None:
    """Chama a funcao com amostras reais e levanta CodigoRejeitado se ela falhar em runtime."""
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
