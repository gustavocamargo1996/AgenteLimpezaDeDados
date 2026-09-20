"""Mede se um modelo serve para a POC, antes de gastar um run inteiro."""
import argparse
import time

from langchain_core.prompts import ChatPromptTemplate

from limpeza import llm, sandbox
from limpeza.correcao import fd as fd_mod
from limpeza.correcao import regras
from limpeza.deteccao import regra as regra_mod
from limpeza.esquemas import (CodigoGerado, DependenciaFuncional, RegraCorrecao,
                              RegraDeteccao)

# Amostras fixas do beers: o que o KMeans mostraria para `abv`.
_SUJOS = """  - "0.075"   (12x na coluna)
  - "0.05%"   (8x na coluna)
  - "0.061"   (15x na coluna)
  - ""   (5x na coluna)"""

_ROTULADOS = """  - sujo: "0.05%"  ->  correto: "0.05"   (8x na coluna)
  - sujo: "0.075"  ->  correto: "0.075"   (12x na coluna)"""

_SPEC = """{"coluna": "abv", "erro_detectado": true, "tipo_erro": "formato",
 "descricao_padrao": "algumas celulas trazem o sinal de porcentagem no fim",
 "condicao_regex": "%$",
 "transformacao": {"tipo": "regex_sub", "padrao": "%$", "substituicao": ""},
 "exemplos": [{"de": "0.05%", "para": "0.05"}], "confianca": 0.95}"""

# (schema, sistema, humano, argumentos, portao a chamar com o codigo, ou None)
CASOS = {
    "deteccao": (RegraDeteccao, regra_mod.SISTEMA, regra_mod.HUMANO,
                 {"coluna": "abv", "amostra": _SUJOS,
                  "total_linhas": 300, "total_distintos": 132},
                 regra_mod.materializar),
    "especificador": (RegraCorrecao, regras.SISTEMA_ESPECIFICADOR, regras.HUMANO_BUDGET,
                      {"coluna": "abv", "amostra": _ROTULADOS,
                       "total_linhas": 300, "total_distintos": 132},
                      None),
    "codigo": (CodigoGerado, regras.SISTEMA_CODIGO, regras.HUMANO_CODIGO,
               {"spec": _SPEC},
               sandbox.validar),
    "fd": (DependenciaFuncional, fd_mod.SISTEMA, fd_mod.HUMANO,
           {"dependente": "state", "candidatos": "brewery-name, city",
            "exemplos": '  - `brewery-name`="21st Amendment Brewery" -> `state`="CA"'},
           None),
}


def _agente(schema, papel, modelo):
    """Ponto unico de construcao; o teste substitui esta funcao."""
    sistema, humano = CASOS[papel][1], CASOS[papel][2]
    prompt = ChatPromptTemplate.from_messages([("system", sistema), ("human", humano)])
    return prompt | llm.construir(schema, papel, modelo=modelo)


def medir(modelo: str, repeticoes: int = 3) -> dict:
    """Invoca cada papel `repeticoes` vezes e conta schema e portao."""
    placar = {}
    for papel, (schema, _s, _h, args, alvo) in CASOS.items():
        schema_ok = portao_ok = 0
        for _ in range(repeticoes):
            try:
                obj = _agente(schema, papel, modelo).invoke(args)
            except Exception:  # falha de parse ou de rede conta como nao preenchido
                continue
            if obj is None:
                continue
            schema_ok += 1
            if alvo is None:
                continue
            try:
                alvo(obj.codigo)
                portao_ok += 1
            except Exception:  # codigo rejeitado pelo portao: conta a parte
                pass
        placar[papel] = {"schema_ok": schema_ok, "total": repeticoes,
                         "portao_ok": None if alvo is None else portao_ok}
    return placar


def _imprimir(modelo: str, placar: dict, segundos: float) -> None:
    """Uma linha por papel, mais o criterio de aprovacao."""
    print(f"\nmodelo: {modelo}   ({segundos:.0f}s)\n")
    print(f"  {'papel':16s}{'schema':>10s}{'portao AST':>14s}")
    for papel, r in placar.items():
        portao = "-" if r["portao_ok"] is None else f"{r['portao_ok']}/{r['total']}"
        print(f"  {papel:16s}{r['schema_ok']:>6d}/{r['total']:<3d}{portao:>14s}")
    det = placar["deteccao"]
    if det["portao_ok"]:
        print("\n  Serve: a deteccao gera codigo que o portao aceita.")
    else:
        print("\n  NAO serve: sem deteccao aceita pelo portao, o limpador sai vazio.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Mede se um modelo consegue preencher os schemas da POC")
    ap.add_argument("--modelo", required=True, help="nome do modelo no provedor ativo")
    ap.add_argument("--repeticoes", type=int, default=3,
                    help="invocacoes por papel; e' o denominador do placar")
    args = ap.parse_args(argv)

    t0 = time.time()
    placar = medir(args.modelo, args.repeticoes)
    _imprimir(args.modelo, placar, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
