"""Teste de fumaca: roda tudo que NAO precisa de LLM.

Valida caminhos, embeddings ONNX, KMeans e o portao AST -- e imprime os
representantes que o agente 1 receberia. Rode isto antes de gastar API:
se o KMeans esta escolhendo mal, nenhum prompt salva.

    python verificar_ambiente.py
"""
import os
import sys

from limpeza import amostragem, config, dados, sandbox
from limpeza.amostragem import Embedder

OK, FALHA = "  [ok] ", "  [!!] "


def secao(titulo):
    print(f"\n{titulo}\n" + "-" * len(titulo))


def main() -> int:
    problemas = 0

    secao("1. Caminhos")
    for rotulo, caminho in [
        ("dataset sujo", config.CSV_SUJO),
        ("dataset limpo", config.CSV_LIMPO),
        ("modelo ONNX", config.ARQ_ONNX),
        ("tokenizer", config.ARQ_TOKENIZER),
    ]:
        existe = caminho.exists()
        problemas += 0 if existe else 1
        print(f"{OK if existe else FALHA}{rotulo}: {caminho}")

    if problemas:
        print("\nCaminho faltando -- ajuste ZERODC_DIR no .env")
        return 1

    secao("2. Dados")
    tabela = dados.carregar(colunas=config.COLUNAS_PADRAO)
    print(f"{OK}{len(tabela.sujo)} linhas, {len(tabela.sujo.columns)} colunas")
    for col in tabela.colunas:
        erradas = int((col.sujo != col.limpo).sum())
        taxa = erradas / len(col.sujo)
        print(f"{OK}{col.nome:8s}: {erradas:5d} erradas ({taxa:5.1%}), "
              f"{len(col.valores_distintos):4d} valores distintos")

    secao("3. Embeddings (MiniLM ONNX local)")
    embedder = Embedder()
    matriz = embedder.codificar(["12.0 oz", "12.0", "N/A", "abacaxi"])
    print(f"{OK}shape {matriz.shape}, norma media {float((matriz ** 2).sum(axis=1).mean()):.4f}")
    similar = float(matriz[0] @ matriz[1])
    distante = float(matriz[0] @ matriz[3])
    coerente = similar > distante
    problemas += 0 if coerente else 1
    print(f"{OK if coerente else FALHA}'12.0 oz'~'12.0' = {similar:.3f} > "
          f"'12.0 oz'~'abacaxi' = {distante:.3f}")

    secao("4. KMeans -- o que o agente 1 vai ver")
    for col in tabela.colunas:
        amostra = amostragem.representantes(col, emb=embedder)
        contagem = col.contagem
        print(f"\n  {col.nome} ({len(amostra.representantes)} representantes "
              f"de {len(col.valores_distintos)}):")
        for v in amostra.representantes:
            distintos = col.limpo[col.sujo == v].unique().tolist()
            if len(distintos) > 1:
                exemplos = ", ".join(f'"{d}"' for d in distintos[:3])
                print(f'    ?? "{v}"  ({contagem.get(v, 0)}x)  AMBIGUO: '
                      f'{len(distintos)} valores corretos distintos ({exemplos}...)')
            elif distintos and distintos[0] != v:
                print(f'    -> "{v}"  ({contagem.get(v, 0)}x)  correto: "{distintos[0]}"')
            else:
                print(f'       "{v}"  ({contagem.get(v, 0)}x)')

    secao("5. Portao AST do codigo gerado")
    casos = [
        ("funcao valida", 'def corrigir(valor):\n    return re.sub(r"\\s*oz\\.?$", "", valor)', True),
        ("nome errado", "def fix(valor):\n    return valor", False),
        ("assinatura errada", "def corrigir(v, extra):\n    return v", False),
        ("com import", "import os\ndef corrigir(valor):\n    return valor", False),
        ("chamada proibida", 'def corrigir(valor):\n    return eval(valor)', False),
        ("codigo solto no topo", 'x = 1\ndef corrigir(valor):\n    return valor', False),
    ]
    for rotulo, codigo, deve_passar in casos:
        try:
            sandbox.materializar(codigo)
            passou = True
            motivo = ""
        except sandbox.CodigoRejeitado as e:
            passou = False
            motivo = str(e)[:60]
        certo = passou == deve_passar
        problemas += 0 if certo else 1
        esperado = "aceitar" if deve_passar else "rejeitar"
        print(f"{OK if certo else FALHA}{rotulo}: devia {esperado}, "
              f"{'aceitou' if passou else 'rejeitou'}" + (f" ({motivo})" if motivo else ""))

    secao("6. Chave de API")
    tem_chave = bool(os.getenv("OPENAI_API_KEY"))
    print(f"{OK if tem_chave else FALHA}OPENAI_API_KEY "
          f"{'presente' if tem_chave else 'AUSENTE -- COPIE (nao renomeie) .env.example para .env'}")

    secao("Resultado")
    if problemas:
        print(f"  {problemas} problema(s). Corrija antes de rodar main.py")
        return 1
    print("  Parte offline integra." + ("" if tem_chave else " Falta so' a chave de API."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
