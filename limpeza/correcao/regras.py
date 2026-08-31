"""Etapa 4, camada 1: agente especificador (JSON) e agente tradutor (codigo) da correcao."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .. import config, sandbox
from ..esquemas import CodigoGerado, RegraCorrecao

# --- Agente 1: especificador ---

SISTEMA_ESPECIFICADOR = """Voce especifica regras de correcao para colunas de tabelas sujas.

Sua saida tem duas partes inseparaveis: a CADEIA DE PENSAMENTO (como voce chegou \
la) e a REGRA em JSON. A cadeia e' o produto principal -- alguem vai le-la para \
entender o dado, entao ela precisa ser especifica e citar valores concretos da \
amostra, nunca generica.

VOCABULARIO FECHADO de transformacao. Escolha exatamente um:
  - regex_sub  : busca `padrao` e troca por `substituicao`
  - mapeamento : dicionario explicito de->para
  - constante  : todo valor que casar vira um valor fixo
  - nenhuma    : nao ha regra local capaz de corrigir

DISCIPLINA (o erro caro aqui e' regra ampla demais, nao regra ausente):
1. Prefira `erro_detectado: false` a inventar defeito que a amostra nao sustenta.
2. Se a transformacao nao for `nenhuma`, `condicao_regex` e' OBRIGATORIA. Ela decide
   QUANDO agir. Sem ela a regra atropela celulas que ja estavam corretas -- e numa
   coluna com 5% de erro isso destroi os outros 95%.
3. Valor ausente (celula vazia) NAO e' corrigivel por regra local: nao da' para
   inventar a informacao que nao esta la. Nesse caso use `tipo_erro: valor_ausente`,
   `transformacao.tipo: nenhuma`, e explique em `limites_da_regra` o que seria
   preciso para resolver (por exemplo, deduzir de outra coluna).
4. Nao use conhecimento de mundo que a amostra nao mostra. Se a amostra tem 3
   valores, a regra vale para o padrao daqueles 3 -- diga isso em vez de fingir
   cobertura universal.
5. Em `limites_da_regra`, cite ao menos um valor concreto da amostra que a regra
   deve deixar intacto.

Escreva a cadeia de pensamento em portugues do Brasil."""

HUMANO_BUDGET = """Coluna: `{coluna}`
Tabela: {total_linhas} linhas, {total_distintos} valores distintos nesta coluna.

Voce TEM o valor correto das celulas abaixo -- e' o orcamento de rotulagem gasto \
nesta coluna. Os representantes foram escolhidos por agrupamento semantico (os mais \
atipicos e os mais tipicos de cada grupo):

{amostra}

O valor correto ja esta dado. Sua tarefa nao e' adivinha-lo: e' reconstruir o \
raciocinio que leva do valor sujo ate ele, e generalizar isso numa regra que \
funcione nas celulas que voce NAO viu."""


def _formatar_item(item: dict) -> str:
    if item.get("ambiguo"):
        exemplos = ", ".join(f'"{e}"' for e in item.get("exemplos_limpos", [])[:4])
        return (
            f'  - sujo: "{item["sujo"]}" ({item["frequencia"]}x)  ->  correto: AMBIGUO -- '
            f'este mesmo valor sujo corresponde a {item["limpos_distintos"]} valores '
            f"corretos diferentes (ex.: {exemplos}). Nao existe mapeamento unico."
        )
    return (
        f'  - sujo: "{item["sujo"]}"  ->  correto: "{item["limpo"]}"'
        f'   ({item["frequencia"]}x na coluna)'
    )


def _formatar_amostra(itens: list[dict]) -> str:
    return "\n".join(_formatar_item(item) for item in itens)


def construir_agente_especificador(modelo: str | None = None):
    llm = ChatOpenAI(
        model=modelo or config.MODELO_LLM,
        temperature=config.TEMPERATURA,
        timeout=config.TIMEOUT_LLM,
        max_retries=2,
    )
    return llm.with_structured_output(RegraCorrecao)


def especificar(
    coluna: str,
    itens: list[dict],
    total_linhas: int,
    total_distintos: int,
    agente=None,
) -> RegraCorrecao:
    agente = agente or construir_agente_especificador()
    prompt = ChatPromptTemplate.from_messages(
        [("system", SISTEMA_ESPECIFICADOR), ("human", HUMANO_BUDGET)]
    )
    cadeia = prompt | agente
    regra: RegraCorrecao = cadeia.invoke(
        {
            "coluna": coluna,
            "amostra": _formatar_amostra(itens),
            "total_linhas": total_linhas,
            "total_distintos": total_distintos,
        }
    )
    regra.coluna = coluna  # o nome da coluna e' fato, nao opiniao do modelo
    return regra


# --- Agente 2: tradutor ---

IDENTIDADE = '''def corrigir(valor):
    """Regra 'nenhuma': nada a corrigir localmente. Devolve o valor intacto."""
    return valor
'''

SISTEMA_CODIGO = """Voce traduz uma especificacao JSON de regra de correcao em uma funcao Python.

CONTRATO RIGIDO -- o codigo passa por validacao AST antes de rodar e e' rejeitado se:
  - houver qualquer coisa alem de UMA definicao de funcao (sem import, sem constante
    de modulo, sem chamada solta);
  - a funcao nao se chamar exatamente `corrigir`;
  - a assinatura nao for exatamente `def corrigir(valor):`;
  - houver import (o modulo `re` JA esta disponivel no escopo -- use direto);
  - houver chamada a eval, exec, open, getattr, globals, __import__ e afins.

REGRAS DE TRADUCAO:
1. `valor` chega sempre como str. Devolva sempre str.
2. Aplique `condicao_regex` como guarda: se o valor NAO casar, devolva `valor`
   inalterado. Isso nao e' opcional -- e' o que impede a regra de estragar celula
   que ja estava correta.
3. Nao acrescente inteligencia que a spec nao pediu. Se a spec so' cobre um padrao,
   o codigo cobre um padrao. Voce e' tradutor, nao co-autor.
4. Nao levante excecao: entrada inesperada devolve `valor` inalterado.
5. Sem docstring longa nem comentario decorativo. O codigo deve caber em poucas linhas."""

HUMANO_CODIGO = """Especificacao a traduzir:

```json
{spec}
```

Gere a funcao `corrigir(valor)`."""

REPARO = """A tentativa anterior foi REJEITADA pelo validador.

Codigo rejeitado:
```python
{codigo}
```

Motivo da rejeicao: {motivo}

Corrija e devolva a funcao novamente, respeitando o contrato rigido."""


def construir_agente_codigo(modelo: str | None = None):
    llm = ChatOpenAI(
        model=modelo or config.MODELO_LLM,
        temperature=config.TEMPERATURA,
        timeout=config.TIMEOUT_LLM,
        max_retries=2,
    )
    return llm.with_structured_output(CodigoGerado)


def traduzir(regra: RegraCorrecao, agente=None) -> dict:
    """Traduz a spec em codigo Python; tenta ate MAX_TENTATIVAS_CODIGO vezes se o portao rejeitar."""
    if regra.transformacao.tipo == "nenhuma" or not regra.erro_detectado:
        return {
            "codigo": IDENTIDADE,
            "nota": "Spec sem transformacao aplicavel; o tradutor nao foi acionado.",
            "funcao": sandbox.materializar(IDENTIDADE),
            "tentativas": 0,
            "rejeicoes": [],
        }

    agente = agente or construir_agente_codigo()
    spec = regra.model_dump_json(indent=2, exclude={"cadeia_de_pensamento"})

    mensagens = [("system", SISTEMA_CODIGO), ("human", HUMANO_CODIGO)]
    rejeicoes: list[str] = []
    ultimo_codigo = ""

    for tentativa in range(1, config.MAX_TENTATIVAS_CODIGO + 1):
        prompt = ChatPromptTemplate.from_messages(mensagens)
        resposta: CodigoGerado = (prompt | agente).invoke(
            {"spec": spec, "codigo": ultimo_codigo, "motivo": rejeicoes[-1] if rejeicoes else ""}
        )
        ultimo_codigo = resposta.codigo.strip()
        try:
            funcao = sandbox.materializar(ultimo_codigo)
            # Testa com os proprios exemplos da spec: se falha neles, nao vale soltar na coluna.
            amostras = [e.de for e in regra.exemplos] or ["12.0 oz", "", "null"]
            sandbox.testar_fumaca(funcao, amostras)
        except sandbox.CodigoRejeitado as erro:  # rejeitado: guarda o motivo e tenta de novo com prompt de reparo
            rejeicoes.append(str(erro))
            mensagens = [("system", SISTEMA_CODIGO), ("human", HUMANO_CODIGO), ("human", REPARO)]
            continue
        return {
            "codigo": ultimo_codigo,
            "nota": resposta.nota_de_traducao,
            "funcao": funcao,
            "tentativas": tentativa,
            "rejeicoes": rejeicoes,
        }

    return {
        "codigo": ultimo_codigo,
        "nota": "TRADUCAO FALHOU: codigo rejeitado pelo validador em todas as tentativas.",
        "funcao": None,  # sem fallback: reporta a falha em vez de esconder o dado
        "tentativas": config.MAX_TENTATIVAS_CODIGO,
        "rejeicoes": rejeicoes,
    }
