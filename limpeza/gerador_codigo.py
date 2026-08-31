"""AGENTE 2 -- o tradutor.

Recebe o JSON do especificador e devolve codigo Python. Nao ve os dados, nao ve
a amostra, nao reinterpreta o problema: traduz a spec. Se o codigo sair errado,
o defeito e' de traducao -- e essa separacao e' justamente o que permite saber
onde o pipeline quebrou.

O codigo passa pelo portao AST antes de existir como funcao. Rejeicao volta
para o LLM como feedback, ate MAX_TENTATIVAS_CODIGO.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from . import config, sandbox
from .esquemas import CodigoGerado, RegraCorrecao

IDENTIDADE = '''def corrigir(valor):
    """Regra 'nenhuma': nada a corrigir localmente. Devolve o valor intacto."""
    return valor
'''

SISTEMA = """Voce traduz uma especificacao JSON de regra de correcao em uma funcao Python.

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

HUMANO = """Especificacao a traduzir:

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


def construir_agente(modelo: str | None = None):
    llm = ChatOpenAI(
        model=modelo or config.MODELO_LLM,
        temperature=config.TEMPERATURA,
        timeout=config.TIMEOUT_LLM,
        max_retries=2,
    )
    return llm.with_structured_output(CodigoGerado)


def traduzir(regra: RegraCorrecao, agente=None) -> dict:
    """Devolve {codigo, nota, funcao, tentativas, rejeicoes}.

    `funcao` vem None se o codigo foi rejeitado ate o fim -- e nesse caso a
    coluna e' reportada como falha de traducao, sem fallback silencioso. Um
    fallback deterministico aqui esconderia exatamente o dado que interessa.
    """
    if regra.transformacao.tipo == "nenhuma" or not regra.erro_detectado:
        return {
            "codigo": IDENTIDADE,
            "nota": "Spec sem transformacao aplicavel; o tradutor nao foi acionado.",
            "funcao": sandbox.materializar(IDENTIDADE),
            "tentativas": 0,
            "rejeicoes": [],
        }

    agente = agente or construir_agente()
    spec = regra.model_dump_json(indent=2, exclude={"cadeia_de_pensamento"})

    mensagens = [("system", SISTEMA), ("human", HUMANO)]
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
            # Amostras vindas da propria spec: se a regra nao sobrevive aos exemplos
            # que a justificaram, nao ha por que solta-la sobre a coluna inteira.
            amostras = [e.de for e in regra.exemplos] or ["12.0 oz", "", "null"]
            sandbox.testar_fumaca(funcao, amostras)
        except sandbox.CodigoRejeitado as erro:
            rejeicoes.append(str(erro))
            mensagens = [("system", SISTEMA), ("human", HUMANO), ("human", REPARO)]
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
        "funcao": None,
        "tentativas": config.MAX_TENTATIVAS_CODIGO,
        "rejeicoes": rejeicoes,
    }
