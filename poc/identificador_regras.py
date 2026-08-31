"""AGENTE 1 -- o especificador.

Recebe os representantes escolhidos pelo KMeans e devolve, numa unica chamada,
a cadeia de pensamento E o JSON da regra. Os dois nascem juntos de proposito:
a cadeia e' o caminho ate a spec, nao uma justificativa escrita depois. Ve os
pares sujo->limpo das representantes (orcamento de rotulagem do ZeroDC) e
reconstroi o caminho ate a resposta, que e' o mecanismo do Auto-CoT original.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from . import config
from .esquemas import RegraCorrecao

SISTEMA = """Voce especifica regras de correcao para colunas de tabelas sujas.

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


def _formatar_amostra(itens: list[dict]) -> str:
    linhas = []
    for item in itens:
        if item.get("ambiguo"):
            exemplos = ", ".join(f'"{e}"' for e in item.get("exemplos_limpos", [])[:4])
            linhas.append(
                f'  - sujo: "{item["sujo"]}" ({item["frequencia"]}x)  ->  correto: AMBIGUO -- '
                f'este mesmo valor sujo corresponde a {item["limpos_distintos"]} valores '
                f"corretos diferentes (ex.: {exemplos}). Nao existe mapeamento unico."
            )
        else:
            linhas.append(
                f'  - sujo: "{item["sujo"]}"  ->  correto: "{item["limpo"]}"'
                f'   ({item["frequencia"]}x na coluna)'
            )
    return "\n".join(linhas)


def construir_agente(modelo: str | None = None):
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
    agente = agente or construir_agente()
    prompt = ChatPromptTemplate.from_messages([("system", SISTEMA), ("human", HUMANO_BUDGET)])
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
