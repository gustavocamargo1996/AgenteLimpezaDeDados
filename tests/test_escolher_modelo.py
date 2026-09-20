"""A ferramenta de escolha de modelo, com agente duble (sem rede)."""
import escolher_modelo
from limpeza.esquemas import CodigoGerado


class _AgenteFalso:
    """Devolve sempre o mesmo objeto; nenhuma chamada sai da maquina."""

    def __init__(self, resposta):
        self._resposta = resposta

    def invoke(self, _args):
        if isinstance(self._resposta, Exception):
            raise self._resposta
        return self._resposta


def test_conta_schema_preenchido_e_portao_aceito(monkeypatch):
    bom = CodigoGerado(codigo="def corrigir(valor):\n    return valor.strip()\n",
                       nota_de_traducao="ok")
    monkeypatch.setattr(escolher_modelo, "_agente",
                        lambda schema, papel, modelo: _AgenteFalso(bom))
    r = escolher_modelo.medir("modelo-falso", repeticoes=2)
    assert r["codigo"]["schema_ok"] == 2
    assert r["codigo"]["portao_ok"] == 2


def test_codigo_que_o_portao_rejeita_e_contado_a_parte(monkeypatch):
    ruim = CodigoGerado(codigo="import re\ndef corrigir(valor):\n    return valor\n",
                        nota_de_traducao="com import")
    monkeypatch.setattr(escolher_modelo, "_agente",
                        lambda schema, papel, modelo: _AgenteFalso(ruim))
    r = escolher_modelo.medir("modelo-falso", repeticoes=2)
    assert r["codigo"]["schema_ok"] == 2, "o schema foi preenchido"
    assert r["codigo"]["portao_ok"] == 0, "mas o portao AST rejeitou"


def test_excecao_nao_derruba_a_medicao(monkeypatch):
    monkeypatch.setattr(escolher_modelo, "_agente",
                        lambda schema, papel, modelo: _AgenteFalso(RuntimeError("boom")))
    r = escolher_modelo.medir("modelo-falso", repeticoes=2)
    assert r["codigo"]["schema_ok"] == 0
