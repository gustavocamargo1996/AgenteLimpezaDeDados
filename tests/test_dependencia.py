"""A deteccao por dependencia funcional, com agente duble (sem rede)."""
import types
from pathlib import Path

import pandas as pd

from limpeza.deteccao import dependencia
from limpeza.esquemas import DependenciaFuncional
from limpeza.tipos import Amostra, Coluna, Tabela, Trabalho

FIXTURES = Path(__file__).parent / "fixtures"
LER = dict(dtype=str, keep_default_na=False, na_values=[])


def _environment():
    return (pd.read_csv(FIXTURES / "environment_dirty_300.csv", **LER),
            pd.read_csv(FIXTURES / "environment_clean_300.csv", **LER))


def test_fixture_tem_os_erros_cross_column_esperados():
    """Fixa o fenomeno que este projeto existe para atacar."""
    sujo, limpo = _environment()
    assert len(sujo) == 300
    assert int((sujo["State"] != limpo["State"]).sum()) == 54
    assert int((sujo["Climate_Zone"] != limpo["Climate_Zone"]).sum()) == 26


def test_fixture_nenhum_erro_e_alcancavel_intra_coluna():
    """Todo erro dessas duas colunas usa um valor que tambem aparece correto."""
    sujo, limpo = _environment()
    for col in ("State", "Climate_Zone"):
        err = sujo[col] != limpo[col]
        ambiguo = pd.DataFrame({"v": sujo[col], "e": err}).groupby("v")["e"].transform(
            lambda s: s.any() and (~s).any())
        assert int((err & ~ambiguo).sum()) == 0, f"{col} tem erro alcancavel intra-coluna"


class _AgenteFalso:
    """Devolve sempre a mesma FD; nenhuma chamada sai da maquina."""

    def __init__(self, resposta):
        self._resposta = resposta

    def invoke(self, _args):
        return self(_args)

    # __call__ (alem de invoke) deixa `prompt | agente` coercer este duble
    # num Runnable, sem precisar de RunnableLambda explicito no teste.
    def __call__(self, _args):
        if isinstance(self._resposta, Exception):
            raise self._resposta
        return self._resposta


def _tabela_environment(colunas):
    sujo, limpo = _environment()
    cols = []
    for nome in colunas:
        cols.append(Coluna(nome=nome, sujo=sujo[nome], limpo=limpo[nome],
                           valores_distintos=sorted(sujo[nome].unique()),
                           contagem=sujo[nome].value_counts().to_dict()))
    return Tabela(sujo=sujo, limpo=limpo, nome="environment", colunas=cols)


def _trabalhos(tabela, linhas=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)):
    """Um Trabalho por coluna, com orcamento rotulado nas primeiras linhas."""
    saida = []
    for col in tabela.colunas:
        rot = [{"indice": i, "sujo": col.sujo.at[i], "limpo": col.limpo.at[i],
                "eh_erro": col.sujo.at[i] != col.limpo.at[i]} for i in linhas]
        saida.append(Trabalho(
            coluna=col,
            amostra=Amostra(representantes=[], linhas=set(linhas), rotulados=rot,
                            total_linhas=len(col.sujo), total_distintos=0),
            detector=None))
    return saida


def _mascara_zerada(tabela):
    return pd.DataFrame(0, index=tabela.sujo.index, columns=tabela.sujo.columns, dtype=int)


def test_marca_os_54_erros_de_state():
    tabela = _tabela_environment(["State"])
    trabalhos = _trabalhos(tabela)
    agente = _AgenteFalso(DependenciaFuncional(
        determinante="City", dependente="State", justificativa="cidade fixa o estado"))
    m = dependencia.detectar_dependencia(trabalhos, tabela, _mascara_zerada(tabela), agente)
    assert int(m["State"].sum()) == 54
    err = tabela.sujo["State"] != tabela.limpo["State"]
    assert int((m["State"].astype(bool) & ~err).sum()) == 0, "marcou celula correta"
    assert int(m["City"].sum()) == 0, "marcou a coluna determinante"


def test_marca_os_26_erros_de_climate_zone():
    tabela = _tabela_environment(["Climate_Zone"])
    agente = _AgenteFalso(DependenciaFuncional(
        determinante="City", dependente="Climate_Zone", justificativa="cidade fixa o clima"))
    m = dependencia.detectar_dependencia(_trabalhos(tabela), tabela,
                                         _mascara_zerada(tabela), agente)
    assert int(m["Climate_Zone"].sum()) == 26


def test_fd_reprovada_no_gate_nao_marca_nada():
    """Determinante que nao explica a coluna: o gate barra antes de marcar."""
    tabela = _tabela_environment(["State"])
    agente = _AgenteFalso(DependenciaFuncional(
        determinante="PM2.5", dependente="State", justificativa="ruim de proposito"))
    m = dependencia.detectar_dependencia(_trabalhos(tabela), tabela,
                                         _mascara_zerada(tabela), agente)
    assert int(m["State"].sum()) == 0


def test_fd_reprovada_nao_e_guardada_no_trabalho():
    """Guardar uma FD reprovada faria a cascata contornar o gate."""
    tabela = _tabela_environment(["State"])
    trabalhos = _trabalhos(tabela)
    agente = _AgenteFalso(DependenciaFuncional(
        determinante="PM2.5", dependente="State", justificativa="ruim de proposito"))
    dependencia.detectar_dependencia(trabalhos, tabela, _mascara_zerada(tabela), agente)
    assert getattr(trabalhos[0], "dependencia", None) is None


def test_fd_aprovada_que_explode_ao_marcar_nao_fica_residente(monkeypatch):
    """Sem marca correspondente, a FD residente seria reaproveitada pela cascata."""
    tabela = _tabela_environment(["State"])
    trabalhos = _trabalhos(tabela)
    agente = _AgenteFalso(DependenciaFuncional(
        determinante="City", dependente="State", justificativa="cidade fixa o estado"))

    def explode(*_a, **_k):
        raise RuntimeError("moda quebrou")

    monkeypatch.setattr(dependencia, "_desvios_da_moda", explode)
    mascara = dependencia.detectar_dependencia(
        trabalhos, tabela, _mascara_zerada(tabela), agente)
    assert getattr(trabalhos[0], "dependencia", None) is None
    assert int(mascara["State"].sum()) == 0


def test_fd_aprovada_e_guardada_no_trabalho():
    tabela = _tabela_environment(["State"])
    trabalhos = _trabalhos(tabela)
    agente = _AgenteFalso(DependenciaFuncional(
        determinante="City", dependente="State", justificativa="cidade fixa o estado"))
    dependencia.detectar_dependencia(trabalhos, tabela, _mascara_zerada(tabela), agente)
    assert trabalhos[0].dependencia.determinante == "City"


def test_grupo_sem_moda_nao_marca_ninguem():
    """Duplo filtro esvaziou o grupo: sem referencia, nao ha o que acusar."""
    tabela = _tabela_environment(["State"])
    mascara = _mascara_zerada(tabela)
    mascara["City"] = 1  # nenhuma linha passa no duplo filtro
    mascara["State"] = 1
    m = dependencia._desvios_da_moda(tabela.sujo, mascara, "City", "State")
    assert int(m.sum()) == 0, "marcou sem ter moda para comparar"


def test_excecao_numa_coluna_nao_derruba_as_outras():
    tabela = _tabela_environment(["State", "Climate_Zone"])
    trabalhos = _trabalhos(tabela)
    chamadas = {"n": 0}

    class _AgenteQueFalhaUmaVez:
        def invoke(self, _args):
            return self(_args)

        def __call__(self, _args):
            chamadas["n"] += 1
            if chamadas["n"] == 1:
                raise RuntimeError("boom")
            return DependenciaFuncional(determinante="City", dependente="Climate_Zone",
                                        justificativa="cidade fixa o clima")

    m = dependencia.detectar_dependencia(trabalhos, tabela, _mascara_zerada(tabela),
                                         _AgenteQueFalhaUmaVez())
    assert int(m["State"].sum()) == 0, "a coluna que falhou nao marca"
    assert int(m["Climate_Zone"].sum()) == 26, "a seguinte segue normalmente"


def test_a_justificativa_da_fd_sai_no_relatorio(tmp_path):
    """A cadeia de pensamento e' produto; a FD tambem tem uma."""
    from limpeza import relatorio
    from limpeza.esquemas import DependenciaFuncional

    from limpeza.tipos import Detector

    tabela = _tabela_environment(["State"])
    trabalhos = _trabalhos(tabela)
    # `_cadeias_md` le detector.erro_provavel sem guarda; em producao o detector
    # nunca e' None (o pipeline sempre poe um, real ou nulo).
    trabalhos[0].detector = Detector(codigo="def detectar(col):\n    return col.isin([])\n",
                                     funcao=None, cadeia="sem regra intra-coluna")
    trabalhos[0].medida = {"deteccao": {}}
    trabalhos[0].dependencia = DependenciaFuncional(
        determinante="City", dependente="State",
        justificativa="cada cidade pertence a um unico estado")

    md = relatorio._cadeias_md("environment", trabalhos)
    assert "cada cidade pertence a um unico estado" in md
    assert "City" in md


def test_cascata_reaproveita_a_fd_em_vez_de_propor(monkeypatch):
    """A FD que marcou e' a que corrige; propor de novo poderia discordar."""
    from limpeza.correcao import cascata, fd as fd_real

    chamou = []
    monkeypatch.setattr(fd_real, "propor_fd",
                        lambda *a, **k: chamou.append(a) or DependenciaFuncional(
                            determinante="OUTRA", dependente="State", justificativa="x"))
    monkeypatch.setattr(cascata, "_camada_codigo",
                        lambda coluna, rotulados, agentes: (
                            types.SimpleNamespace(
                                transformacao=types.SimpleNamespace(tipo="nenhuma"),
                                descricao_padrao="fake: sem codigo"),
                            None, ""))

    tabela = _tabela_environment(["State"])
    trabalhos = _trabalhos(tabela)
    trabalhos[0].dependencia = DependenciaFuncional(
        determinante="City", dependente="State", justificativa="ja proposta na deteccao")
    mascara = _mascara_zerada(tabela)
    mascara.loc[1, "State"] = 1  # uma celula marcada, para a cascata ter o que fazer

    cascata.rodar_cascata(trabalhos, tabela, mascara,
                          {"especificador": None, "codigo": None, "fd": None})
    assert chamou == [], "a cascata propos FD de novo em vez de reaproveitar"


def test_cascata_revalida_o_gate_da_fd_reaproveitada(monkeypatch):
    """Reaproveitar a FD aprovada na deteccao nao dispensa o gate na cascata:
    a mascara aqui e' a combinada, com informacao nova (ver docs/DECISOES.md#gates-de-100)."""
    from limpeza.correcao import cascata, fd as fd_real

    chamadas = []
    monkeypatch.setattr(fd_real, "validar_fd",
                        lambda *a, **k: chamadas.append(a) or False)
    monkeypatch.setattr(cascata, "_camada_codigo",
                        lambda coluna, rotulados, agentes: (
                            types.SimpleNamespace(
                                transformacao=types.SimpleNamespace(tipo="nenhuma"),
                                descricao_padrao="fake: sem codigo"),
                            None, ""))

    tabela = _tabela_environment(["State"])
    trabalhos = _trabalhos(tabela)
    trabalhos[0].dependencia = DependenciaFuncional(
        determinante="City", dependente="State", justificativa="ja aprovada na deteccao")
    mascara = _mascara_zerada(tabela)
    mascara.loc[1, "State"] = 1  # uma celula marcada, para a cascata ter o que fazer

    cascata.rodar_cascata(trabalhos, tabela, mascara,
                          {"especificador": None, "codigo": None, "fd": None})

    assert chamadas, "validar_fd nao foi chamado: a FD reaproveitada nao foi revalidada"
    trilha = trabalhos[0].correcao.trilha
    assert trilha["gate_fd"] is False, "gate reprovado tinha de barrar a aplicacao"
    assert trilha["contagem"]["fd"] == 0
    assert trilha["contagem"]["nao_resolvida"] == 1, "celula reprovada tinha de escalar para flag"
