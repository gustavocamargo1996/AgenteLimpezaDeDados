"""Camada 2 da cascata: informacao mutua, portao de FD (gate de 100%) e a aplicacao."""
import types

import pandas as pd

from limpeza.correcao import cascata, fd
from limpeza.esquemas import DependenciaFuncional


def test_calc_mi_acha_a_determinante_verdadeira():
    df = pd.DataFrame({
        "A": ["x", "x", "y", "y", "z", "z"],
        "B": ["1", "1", "2", "2", "3", "3"],   # B determinada por A
        "C": ["p", "q", "p", "q", "p", "q"],   # C: pares (A,C) ocorrem 1x -> filtrados
    })
    mi = fd.calc_mi(df, "A")
    assert mi["B"] >= 0.5
    assert mi["B"] > mi["C"]
    cand = fd.candidatos_determinantes(df, "A", limiar=0.5)
    assert "B" in cand and "C" not in cand and "A" not in cand


def test_validar_fd_reprova_quando_a_moda_diverge_do_ground_truth():
    df = pd.DataFrame({
        "brewery": ["X", "X", "X"],
        "city": ["A", "A", "B"],   # X quase sempre A; idx2 B (marcado)
    })
    mascara = pd.DataFrame({"brewery": [0, 0, 0], "city": [0, 0, 1]})
    fd_ok = DependenciaFuncional(determinante="brewery", dependente="city", justificativa="t")

    rot_falha = [{"indice": 2, "sujo": "B", "limpo": "B_TRUE", "eh_erro": True}]
    assert fd.validar_fd(fd_ok, rot_falha, df, mascara, "city") is False


def test_validar_fd_passa_quando_a_moda_bate_e_aplicar_fd_corrige():
    df = pd.DataFrame({
        "brewery": ["X", "X", "X"],
        "city": ["A", "A", "B"],
    })
    mascara = pd.DataFrame({"brewery": [0, 0, 0], "city": [0, 0, 1]})
    fd_ok = DependenciaFuncional(determinante="brewery", dependente="city", justificativa="t")

    rot_ok = [{"indice": 2, "sujo": "B", "limpo": "A", "eh_erro": True}]
    assert fd.validar_fd(fd_ok, rot_ok, df, mascara, "city") is True

    mud = fd.aplicar_fd(fd_ok, df, [2], mascara, "city")
    assert mud.get(2) == "A"


def test_determinante_inexistente_reprova_sem_lancar_e_sem_mudanca():
    df = pd.DataFrame({
        "brewery": ["X", "X", "X"],
        "city": ["A", "A", "B"],
    })
    mascara = pd.DataFrame({"brewery": [0, 0, 0], "city": [0, 0, 1]})
    rot_ok = [{"indice": 2, "sujo": "B", "limpo": "A", "eh_erro": True}]
    fd_ruim = DependenciaFuncional(determinante="inexistente", dependente="city", justificativa="t")

    assert fd.validar_fd(fd_ruim, rot_ok, df, mascara, "city") is False
    assert fd.aplicar_fd(fd_ruim, df, [2], mascara, "city") == {}


def test_cascata_camada_2_fd_passa_resolve_as_marcadas(monkeypatch):
    """Camada 1 (codigo) reprova -- escala tudo; a FD passa e resolve as marcadas."""
    df = pd.DataFrame({
        "brewery": ["A", "A", "B", "B"],
        "state":   ["",  "CA", "NY", ""],
    })
    mascara_completa = pd.DataFrame({"brewery": [0, 0, 0, 0], "state": [1, 0, 0, 1]})
    mascara_col = mascara_completa["state"]
    rotulados = {
        "rows": [
            {"indice": 0, "sujo": "", "limpo": "CA", "eh_erro": True},
            {"indice": 1, "sujo": "CA", "limpo": "CA", "eh_erro": False},
        ],
        "itens": [], "total_linhas": 4, "total_distintos": 3,
    }

    def fake_camada_codigo(coluna, rot, agentes):
        # funcao None -> gate reprova -> tudo escala para a camada 2.
        return (types.SimpleNamespace(transformacao=types.SimpleNamespace(tipo="nenhuma"),
                                      descricao_padrao="fake: gate reprova"), None, "")

    fd_fake = DependenciaFuncional(determinante="brewery", dependente="state",
                                   justificativa="mesma cervejaria -> mesmo estado")

    monkeypatch.setattr(cascata, "_camada_codigo", fake_camada_codigo)
    monkeypatch.setattr(cascata.fd_mod, "candidatos_determinantes",
                        lambda df_, alvo, limiar=None: ["brewery"])
    monkeypatch.setattr(cascata.fd_mod, "propor_fd",
                        lambda coluna, rows, candidatos, df_, agente=None: fd_fake)
    monkeypatch.setattr(cascata.fd_mod, "validar_fd", lambda fd, rows, df_, mascara, coluna: True)
    monkeypatch.setattr(cascata.fd_mod, "aplicar_fd",
                        lambda fd, df_, pendentes, mascara, coluna: {0: "CA", 3: "NY"})

    correcoes, trilha = cascata.rodar_coluna(
        "state", df, mascara_col, rotulados, mascara_completa, agentes={},
    )

    cont = trilha["contagem"]
    assert cont == {"codigo": 0, "fd": 2, "nao_resolvida": 0}
    assert trilha["gate_codigo"] is False and trilha["gate_fd"] is True
    assert trilha["trilha_celula"][0] == "fd" and trilha["trilha_celula"][3] == "fd"
    assert correcoes == {0: "CA", 3: "NY"}
    assert sum(cont.values()) == trilha["marcadas"] == 2
