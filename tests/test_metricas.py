"""metricas_deteccao/metricas_correcao: P/R/F1, degenerado (n/d, nunca 0.0), holdout."""
import pandas as pd

from limpeza import metricas


def test_coluna_sem_erro_real_e_nao_mensuravel():
    serie = pd.Series(["a", "b", "c"])
    m = metricas.metricas_deteccao(pd.Series([0, 0, 0]), serie, serie, pd.Index([]))
    assert m["mensuravel"] is False
    assert m["f1"] is None, "0.0 mentiria: nao ha erro para achar"


def test_deteccao_perfeita_da_f1_1():
    sujo = pd.Series(["N/A", "42", "55"])
    limpo = pd.Series(["", "42", "55"])
    m = metricas.metricas_deteccao(pd.Series([1, 0, 0]), sujo, limpo, pd.Index([]))
    assert m["precisao"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_precisao_recall_f1_parciais():
    sujo = pd.Series(["a", "B", "c", "d", "e"])
    limpo = pd.Series(["a", "b", "c", "D", "e"])   # erros em idx1 e idx3
    mask = pd.Series([0, 1, 0, 0, 1])              # preve idx1 (TP) e idx4 (FP); erra idx3 (FN)
    d = metricas.metricas_deteccao(mask, sujo, limpo)
    assert d["precisao"] == 0.5 and d["recall"] == 0.5 and d["f1"] == 0.5
    assert d["mensuravel"] is True


def test_degenerado_marca_sem_erro_real_conta_falsos_positivos():
    """tp+fn=0 mas a mascara marca celulas: ainda n/d, porem os FP sao contabilizados."""
    sujo0 = pd.Series(["a", "b", "c"])
    limpo0 = pd.Series(["a", "b", "c"])            # sem erro real
    mask0b = pd.Series([1, 0, 1])                  # tudo falso positivo; tp+fn=0
    d0b = metricas.metricas_deteccao(mask0b, sujo0, limpo0)
    assert d0b["mensuravel"] is False and d0b["f1"] is None
    assert d0b["falsos_positivos"] == 2


def test_holdout_remove_o_falso_positivo():
    sujo = pd.Series(["a", "B", "c", "d", "e"])
    limpo = pd.Series(["a", "b", "c", "D", "e"])
    mask = pd.Series([0, 1, 0, 0, 1])
    dh = metricas.metricas_deteccao(mask, sujo, limpo, holdout=[4])  # remove o FP
    assert dh["precisao"] == 1.0 and dh["recall"] == 0.5 and dh["mensuravel"] is True


def test_metricas_correcao_holdout_exclui_celulas_rotuladas():
    sujo = pd.Series(["a", "B", "c", "d", "e"])
    limpo = pd.Series(["a", "b", "c", "D", "e"])
    corr = pd.Series(["a", "b", "c", "D", "e"])    # corrigiu idx1 e idx3
    mc = metricas.metricas_correcao(corr, sujo, limpo, holdout=[1])
    assert mc["erradas"] == 1 and mc["acertos"] == 1 and mc["dano"] == 0
