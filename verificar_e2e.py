"""Testes OFFLINE do modo --e2e (nao chamam LLM, nao gastam API).

Aditivo e separado de verificar_ambiente.py (que serve de regressao do modo
antigo). Cobre os testes do plano secao 8:
  1. calc_mi num df com A->B conhecida;
  2. metricas_deteccao (P/R/F1), incluindo o caso degenerado (tp+fn=0 -> n/d,
     nao 0.0) e o holdout;
  3. portao de FD (validar_fd reprova com 1 erro; passa sem erro);
  4. sandbox com detectar(col)->pd.Series (series_mode): materializa o idioma
     contains, smoke rejeita retorno escalar, allowlist rejeita I/O e cross-row
     (col.to_csv/col.duplicated/pd.read_csv), o contrato antigo corrigir(valor)
     escalar fica intacto, e um teste-controle declara que a fronteira REAL de
     seguranca e' operacional (nao a allowlist ser exaustiva).
  5. construir_mascara Series (por-coluna, POSICIONAL): marca so' o marcador,
     granularidade de falha por-coluna (nao-Series/tamanho errado -> coluna 0 +
     log), e nao le clean.
  7b. _classificar: aplica a funcao Series a uma lista, len-correto; funcao None
     / retorno escalar / excecao -> [False]*n.

Loop de refinamento da deteccao (--iteracoes-deteccao N>1), tudo sem API:
  8. _amostrar_oraculo: 1 previsto-sujo + 1 previsto-limpo, respeita `usados`,
     deterministico;
  9. _montar_feedback_det: marca FP/FN e e' cumulativo;
 10. refinar_regra_deteccao com agente FAKE: sucesso muda a funcao, rejeicao
     (regex invalida) mantem a anterior.
 11. _selecionar_suspeito (scorer suspeito do previsto-limpo): raridade de char
     alnum isolada, ponderacao por frequencia-DE-CELULA, bonus 'x' opcional,
     determinismo/respeita referencia e guarda de div-zero (coluna degenerada).

    python verificar_e2e.py
"""
import types

import numpy as np
import pandas as pd
from langchain_core.runnables import RunnableLambda

from limpeza import config, deteccao, empacotar, metricas, sandbox
from limpeza.correcao import cascata, fd
from limpeza.deteccao import oraculo, refino, regra
from limpeza.esquemas import DependenciaFuncional, RegraDeteccao
from limpeza.tipos import Coluna

OK, FALHA = "  [ok] ", "  [!!] "


class Contador:
    def __init__(self):
        self.problemas = 0

    def check(self, condicao: bool, rotulo: str):
        if condicao:
            print(f"{OK}{rotulo}")
        else:
            print(f"{FALHA}{rotulo}")
            self.problemas += 1


def secao(titulo):
    print(f"\n{titulo}\n" + "-" * len(titulo))


def _mat_det(codigo: str):
    """Materializa uma funcao de deteccao Series (`detectar(col)`) no series_mode."""
    return sandbox.materializar(
        codigo, nome_funcao="detectar", nome_argumento="col", series_mode=True
    )


def teste_mi(c: Contador):
    secao("1. calc_mi: A->B conhecida")
    df = pd.DataFrame({
        "A": ["x", "x", "y", "y", "z", "z"],
        "B": ["1", "1", "2", "2", "3", "3"],   # B determinada por A
        "C": ["p", "q", "p", "q", "p", "q"],   # C: pares (A,C) ocorrem 1x -> filtrados
    })
    mi = fd.calc_mi(df, "A")
    c.check(mi["B"] >= 0.5, f"B tem MI normalizada alta de A: {mi['B']}")
    c.check(mi["B"] > mi["C"], f"B ({mi['B']}) > C ({mi['C']}) como determinante de A")
    cand = fd.candidatos_determinantes(df, "A", limiar=0.5)
    c.check("B" in cand and "C" not in cand and "A" not in cand,
            f"candidatos determinantes de A = {cand}")


def teste_metricas_deteccao(c: Contador):
    secao("2. metricas_deteccao: P/R/F1, degenerado (tp+fn=0 -> n/d) e holdout")
    sujo = pd.Series(["a", "B", "c", "d", "e"])
    limpo = pd.Series(["a", "b", "c", "D", "e"])   # erros em idx1 e idx3
    mask = pd.Series([0, 1, 0, 0, 1])              # preve idx1 (TP) e idx4 (FP); erra idx3 (FN)
    d = metricas.metricas_deteccao(mask, sujo, limpo)
    c.check(d["precisao"] == 0.5 and d["recall"] == 0.5 and d["f1"] == 0.5
            and d["mensuravel"] is True,
            f"P/R/F1 = {d['precisao']}/{d['recall']}/{d['f1']} mensuravel={d['mensuravel']} "
            "(esperado 0.5/0.5/0.5, mensuravel)")

    # Degenerado A: sem erro real e sem predicao (tp+fn=0). Agora devolve n/d
    # (mensuravel:false), NAO 0.0 silencioso.
    sujo0 = pd.Series(["a", "b", "c"])
    limpo0 = pd.Series(["a", "b", "c"])            # sem erro real
    mask0 = pd.Series([0, 0, 0])
    d0 = metricas.metricas_deteccao(mask0, sujo0, limpo0)
    c.check(d0["mensuravel"] is False and d0["precisao"] is None
            and d0["recall"] is None and d0["f1"] is None,
            f"degenerado tp+fn=0 -> mensuravel:false, P/R/F1 = n/d (nao 0.0): "
            f"mensuravel={d0['mensuravel']} precisao={d0['precisao']}")

    # Degenerado B (plano secao 8): a mascara MARCA celulas mas nenhuma e' erro
    # real -> ainda n/d, mas os FP sao contabilizados.
    mask0b = pd.Series([1, 0, 1])                  # tudo falso positivo; tp+fn=0
    d0b = metricas.metricas_deteccao(mask0b, sujo0, limpo0)
    c.check(d0b["mensuravel"] is False and d0b["f1"] is None
            and d0b["falsos_positivos"] == 2,
            f"marca sem erro real -> n/d, mas FP=2 contabilizados: "
            f"mensuravel={d0b['mensuravel']} FP={d0b['falsos_positivos']}")

    dh = metricas.metricas_deteccao(mask, sujo, limpo, holdout=[4])  # remove o FP
    c.check(dh["precisao"] == 1.0 and dh["recall"] == 0.5 and dh["mensuravel"] is True,
            f"holdout remove o FP idx4: P={dh['precisao']} R={dh['recall']} (esperado 1.0/0.5)")

    # correcao: holdout exclui as celulas rotuladas
    corr = pd.Series(["a", "b", "c", "D", "e"])    # corrigiu idx1 e idx3
    mc = metricas.metricas_correcao(corr, sujo, limpo, holdout=[1])
    c.check(mc["erradas"] == 1 and mc["acertos"] == 1 and mc["dano"] == 0,
            f"metricas_correcao com holdout=[1]: erradas={mc['erradas']} "
            f"acertos={mc['acertos']} dano={mc['dano']}")


def teste_gate_fd(c: Contador):
    secao("3. Portao de FD: reprova com 1 erro, passa sem erro")
    df = pd.DataFrame({
        "brewery": ["X", "X", "X"],
        "city": ["A", "A", "B"],   # X quase sempre A; idx2 B (marcado)
    })
    mascara = pd.DataFrame({
        "brewery": [0, 0, 0],
        "city": [0, 0, 1],         # idx2 detectado como erro
    })
    fd_ok = DependenciaFuncional(determinante="brewery", dependente="city", justificativa="t")

    rot_falha = [{"indice": 2, "sujo": "B", "limpo": "B_TRUE", "eh_erro": True}]
    reprova = fd.validar_fd(fd_ok, rot_falha, df, mascara, "city")
    c.check(reprova is False,
            "FD reprova quando a moda ('A') diverge do ground truth ('B_TRUE')")

    rot_ok = [{"indice": 2, "sujo": "B", "limpo": "A", "eh_erro": True}]
    passa = fd.validar_fd(fd_ok, rot_ok, df, mascara, "city")
    c.check(passa is True, "FD passa quando a moda ('A') bate com o ground truth ('A')")

    mud = fd.aplicar_fd(fd_ok, df, [2], mascara, "city")
    c.check(mud.get(2) == "A", f"aplicar_fd corrige idx2 pela moda: {mud}")

    fd_ruim = DependenciaFuncional(determinante="inexistente", dependente="city", justificativa="t")
    c.check(fd.validar_fd(fd_ruim, rot_ok, df, mascara, "city") is False,
            "determinante inexistente -> reprova sem lancar")
    c.check(fd.aplicar_fd(fd_ruim, df, [2], mascara, "city") == {},
            "determinante inexistente -> nenhuma mudanca")


def _rejeita(codigo: str, series_mode: bool) -> bool:
    """True se `validar` rejeita `codigo` no modo pedido (sem executar)."""
    try:
        sandbox.validar(
            codigo, nome_funcao="detectar", nome_argumento="col",
            series_mode=series_mode,
        )
        return False
    except sandbox.CodigoRejeitado:
        return True


def teste_sandbox_detectar(c: Contador):
    secao("4. Sandbox detectar(col)->pd.Series (series_mode) + allowlist")

    # (a) Idioma contains: materializa e roda sobre pd.Series(['0.05','0.05%']).
    f_pct = _mat_det(
        "def detectar(col):\n"
        "    return col.astype(str).str.contains('%', regex=False, na=False)\n"
    )
    res = f_pct(pd.Series(["0.05", "0.05%"]))
    c.check(isinstance(res, pd.Series) and list(res) == [False, True],
            f"contains materializa e roda: ['0.05','0.05%'] -> {list(res)}")

    # (b) Contrato antigo corrigir(valor) ESCALAR intacto (defaults, sem series).
    f_old = sandbox.materializar("def corrigir(valor):\n    return valor.strip()")
    c.check(f_old(" x ") == "x", "contrato antigo corrigir(valor) intacto (defaults)")

    # (c) Sob contrato 'detectar', a funcao 'corrigir' e' rejeitada.
    c.check(_rejeita("def corrigir(col):\n    return col", series_mode=True),
            "sob contrato 'detectar', funcao 'corrigir' e' rejeitada")

    # (d) Smoke Series REJEITA retorno escalar ('%' in col -> bool).
    try:
        f_escalar = _mat_det("def detectar(col):\n    return '%' in col\n")
        sandbox.testar_fumaca(
            f_escalar, pd.Series(regra._AMOSTRAS_FUMACA), series_mode=True
        )
        rejeitou_escalar = False
    except sandbox.CodigoRejeitado:
        rejeitou_escalar = True
    c.check(rejeitou_escalar,
            "smoke Series rejeita retorno escalar ('%' in col)")

    # (e) SEGURANCA -- allowlist rejeita I/O e cross-row (na validacao, sem rodar).
    c.check(_rejeita("def detectar(col):\n    return col.to_csv('x')\n", series_mode=True),
            "allowlist rejeita I/O: col.to_csv('x')")
    c.check(_rejeita("def detectar(col):\n    return col.duplicated()\n", series_mode=True),
            "allowlist rejeita cross-row: col.duplicated()")
    c.check(_rejeita("def detectar(col):\n    return pd.read_csv('x')\n", series_mode=True),
            "allowlist rejeita travessia/pd: pd.read_csv('x')")
    c.check(_rejeita("def detectar(col):\n    return pd.io.common.get_handle('x','w')\n",
                     series_mode=True),
            "allowlist rejeita submodulo: pd.io.common.get_handle('x','w')")
    c.check(_rejeita("def detectar(col):\n    return col.mode()\n", series_mode=True),
            "allowlist rejeita cross-row: col.mode()")

    # (f) A allowlist NAO barra no modo ESCALAR (default): o mesmo atributo passa
    # no gate escalar -- prova que a allowlist e' ADITIVA e so' do series_mode.
    c.check(not _rejeita("def detectar(col):\n    return col.duplicated()\n", series_mode=False),
            "escalar (default) NAO aplica a allowlist (aditiva, so' no series_mode)")

    # (g) TESTE-CONTROLE (negativo, declarativo): a allowlist NAO e' prova de
    # contencao total. `str.contains` esta na allowlist e ainda assim permite
    # regex arbitraria; a fronteira REAL de seguranca e' OPERACIONAL (beers,
    # dados publicos, offline), nao a lista ser exaustiva. Este idioma "perigoso
    # porem permitido" passa de proposito -- documenta o limite honesto.
    passou_permitido = not _rejeita(
        "def detectar(col):\n    return col.str.contains(r'(a+)+$', regex=True, na=False)\n",
        series_mode=True,
    )
    c.check(passou_permitido,
            "controle: idioma permitido passa; fronteira real e' OPERACIONAL, "
            "nao a allowlist ser exaustiva (limite declarado)")


def teste_construir_mascara(c: Contador):
    secao("5. construir_mascara Series: por-coluna, POSICIONAL, sem clean")
    # Marca so' o marcador '%': idioma contains sobre a coluna inteira.
    df = pd.DataFrame({"c": ["0.05", "0.06%", "0.07"]})
    f_pct = _mat_det(
        "def detectar(col):\n"
        "    return col.astype(str).str.contains('%', regex=False, na=False)\n"
    )
    log = []
    mask = deteccao.aplicar_detectores(df, {"c": f_pct}, log)
    c.check(list(mask["c"]) == [0, 1, 0] and set(mask["c"].unique()) <= {0, 1},
            f"mascara marca so' a celula com '%': {list(mask['c'])}")
    c.check(log == [], "sem log quando detectar devolve Series booleana valida")

    # Retorno nao-Series (escalar) -> coluna INTEIRA 0 + 1 log por-coluna.
    f_escalar = _mat_det("def detectar(col):\n    return True\n")
    log2 = []
    mask2 = deteccao.aplicar_detectores(df, {"c": f_escalar}, log2)
    c.check(list(mask2["c"]) == [0, 0, 0] and len(log2) == 1
            and log2[0]["coluna"] == "c" and "nao-Series" in log2[0]["motivo"],
            f"retorno nao-Series -> coluna 0 + 1 log por-coluna: {log2}")

    # Tamanho divergente -> coluna INTEIRA 0 + 1 log por-coluna (posicional,
    # nunca boolean-index por indice divergente).
    f_curta = _mat_det("def detectar(col):\n    return col.isin([])\n")
    # forca tamanho errado: uma funcao que devolve Series de outro tamanho.
    def _f_tam_errado(coluna):
        return pd.Series([True])
    log3 = []
    mask3 = deteccao.aplicar_detectores(df, {"c": _f_tam_errado}, log3)
    c.check(list(mask3["c"]) == [0, 0, 0] and len(log3) == 1
            and "tamanho divergente" in log3[0]["motivo"],
            f"tamanho divergente -> coluna 0 + log: {log3}")

    # DETECTA_NADA (col.isin([])) -> tudo 0, sem log.
    f_nada = _mat_det(deteccao.DETECTA_NADA)
    log4 = []
    mask4 = deteccao.aplicar_detectores(df, {"c": f_nada}, log4)
    c.check(list(mask4["c"]) == [0, 0, 0] and log4 == [],
            f"DETECTA_NADA (col.isin([])) marca nada, sem log: {list(mask4['c'])}")

    # auditavel: construir_mascara nao LE clean. Duas checagens objetivas:
    #  (a) a assinatura nao tem parametro clean/limpo;
    #  (b) o codigo executavel (sem comentarios/docstrings) nao usa nome
    #      clean/limpo nem read_csv. Usamos tokenize para ignorar o texto do
    #      docstring (que legitimamente MENCIONA que nao le clean).
    import inspect
    import io
    import tokenize

    params = set(inspect.signature(deteccao.aplicar_detectores).parameters)
    params |= set(inspect.signature(deteccao.construir_mascara).parameters)
    c.check("clean" not in params and "limpo" not in params,
            f"assinatura sem parametro clean/limpo: {sorted(params)}")

    fonte = (inspect.getsource(deteccao.aplicar_detectores)
             + inspect.getsource(deteccao.construir_mascara))
    nomes = set()
    for tok in tokenize.generate_tokens(io.StringIO(fonte).readline):
        if tok.type == tokenize.NAME:
            nomes.add(tok.string)
    c.check({"clean", "limpo", "read_csv"}.isdisjoint(nomes),
            "codigo executavel nao referencia clean/limpo/read_csv (ignorando docstring)")


def teste_classificar(c: Contador):
    secao("5b. _classificar: aplicacao Series posicional, robusta a falha")
    f_oz = _mat_det("def detectar(col):\n    return col.str.endswith('oz')\n")
    marcas = oraculo._classificar(f_oz, ["12", "12 oz", "16", "16 oz"])
    c.check(marcas == [False, True, False, True],
            f"classifica lista posicionalmente, len-correto: {marcas}")
    c.check(all(isinstance(x, bool) for x in marcas),
            "devolve python bool (nao numpy.bool_)")

    # funcao None -> [False]*n
    c.check(oraculo._classificar(None, ["a", "b", "c"]) == [False, False, False],
            "funcao None -> [False]*n")

    # retorno escalar (nao-Series) -> [False]*n (robusto, nunca lanca)
    f_escalar = _mat_det("def detectar(col):\n    return True\n")
    c.check(oraculo._classificar(f_escalar, ["a", "b"]) == [False, False],
            "retorno nao-Series -> [False]*n")

    # excecao em runtime -> [False]*n
    def _f_boom(coluna):
        raise ValueError("boom")
    c.check(oraculo._classificar(_f_boom, ["a", "b"]) == [False, False],
            "excecao na funcao -> [False]*n")

    # lista vazia -> []
    c.check(oraculo._classificar(f_oz, []) == [],
            "lista vazia -> []")


def teste_cascata_fd_passa(c: Contador):
    """Ramo camada 2 (FD) PASSA dentro de rodar_cascata -- lacuna de integracao
    que a auditoria apontou (antes so' o gate de FD era testado isolado). Camada 1
    reprova (escala tudo), FD passa e resolve as celulas marcadas."""
    secao("7. Cascata: ramo FD-passa (offline)")

    from limpeza.esquemas import DependenciaFuncional

    # 'state' vazio em 0 e 3; 'brewery' limpa determina o valor.
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
        # funcao None -> gate reprova -> tudo escala para a camada 2. 3-tupla:
        # codigo_str irrelevante (nao entra na trilha porque o gate reprova).
        return (types.SimpleNamespace(transformacao=types.SimpleNamespace(tipo="nenhuma"),
                                      descricao_padrao="fake: gate reprova"), None, "")

    def fake_candidatos(df_, alvo, limiar=None):
        return ["brewery"]  # ha determinante candidato -> camada 2 roda

    fd_fake = DependenciaFuncional(determinante="brewery", dependente="state",
                                   justificativa="mesma cervejaria -> mesmo estado")

    def fake_propor(coluna, rows, candidatos, df_, agente=None):
        return fd_fake

    def fake_validar(fd, rows, df_, mascara, coluna):
        return True  # gate FD passa

    def fake_aplicar(fd, df_, pendentes, mascara, coluna):
        return {0: "CA", 3: "NY"}  # moda por brewery resolve as duas marcadas

    orig = (cascata._camada_codigo, cascata.fd_mod.candidatos_determinantes,
            cascata.fd_mod.propor_fd, cascata.fd_mod.validar_fd,
            cascata.fd_mod.aplicar_fd)
    (cascata._camada_codigo, cascata.fd_mod.candidatos_determinantes,
     cascata.fd_mod.propor_fd, cascata.fd_mod.validar_fd,
     cascata.fd_mod.aplicar_fd) = (
        fake_camada_codigo, fake_candidatos, fake_propor, fake_validar,
        fake_aplicar)
    try:
        correcoes, trilha = cascata.rodar_coluna(
            "state", df, mascara_col, rotulados, mascara_completa, agentes={},
        )
    finally:
        (cascata._camada_codigo, cascata.fd_mod.candidatos_determinantes,
         cascata.fd_mod.propor_fd, cascata.fd_mod.validar_fd,
         cascata.fd_mod.aplicar_fd) = orig

    cont = trilha["contagem"]
    c.check(cont == {"codigo": 0, "fd": 2, "nao_resolvida": 0},
            f"contagem por camada = {cont}")
    c.check(trilha["gate_codigo"] is False and trilha["gate_fd"] is True,
            "gate: codigo reprovou, FD passou")
    c.check(trilha["trilha_celula"][0] == "fd" and trilha["trilha_celula"][3] == "fd",
            "trilha por celula: 0 e 3 resolvidas por fd")
    c.check(correcoes == {0: "CA", 3: "NY"}, f"correcoes finais = {correcoes}")
    soma = sum(cont.values())
    c.check(soma == trilha["marcadas"] == 2, f"soma {soma} == marcadas {trilha['marcadas']}")


def _gerar_e_importar(nome, detectores, plano, colunas):
    """Gera um limpador em dir temporario, py_compila e importa como modulo."""
    import importlib.util
    import py_compile
    import tempfile
    from pathlib import Path

    pasta = Path(tempfile.mkdtemp(prefix="limpador_teste_"))
    caminho = pasta / f"{nome}.py"
    empacotar.escrever_limpador(
        caminho=caminho, dataset="teste",
        detectores_codigo=detectores, plano_correcao=plano, colunas=colunas,
    )
    py_compile.compile(str(caminho), doraise=True)  # levanta PyCompileError se nao compilar
    spec = importlib.util.spec_from_file_location(nome, caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, caminho


def teste_gerar_limpador(c: Contador):
    """gerar_limpador escreve um .py que py_compila, importa e aplica: corrige as
    celulas certas (codigo e FD) e FLAGA as marcadas nao-cobertas."""
    secao("14. gerar_limpador: .py autonomo compila, importa, aplica")

    detectores = {
        "ounces": "def detectar(col):\n    return col.str.endswith('oz')\n",
        "abv": "def detectar(col):\n    return col.str.contains('%', regex=False, na=False)\n",
        "state": "def detectar(col):\n    return col.eq('')\n",
    }
    plano = {
        "ounces": [{"tipo": "codigo",
                    "codigo": "def corrigir(valor):\n    return re.sub(r'\\s*oz$', '', valor)\n"}],
        "abv": [],  # detecta mas nao tem corretor/FD -> vira flag
        "state": [{"tipo": "fd", "determinante": "brewery", "dependente": "state"}],
    }
    colunas = ["ounces", "abv", "state"]
    mod, caminho = _gerar_e_importar("limpador_basico", detectores, plano, colunas)
    c.check(caminho.exists(), f"limpador escrito e compilado: {caminho.name}")

    df = pd.DataFrame({
        "brewery": ["A", "A", "B", "B"],
        "ounces": ["12 oz", "16", "10 oz", "20"],
        "abv": ["0.05", "0.06%", "0.07", "0.08%"],
        "state": ["", "CA", "NY", ""],
    })
    corrigido, flags = mod.aplicar(df)

    c.check(list(corrigido["ounces"]) == ["12", "16", "10", "20"],
            f"codigo corrige ounces (strip oz): {list(corrigido['ounces'])}")
    c.check(list(corrigido["state"]) == ["CA", "CA", "NY", "NY"],
            f"FD corrige state (moda por brewery): {list(corrigido['state'])}")
    c.check(list(corrigido["abv"]) == ["0.05", "0.06%", "0.07", "0.08%"],
            f"abv sem corretor -> intacto: {list(corrigido['abv'])}")
    c.check(list(flags["abv"]) == [False, True, False, True],
            f"abv marcado e nao-corrigido -> FLAG idx1,idx3: {list(flags['abv'])}")
    c.check(not flags["ounces"].any() and not flags["state"].any(),
            "ounces e state resolvidos -> sem flag")
    c.check(list(corrigido["brewery"]) == ["A", "A", "B", "B"],
            "brewery (sem detector/plano) fica intacto")


def teste_fd_duplo_filtro(c: Contador):
    """DUPLO-FILTRO da FD: a moda EXCLUI as celulas marcadas do dependente. Caso
    onde SEM o filtro a moda daria o sujo dominante (erro), COM o filtro da' o
    certo."""
    secao("15. FD duplo-filtro: moda exclui o dependente marcado")

    detectores = {"state": "def detectar(col):\n    return col.eq('')\n"}
    plano = {"state": [{"tipo": "fd", "determinante": "brewery", "dependente": "state"}]}
    mod, _ = _gerar_e_importar("limpador_duplo", detectores, plano, ["state"])

    # brewery unica 'A'. state: 3 vazios MARCADOS + 2 'CA' limpos. SEM o filtro do
    # dependente a moda do grupo seria '' (3 vs 2) e nada seria corrigido; COM o
    # duplo-filtro o pool e' ['CA','CA'] -> 'CA'.
    df = pd.DataFrame({
        "brewery": ["A", "A", "A", "A", "A"],
        "state":   ["", "", "", "CA", "CA"],
    })
    corrigido, flags = mod.aplicar(df)
    c.check(list(corrigido["state"]) == ["CA", "CA", "CA", "CA", "CA"],
            f"duplo-filtro exclui os vazios da moda -> tudo 'CA': {list(corrigido['state'])}")
    c.check(not flags["state"].any(), "todas as marcadas resolvidas pela FD (sem flag)")


def teste_coocorrencia_codigo_fd(c: Contador):
    """CO-OCORRENCIA codigo+FD na MESMA coluna: o codigo pega umas celulas, a FD
    pega o resto -- AMBOS aplicados, na ordem, sem descartar a FD."""
    secao("16. Co-ocorrencia codigo+FD numa coluna (ambos, na ordem)")

    detectores = {
        "city": "def detectar(col):\n"
                "    return col.str.contains(r' [A-Z]{2}$', regex=True, na=False) | col.eq('')\n",
    }
    plano = {
        "city": [
            {"tipo": "codigo",
             "codigo": "def corrigir(valor):\n    return re.sub(r'\\s*[A-Z]{2}$', '', valor)\n"},
            {"tipo": "fd", "determinante": "brewery", "dependente": "city"},
        ],
    }
    mod, _ = _gerar_e_importar("limpador_coocorr", detectores, plano, ["city"])

    df = pd.DataFrame({
        "brewery": ["A", "A", "A", "B", "B"],
        "city":    ["Portland OR", "Salem", "", "Austin", ""],
    })
    corrigido, flags = mod.aplicar(df)
    # idx0 'Portland OR' -> CODIGO strip ' OR' -> 'Portland' (a FD sozinha daria
    # 'Salem'; prova que o codigo rodou). idx2/idx4 vazios escalam -> FD ->
    # 'Salem'/'Austin'.
    c.check(corrigido["city"].iloc[0] == "Portland",
            f"idx0 resolvido pelo CODIGO (FD daria 'Salem'): {corrigido['city'].iloc[0]!r}")
    c.check(corrigido["city"].iloc[2] == "Salem" and corrigido["city"].iloc[4] == "Austin",
            f"idx2/idx4 resolvidos pela FD apos o codigo: "
            f"{corrigido['city'].iloc[2]!r}/{corrigido['city'].iloc[4]!r}")
    c.check(list(corrigido["city"]) == ["Portland", "Salem", "Salem", "Austin", "Austin"],
            f"coluna final: {list(corrigido['city'])}")
    c.check(not flags["city"].any(), "todas as marcadas resolvidas (codigo+FD)")


def teste_ordem_fd_df_entrada(c: Contador):
    """ORDEM: a FD agrupa sobre o df de ENTRADA mesmo quando outra coluna
    processada ANTES (aqui 'city', o determinante) ja foi corrigida na copia."""
    secao("17. Ordem: FD le o df de entrada, nao a copia ja corrigida")

    detectores = {
        "city": "def detectar(col):\n    return col.str.contains(r' [A-Z]{2}$', regex=True, na=False)\n",
        "state": "def detectar(col):\n    return col.eq('')\n",
    }
    plano = {
        "city": [{"tipo": "codigo",
                  "codigo": "def corrigir(valor):\n    return re.sub(r'\\s*[A-Z]{2}$', '', valor)\n"}],
        "state": [{"tipo": "fd", "determinante": "city", "dependente": "state"}],
    }
    # 'city' processada ANTES de 'state'; a FD de state usa city como determinante.
    mod, _ = _gerar_e_importar("limpador_ordem", detectores, plano, ["city", "state"])

    df = pd.DataFrame({
        "city":  ["Boston MA", "Boston", "Boston", "Boston"],
        "state": ["", "MA", "MA", "MA"],
    })
    corrigido, flags = mod.aplicar(df)
    # city[0] 'Boston MA' e' marcado e corrigido -> 'Boston'. state[0] vazio,
    # marcado. Lendo o df ORIGINAL, valor_det='Boston MA' e NENHUMA linha
    # nao-marcada de city vale 'Boston MA' -> pool vazio -> state[0] FLAG. Se a FD
    # lesse a copia (city ja 'Boston'), o pool seria idx1-3 -> moda 'MA' e state[0]
    # viraria 'MA' -- o bug que este teste barra.
    c.check(corrigido["city"].iloc[0] == "Boston",
            f"city corrigida primeiro (strip ' MA'): {corrigido['city'].iloc[0]!r}")
    c.check(corrigido["state"].iloc[0] == "" and bool(flags["state"].iloc[0]) is True,
            "FD leu o df ORIGINAL (valor_det='Boston MA', pool vazio) -> state[0] FLAG, "
            f"nao 'MA': state[0]={corrigido['state'].iloc[0]!r}")


def teste_determinante_sem_detector(c: Contador):
    """Determinante SEM detector -> _mascara da' tudo-False nele; a FD aplica (nao
    filtra por um determinante 'marcado' inexistente)."""
    secao("18. Determinante sem detector: mascara tudo-False, FD aplica")

    detectores = {"state": "def detectar(col):\n    return col.eq('')\n"}
    plano = {"state": [{"tipo": "fd", "determinante": "brewery", "dependente": "state"}]}
    # 'brewery' NAO esta em colunas -> sem detector -> mask[brewery] tudo-False.
    mod, _ = _gerar_e_importar("limpador_det_sem_det", detectores, plano, ["state"])

    df = pd.DataFrame({
        "brewery": ["A", "A", "B", "B"],
        "state":   ["", "CA", "NY", ""],
    })
    corrigido, flags = mod.aplicar(df)
    c.check("brewery" not in mod._DETECTORES,
            "brewery nao tem entrada em _DETECTORES do limpador")
    c.check(list(corrigido["state"]) == ["CA", "CA", "NY", "NY"],
            f"FD aplica com determinante sem detector: {list(corrigido['state'])}")
    c.check(not flags["state"].any(), "todas as marcadas resolvidas (sem flag)")


class FakeEmbedder:
    """Embedder OFFLINE e deterministico -- nao carrega ONNX.

    Vetor por valor = (comprimento, soma dos ordinais). Basta para o
    farthest-point de `_amostrar_oraculo` ser estavel e testavel sem os arquivos
    do modelo. Reproduz so' o contrato usado: `.codificar(list[str]) -> ndarray`.
    """

    def codificar(self, textos):
        return np.array(
            [[float(len(t)), float(sum(ord(ch) for ch in t))] for t in textos]
        )


def teste_amostrar_oraculo(c: Contador):
    secao("8. _amostrar_oraculo: 1 sujo + 1 limpo, respeita usados, deterministico")
    emb = FakeEmbedder()
    funcao = _mat_det("def detectar(col):\n    return col.str.endswith('oz')\n")
    valores = ["12", "16", "12.0 oz", "16.0 oz"]
    sujo_col = pd.Series(["12", "16", "12.0 oz", "16.0 oz"])

    esc = oraculo._amostrar_oraculo(funcao, valores, emb, usados=set(), n=2, sujo_col=sujo_col)
    sujo = [v for v in esc if v.endswith("oz")]
    limpo = [v for v in esc if not v.endswith("oz")]
    c.check(len(esc) == 2 and len(sujo) == 1 and len(limpo) == 1,
            f"escolheu 1 previsto-sujo + 1 previsto-limpo: {esc}")

    esc2 = oraculo._amostrar_oraculo(funcao, valores, emb, usados=set(), n=2, sujo_col=sujo_col)
    c.check(esc == esc2, f"deterministico (duas chamadas iguais): {esc} == {esc2}")

    esc3 = oraculo._amostrar_oraculo(funcao, valores, emb, usados={"12.0 oz"}, n=2, sujo_col=sujo_col)
    c.check("12.0 oz" not in esc3 and any(v.endswith("oz") for v in esc3),
            f"respeita usados (nao repete '12.0 oz', pega o outro sujo): {esc3}")


def teste_montar_feedback(c: Contador):
    secao("9. _montar_feedback_det: marca FP/FN e e' cumulativo")
    funcao = _mat_det("def detectar(col):\n    return col.str.endswith('oz')\n")
    # '12.0 oz': regra marca SUJO, oraculo diz correto -> FALSO POSITIVO.
    # '17'    : regra deixa LIMPO, oraculo diz erro     -> FALSO NEGATIVO.
    rotulos = [
        {"valor": "12.0 oz", "eh_erro_real": False},
        {"valor": "17", "eh_erro_real": True},
    ]
    texto = refino._montar_feedback_det(funcao, rotulos)
    linha_fp = next(l for l in texto.splitlines() if '"12.0 oz"' in l)
    linha_fn = next(l for l in texto.splitlines() if '"17"' in l)
    c.check("FALSO POSITIVO" in linha_fp, f"FP marcado para '12.0 oz': {linha_fp.strip()}")
    c.check("FALSO NEGATIVO" in linha_fn, f"FN marcado para '17': {linha_fn.strip()}")

    # Cumulativo: um terceiro rotulo entra e os anteriores permanecem.
    rotulos.append({"valor": "20.0 oz", "eh_erro_real": True})  # sujo & erro -> OK
    texto2 = refino._montar_feedback_det(funcao, rotulos)
    c.check('"12.0 oz"' in texto2 and '"17"' in texto2 and '"20.0 oz"' in texto2
            and len(texto2.splitlines()) == 3,
            "cumulativo: inclui todos os rotulos acumulados (3 linhas)")


def _coluna_teste(nome, sujo_col, limpo_col):
    """Coluna sintetica para os testes do loop de refinamento."""
    return Coluna(nome=nome, sujo=sujo_col, limpo=limpo_col,
                  valores_distintos=sorted(sujo_col.unique().tolist()),
                  contagem=sujo_col.value_counts().to_dict())


def _refinar(coluna, detector, agente, emb, iteracoes=1, amostras=2):
    """Roda o refinamento com o orcamento do teste fixado no config."""
    antes = (config.ITERACOES_DETECCAO, config.AMOSTRAS_POR_ITERACAO)
    config.ITERACOES_DETECCAO, config.AMOSTRAS_POR_ITERACAO = iteracoes, amostras
    try:
        return deteccao.refinar_regra_deteccao(detector, coluna, agente, emb=emb)
    finally:
        config.ITERACOES_DETECCAO, config.AMOSTRAS_POR_ITERACAO = antes


def teste_refinar_deteccao(c: Contador):
    secao("10. refinar_regra_deteccao: fake agent (sucesso muda; rejeicao mantem)")
    emb = FakeEmbedder()
    # ounces-like: '12.0 oz'/'16.0 oz' sao erros (limpo '12'/'16'); '12'/'16' ok.
    sujo_col = pd.Series(["12.0 oz", "16.0 oz", "12", "16"])
    limpo_col = pd.Series(["12", "16", "12", "16"])
    col = _coluna_teste("ounces", sujo_col, limpo_col)

    funcao0 = _mat_det(deteccao.DETECTA_NADA)
    regra0 = RegraDeteccao(erro_provavel=False, condicao_regex=None,
                           codigo=deteccao.DETECTA_NADA, cadeia="parte de col.isin([])")

    # --- SUCESSO: o fake devolve uma regra Series que flaga o sufixo 'oz' ---
    def fake_ok(_prompt_value):
        return RegraDeteccao(
            erro_provavel=True, condicao_regex=r"\s*oz$",
            codigo="def detectar(col):\n    return col.str.endswith('oz')\n",
            cadeia="o feedback disse que '12.0 oz' e' erro; flago o sufixo oz",
        )

    det = _refinar(col, regra.detector_de(regra0, funcao0),
                   RunnableLambda(fake_ok), emb)
    f = det.funcao
    c.check(f is not funcao0
            and oraculo._classificar(f, ["12.0 oz", "12"]) == [True, False],
            "sucesso: funcao final DIFERE, marca '12.0 oz' e NAO marca '12'")
    passo = det.historico[0]
    c.check(len(det.historico) == 1 and passo["mudou"] is True
            and passo["status"] == "aplicada",
            f"historico com 1 passo, mudou=sim, status={passo['status']}")
    c.check(det.orcamento["n_valores"] >= 1 and det.orcamento["n_celulas"] >= 1,
            f"orcamento registrado: {det.orcamento['n_valores']}v/"
            f"{det.orcamento['n_celulas']}c")

    # --- REJEICAO: o fake devolve regex invalida (Series) -> falha no smoke,
    # mantem a regra anterior. `str.contains` passa na allowlist, mas '(' e'
    # regex mal formada -> re.error em runtime -> CodigoRejeitado. ---
    def fake_ruim(_prompt_value):
        return RegraDeteccao(
            erro_provavel=True, condicao_regex=None,
            codigo="def detectar(col):\n    return col.str.contains('(', regex=True, na=False)\n",
            cadeia="regex proposital invalida (paren sem fechar)",
        )

    det_r = _refinar(col, regra.detector_de(regra0, funcao0),
                     RunnableLambda(fake_ruim), emb)
    c.check(det_r.funcao is funcao0 and det_r.codigo == regra0.codigo,
            "rejeicao: mantem EXATAMENTE a funcao/codigo anterior (mesma funcao viva)")
    passo_r = det_r.historico[0]
    c.check(passo_r["mudou"] is False and passo_r["status"].startswith("rejeitada"),
            f"historico marca rejeicao: {passo_r['status']}")


def teste_selecionar_suspeito(c: Contador):
    secao("11. _selecionar_suspeito: raridade alnum, freq-de-celula, bonus x, guardas")
    emb = FakeEmbedder()

    # (a) Raridade ALNUM isolada (bonus x DESLIGADO): numa coluna majoritariamente
    # numerica, o valor com uma LETRA rara ('a') deve ser o pick vs numericos
    # comuns. NAO usa '%' (nao-alnum, invisivel de proposito).
    candidatos = ["10", "11", "12", "13", "12a"]
    sujo_col = pd.Series(["10", "11", "12", "13", "10", "11", "12", "13", "12a"])
    pick = oraculo._selecionar_suspeito(
        candidatos, referencia=set(), n=1, embedder=emb, sujo_col=sujo_col,
        usar_bonus_x=False,
    )
    c.check(pick == ["12a"],
            f"rankeia char alnum raro ('a') acima de numerico comum, bonus OFF: {pick}")

    # (b) char_counts pondera por frequencia-DE-CELULA (nao por valor distinto).
    # '9' esta em 1 valor distinto mas 10 celulas -> frequente -> raridade BAIXA;
    # '3' em 1 valor distinto e 1 celula -> raro -> raridade ALTA.
    col_freq = pd.Series(["9"] * 10 + ["3"])
    cc = oraculo._contar_chars_alnum(col_freq)
    c.check(cc.get("9") == 10 and cc.get("3") == 1,
            f"char_counts conta CELULAS (nao distintos): 9={cc.get('9')} 3={cc.get('3')}")
    s9 = oraculo._score_artefato("9", cc, usar_bonus_x=False)
    s3 = oraculo._score_artefato("3", cc, usar_bonus_x=False)
    c.check(s9 < s3 and abs(s9 - 0.1) < 1e-9 and abs(s3 - 1.0) < 1e-9,
            f"raridade cai com freq-de-celula: score('9')={s9} < score('3')={s3}")

    # (c) Bonus de 'x' LIGADO sobe um valor com 'x' vs DESLIGADO.
    cc_x = oraculo._contar_chars_alnum(pd.Series(["1x2", "10", "20"]))
    on = oraculo._score_artefato("1x2", cc_x, usar_bonus_x=True)
    off = oraculo._score_artefato("1x2", cc_x, usar_bonus_x=False)
    c.check(on > off, f"bonus x ligado sobe o valor com 'x': on={on} > off={off}")

    # (d) Deterministico + respeita referencia (nao devolve valor da referencia).
    cand2 = ["10", "20x", "30", "40"]
    col2 = pd.Series(["10", "20x", "30", "40", "99"])
    r1 = oraculo._selecionar_suspeito(cand2, referencia={"99"}, n=2, embedder=emb,
                                       sujo_col=col2, usar_bonus_x=False)
    r2 = oraculo._selecionar_suspeito(cand2, referencia={"99"}, n=2, embedder=emb,
                                       sujo_col=col2, usar_bonus_x=False)
    c.check(r1 == r2, f"deterministico (duas chamadas iguais): {r1} == {r2}")
    c.check(set(r1) <= set(cand2) and "99" not in r1,
            f"respeita referencia: picks em candidatos e nunca a referencia '99': {r1}")

    # (e) Guarda de div-zero: coluna degenerada (todos sem char alnum -> artefato
    # max=0) nao lanca e ainda devolve o pick por diversidade.
    cand_deg = ["--", "++", "**"]
    col_deg = pd.Series(["--", "++", "**"])
    try:
        deg = oraculo._selecionar_suspeito(cand_deg, referencia=set(), n=1, embedder=emb,
                                            sujo_col=col_deg, usar_bonus_x=True)
        lancou = False
    except Exception:  # noqa: BLE001
        deg, lancou = None, True
    c.check(not lancou and isinstance(deg, list) and len(deg) == 1,
            f"artefato max=0 (coluna nao-alnum) nao lanca, retorna 1 pick: {deg}")


def teste_precisao_no_oraculo(c: Contador):
    secao("12. _precisao_no_oraculo: precisao no rotulado + Armadilha 1 (cega sem clean)")
    # Rotulos montados a mao: 2 sujos ('A','B') + 2 limpos ('C','D').
    rotulos = [
        {"valor": "A", "eh_erro_real": True},
        {"valor": "B", "eh_erro_real": True},
        {"valor": "C", "eh_erro_real": False},
        {"valor": "D", "eh_erro_real": False},
    ]

    # Marca so' os 2 sujos -> precisao 1.0.
    so_sujos = _mat_det("def detectar(col):\n    return col.isin(['A', 'B'])\n")
    p1 = refino._precisao_no_oraculo(so_sujos, rotulos)
    c.check(p1 == 1.0, f"marca so' os 2 sujos -> precisao 1.0: {p1}")

    # Marca 2 limpos + 1 sujo -> 1/3 = 0.333 (mira do plano; inequivoco < 0.8).
    ampla = _mat_det("def detectar(col):\n    return col.isin(['A', 'C', 'D'])\n")
    p2 = refino._precisao_no_oraculo(ampla, rotulos)
    c.check(p2 is not None and abs(p2 - 1 / 3) < 1e-9,
            f"marca 2 limpos + 1 sujo -> precisao 0.333: {p2}")
    c.check(p2 < config.LIMITE_PRECISAO_DETECCAO,
            f"0.333 < LIMITE {config.LIMITE_PRECISAO_DETECCAO} (a guarda rejeitaria): {p2}")

    # Marca 0 -> None (over-flagging nao mensuravel; guarda inerte).
    nada = _mat_det(deteccao.DETECTA_NADA)
    p3 = refino._precisao_no_oraculo(nada, rotulos)
    c.check(p3 is None, f"marca 0 -> None (guarda inerte): {p3}")

    # ARMADILHA 1 DOCUMENTADA: com rotulos TODOS sujos, uma regra genuinamente
    # ampla (aqui 'return True', marca TUDO) tem precisao 1.0 no rotulado -- a
    # guarda e' CEGA ao over-flagging porque nao ha clean no oraculo para punir.
    # Nao e' resolvivel sem acesso amplo ao clean; NAO tentar corrigir com fracao
    # de celula (quebraria ounces). Limitacao declarada; mitigada em parte pelo
    # piso final (a amplitude tende a fazer o amostrador colher clean ao vivo).
    rotulos_so_sujos = [
        {"valor": "12.0 oz", "eh_erro_real": True},
        {"valor": "16.0 oz", "eh_erro_real": True},
    ]
    marca_tudo = _mat_det("def detectar(col):\n    return col.notna()\n")
    p_arm = refino._precisao_no_oraculo(marca_tudo, rotulos_so_sujos)
    c.check(p_arm == 1.0 and not (p_arm < config.LIMITE_PRECISAO_DETECCAO),
            f"Armadilha 1: rotulos todos sujos + regra ampla -> precisao 1.0, "
            f"guarda NAO fira (cegueira declarada): {p_arm}")


def teste_guarda_e_piso(c: Contador):
    secao("13. guarda (A) rejeita ampla / aceita precisa; piso (B); ounces-like nao punido")
    emb = FakeEmbedder()

    # Dominio comum: 4 valores distintos; SO' 'A' e' erro real (limpo != valor);
    # 'B','C','D' sao limpos. amostras_por_iter=4 rotula TODOS -> precisao
    # deterministica independente da ordem de amostragem.
    sujo_col = pd.Series(["A", "B", "C", "D"])
    limpo_col = pd.Series(["A_ok", "B", "C", "D"])
    col = _coluna_teste("col", sujo_col, limpo_col)

    def novo_nada():
        return _mat_det(deteccao.DETECTA_NADA)

    def regra_nada():
        return RegraDeteccao(erro_provavel=False, condicao_regex=None,
                             codigo=deteccao.DETECTA_NADA, cadeia="col.isin([])")

    # --- GUARDA (A) REJEITA: fake devolve regra AMPLA que marca {A,B,C}
    # (2 limpos + 1 sujo -> precisao 0.333, claramente < 0.8) -> REJEITA. ---
    def fake_ampla(_prompt_value):
        return RegraDeteccao(
            erro_provavel=True, condicao_regex=None,
            codigo="def detectar(col):\n    return col.isin(['A', 'B', 'C'])\n",
            cadeia="regra ampla proposital: marca A, B e C (2 sao limpos)",
        )

    funcao0 = novo_nada()
    regra0 = regra_nada()
    det_rej = _refinar(col, regra.detector_de(regra0, funcao0),
                       RunnableLambda(fake_ampla), emb, amostras=4)
    c.check(det_rej.funcao is funcao0 and det_rej.codigo == regra0.codigo,
            "guarda (A) rejeita ampla: mantem a resident anterior (mesma funcao viva)")
    passo_rej = det_rej.historico[0]
    c.check(passo_rej["mudou"] is False
            and passo_rej["status"].startswith("rejeitada por precisao"),
            f"historico distingue rejeicao por precisao: {passo_rej['status']}")

    # --- GUARDA (A) ACEITA: fake devolve regra que marca SO' o sujo 'A'
    # (precisao 1.0) -> ACEITA. ---
    def fake_precisa(_prompt_value):
        return RegraDeteccao(
            erro_provavel=True, condicao_regex=None,
            codigo="def detectar(col):\n    return col.eq('A')\n",
            cadeia="marca so' 'A', o unico erro real",
        )

    funcao0b = novo_nada()
    det_ac = _refinar(col, regra.detector_de(regra_nada(), funcao0b),
                      RunnableLambda(fake_precisa), emb, amostras=4)
    f_ac = det_ac.funcao
    c.check(f_ac is not funcao0b
            and oraculo._classificar(f_ac, ["A", "B"]) == [True, False],
            "guarda (A) aceita regra precisa (precisao 1.0): funcao muda, marca so' 'A'")
    c.check(det_ac.historico[0]["status"] == "aplicada",
            f"aceita registra status 'aplicada': {det_ac.historico[0]['status']}")

    # --- PISO (B): resident FINAL ampla + revisao ampla (rejeitada pela guarda)
    # -> ao fim, resident cai para DETECTA_NADA (nao a ampla). ---
    ampla_cod = "def detectar(col):\n    return col.isin(['A', 'B', 'C'])\n"
    funcao0_ampla = _mat_det(ampla_cod)
    regra0_ampla = RegraDeteccao(erro_provavel=True, condicao_regex=None,
                                 codigo=ampla_cod, cadeia="resident ja ampla")
    det_piso = _refinar(col, regra.detector_de(regra0_ampla, funcao0_ampla),
                        RunnableLambda(fake_ampla), emb, amostras=4)
    f_piso = det_piso.funcao
    c.check(oraculo._classificar(f_piso, ["A", "B", "C"]) == [False, False, False],
            "piso (B): resident final ampla derrubada -> funcao marca NADA")
    c.check(det_piso.codigo == deteccao.DETECTA_NADA,
            f"piso (B): detector.codigo espelha DETECTA_NADA: {det_piso.codigo!r}")
    entradas_piso = [p for p in det_piso.historico if p["status"].startswith("piso")]
    c.check(len(entradas_piso) == 1 and "iteracao" in entradas_piso[0],
            "piso (B): 1 entry de piso no historico, com chave 'iteracao' (contrato do relatorio)")
    # O entry do piso tem O MESMO CONJUNTO DE CHAVES das entradas do laco -- senao
    # relatorio._bloco_refinamento quebraria ao vivo.
    chaves_laco = set(det_piso.historico[0].keys())
    c.check(set(entradas_piso[0].keys()) == chaves_laco,
            f"entry de piso tem as mesmas chaves das entradas do laco: {sorted(entradas_piso[0].keys())}")

    # --- OUNCES-LIKE: rotulos TODOS sujos; regra marca TODOS -> precisao 1.0 ->
    # NAO rejeitada, piso NAO dispara (coluna muito-suja nao e' punida). ---
    sujo_oz = pd.Series(["12.0 oz", "16.0 oz", "20.0 oz"])
    limpo_oz = pd.Series(["12", "16", "20"])  # 100% erro real
    col_oz = _coluna_teste("ounces", sujo_oz, limpo_oz)

    def fake_oz(_prompt_value):
        return RegraDeteccao(
            erro_provavel=True, condicao_regex=r"\s*oz$",
            codigo="def detectar(col):\n    return col.str.endswith('oz')\n",
            cadeia="marca o sufixo oz (todos os rotulos sao erro)",
        )

    funcao0_oz = novo_nada()
    det_oz = _refinar(col_oz, regra.detector_de(regra_nada(), funcao0_oz),
                      RunnableLambda(fake_oz), emb, amostras=3)
    f_oz = det_oz.funcao
    c.check(f_oz is not funcao0_oz
            and oraculo._classificar(f_oz, ["12.0 oz"]) == [True],
            "ounces-like: regra que marca ~100% (precisao 1.0) e' ACEITA")
    c.check(det_oz.historico[0]["status"] == "aplicada"
            and not any(p["status"].startswith("piso") for p in det_oz.historico),
            "ounces-like: status 'aplicada' e NENHUM piso disparado")


def teste_prompts_constroem(c: Contador):
    """Regressao: os prompts de deteccao devem construir um ChatPromptTemplate
    REAL sem erro. Pega chave literal nao-escapada (ex.: regex `[A-Z]{2}` que o
    LangChain lia como variavel '2' e crashava TODA run ao vivo -- os testes com
    agente fake nao exercitavam a construcao do template)."""
    secao("12. Prompts de deteccao constroem ChatPromptTemplate (regressao)")
    from langchain_core.prompts import ChatPromptTemplate
    try:
        p1 = ChatPromptTemplate.from_messages(
            [("system", regra.SISTEMA), ("human", regra.HUMANO)])
        p1.format(coluna="city", amostra="x", total_linhas=1, total_distintos=1)
        p2 = ChatPromptTemplate.from_messages(
            [("system", refino.SISTEMA_UPDATE), ("human", refino.HUMANO_UPDATE)])
        p2.format(coluna="abv", codigo_atual="def detectar(col): return col.isin([])",
                  feedback="x")
        ok = True
    except Exception as exc:  # noqa: BLE001
        ok = False
        print("      erro:", str(exc)[:120])
    c.check(ok, "SISTEMA/HUMANO e SISTEMA_UPDATE/HUMANO_UPDATE constroem e formatam")
    render = ChatPromptTemplate.from_messages([("system", regra.SISTEMA)]).format()
    c.check("[A-Z]" + chr(123) + "2" + chr(125) in render,
            "o LLM ve o regex [A-Z]{2} correto (chave dobrada des-escapa)")


def main() -> int:
    c = Contador()
    teste_prompts_constroem(c)
    teste_mi(c)
    teste_metricas_deteccao(c)
    teste_gate_fd(c)
    teste_sandbox_detectar(c)
    teste_construir_mascara(c)
    teste_classificar(c)
    teste_cascata_fd_passa(c)
    teste_gerar_limpador(c)
    teste_fd_duplo_filtro(c)
    teste_coocorrencia_codigo_fd(c)
    teste_ordem_fd_df_entrada(c)
    teste_determinante_sem_detector(c)
    teste_amostrar_oraculo(c)
    teste_montar_feedback(c)
    teste_refinar_deteccao(c)
    teste_selecionar_suspeito(c)
    teste_precisao_no_oraculo(c)
    teste_guarda_e_piso(c)

    secao("Resultado")
    if c.problemas:
        print(f"  {c.problemas} problema(s) nos testes offline do e2e.")
        return 1
    print("  Todos os testes offline do e2e passaram.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
