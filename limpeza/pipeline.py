"""A espinha: as etapas do gerador, em ordem."""
from datetime import datetime
from pathlib import Path

from . import (amostragem, config, correcao, dados, deteccao, empacotar,
               metricas, relatorio)
from .correcao import fd, regras
from .tipos import Coluna, Trabalho


def gerar_limpador(caminho_sujo=None, caminho_limpo=None, colunas=None,
                   saida=None) -> tuple[Path, dict]:
    """Gera um limpador.py a partir do sujo e do orcamento de rotulos."""
    tabela = dados.carregar(caminho_sujo, caminho_limpo, colunas)
    agentes = _construir_agentes()
    carimbo = datetime.now().strftime("%Y-%m-%d_%H%M")
    saida = Path(saida) if saida else relatorio.criar_pasta("e2e", carimbo)
    saida.mkdir(parents=True, exist_ok=True)
    _anunciar(tabela)

    trabalhos = []
    for coluna in tabela.colunas:
        trabalhos.append(_processar_coluna(coluna, agentes))

    mascara = deteccao.construir_mascara(trabalhos, tabela)
    corrigido = correcao.rodar_cascata(trabalhos, tabela, mascara, agentes)
    limpador = empacotar.gerar_limpador(
        trabalhos, saida / f"limpador_{config.DATASET}_{carimbo}.py"
    )
    medida = metricas.avaliar(trabalhos, tabela, mascara, corrigido)
    relatorio.escrever(saida, trabalhos, mascara, corrigido, medida)
    return limpador, medida


def _processar_coluna(coluna: Coluna, agentes: dict) -> Trabalho:
    """Amostra, detecta e refina uma coluna. Falha vira detector nulo."""
    # Unica fronteira de resiliencia do run: uma coluna que quebra sai sem
    # marcacao, e as demais seguem.
    amostra = amostragem.representantes(coluna)
    print(f"  {coluna.nome}: deteccao a partir de "
          f"{len(amostra.representantes)} representantes sujos...", flush=True)
    try:
        detector = deteccao.gerar_regra_deteccao(coluna, amostra, agentes["deteccao"])
        if detector.codigo != deteccao.DETECTA_NADA:
            print(f"  {coluna.nome}: refinando por "
                  f"{config.ITERACOES_DETECCAO} iteracoes com oraculo...", flush=True)
            detector = deteccao.refinar_regra_deteccao(
                detector, coluna, agentes["deteccao"]
            )
            _anunciar_oraculo(coluna, detector)
    except Exception as exc:  # noqa: BLE001 -- 1 coluna ruim nao derruba o run
        print(f"  {coluna.nome}: deteccao falhou ({type(exc).__name__}) "
              "-> coluna nao marcada", flush=True)
        detector = deteccao.detector_nulo(
            f"deteccao falhou ({type(exc).__name__}); coluna nao marcada"
        )
    return Trabalho(coluna=coluna, amostra=amostra, detector=detector)


def _construir_agentes() -> dict:
    """Um agente por papel, todos no modelo configurado."""
    modelo = config.MODELO_LLM
    return {
        "deteccao": deteccao.construir_agente(modelo),
        "especificador": regras.construir_agente_especificador(modelo),
        "codigo": regras.construir_agente_codigo(modelo),
        "fd": fd.construir_agente(modelo),
    }


def _anunciar(tabela) -> None:
    """Uma linha com o que este run vai processar."""
    nomes = [c.nome for c in tabela.colunas]
    marca = f" sufixo={config.SUFIXO}" if config.SUFIXO else ""
    print(f"[e2e] dataset={config.DATASET}{marca} | {len(tabela.sujo)} linhas "
          f"| colunas={nomes}", flush=True)
    print(f"[e2e] modelo={config.MODELO_LLM} | "
          f"iteracoes_deteccao={config.ITERACOES_DETECCAO} "
          f"amostras_iter={config.AMOSTRAS_POR_ITERACAO}\n", flush=True)


def _anunciar_oraculo(coluna: Coluna, detector) -> None:
    """O custo do refinamento da coluna, em valores e celulas rotuladas."""
    orcamento = detector.orcamento
    if orcamento:
        print(f"  {coluna.nome}: oraculo rotulou "
              f"{orcamento.get('n_valores', 0)} valores / "
              f"{orcamento.get('n_celulas', 0)} celulas", flush=True)
