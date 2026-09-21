"""A espinha: as etapas do gerador, em ordem."""
from datetime import datetime
from pathlib import Path

from . import (amostragem, config, correcao, dados, deteccao, empacotar,
               erros, metricas, relatorio)
from .correcao import fd, regras
from .tipos import Coluna, Trabalho


def gerar_limpador(caminho_sujo, caminho_limpo, colunas=None,
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
    _avisar_limpador_vazio(trabalhos)

    mascara_intra = deteccao.construir_mascara(trabalhos, tabela)
    mascara_fd = deteccao.detectar_dependencia(
        trabalhos, tabela, mascara_intra, agentes["fd"])
    # A etapa nova recebe a mascara ANTERIOR: e' isso que evita a circularidade.
    mascara = ((mascara_intra + mascara_fd) > 0).astype(int)
    corrigido = correcao.rodar_cascata(trabalhos, tabela, mascara, agentes)
    limpador = empacotar.gerar_limpador(
        trabalhos, saida / f"limpador_{tabela.nome}_{carimbo}.py", tabela.nome
    )
    medida = metricas.avaliar(trabalhos, tabela, mascara, corrigido)
    relatorio.escrever(saida, tabela.nome, trabalhos, mascara, corrigido, medida)
    return limpador, medida


def _processar_coluna(coluna: Coluna, agentes: dict) -> Trabalho:
    """Amostra, detecta e refina uma coluna. Aqui mora toda a resiliencia do run."""
    # Refino que quebra devolve a regra de 1-passe (ja validada), nunca deixa a coluna sem regra.
    amostra = amostragem.representantes(coluna)
    print(f"  {coluna.nome}: deteccao a partir de "
          f"{len(amostra.representantes)} representantes sujos...", flush=True)

    try:
        detector = deteccao.gerar_regra_deteccao(coluna, amostra, agentes["deteccao"])
    except Exception as exc:  # noqa: BLE001 -- sem regra: a coluna nao e' marcada
        motivo = erros.descrever(exc)
        print(f"  {coluna.nome}: deteccao falhou ({motivo}) "
              "-> coluna nao marcada", flush=True)
        return Trabalho(coluna=coluna, amostra=amostra,
                        detector=deteccao.detector_nulo(
                            f"deteccao falhou ({motivo}); coluna nao marcada"))

    if config.ITERACOES_DETECCAO > 1 and detector.codigo != deteccao.DETECTA_NADA:
        print(f"  {coluna.nome}: refinando por "
              f"{config.ITERACOES_DETECCAO} iteracoes com oraculo...", flush=True)
        try:
            detector = deteccao.refinar_regra_deteccao(
                detector, coluna, agentes["deteccao"]
            )
            _anunciar_oraculo(coluna, detector)
        except Exception as exc:  # noqa: BLE001 -- refino quebrou: fica o 1-passe
            print(f"  {coluna.nome}: refino falhou ({erros.descrever(exc)}) "
                  "-> mantem a regra de 1-passe", flush=True)
    return Trabalho(coluna=coluna, amostra=amostra, detector=detector)


def _avisar_limpador_vazio(trabalhos: list) -> None:
    """Avisa alto quando nenhuma coluna saiu com regra de deteccao."""
    # Sem isso, LLM fora do ar termina com exit 0 e um limpador vazio: job verde.
    com_regra = [t for t in trabalhos
                 if t.detector is not None
                 and t.detector.codigo != deteccao.DETECTA_NADA]
    if trabalhos and not com_regra:
        print(f"\n  AVISO: 0/{len(trabalhos)} colunas receberam regra de "
              "deteccao; o limpador esta vazio. Verifique PROVEDOR, "
              "MODELO_LLM e o log de falhas acima.\n", flush=True)


def _construir_agentes() -> dict:
    """Constroi os quatro agentes sem fixar modelo: cada papel resolve o seu."""
    # Sem argumento, `None` flui ate llm.modelo_do_papel: MODELO_<PAPEL> > MODELO_LLM.
    return {
        "deteccao": deteccao.construir_agente(),
        "especificador": regras.construir_agente_especificador(),
        "codigo": regras.construir_agente_codigo(),
        "fd": fd.construir_agente(),
    }


def _anunciar(tabela) -> None:
    nomes = [c.nome for c in tabela.colunas]
    print(f"[e2e] dataset={tabela.nome} | {len(tabela.sujo)} linhas "
          f"| colunas={nomes}", flush=True)
    print(f"[e2e] modelo={config.MODELO_LLM} | "
          f"iteracoes_deteccao={config.ITERACOES_DETECCAO} "
          f"amostras_iter={config.AMOSTRAS_POR_ITERACAO}\n", flush=True)


def _anunciar_oraculo(coluna: Coluna, detector) -> None:
    orcamento = detector.orcamento
    if orcamento:
        print(f"  {coluna.nome}: oraculo rotulou "
              f"{orcamento.get('n_valores', 0)} valores / "
              f"{orcamento.get('n_celulas', 0)} celulas", flush=True)
