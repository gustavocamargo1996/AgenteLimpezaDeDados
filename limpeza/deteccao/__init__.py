"""Etapa 3: onde estao os erros."""
from .mascara import construir_mascara
from .refino import refinar_regra_deteccao
from .regra import (DETECTA_NADA, construir_agente, detector_nulo,
                    gerar_regra_deteccao)

__all__ = ["DETECTA_NADA", "construir_agente", "detector_nulo",
           "gerar_regra_deteccao", "refinar_regra_deteccao", "construir_mascara"]
