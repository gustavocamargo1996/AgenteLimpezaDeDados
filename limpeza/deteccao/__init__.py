"""Etapa 3: onde estao os erros."""
from .mascara import aplicar_detectores, construir_mascara
from .refino import refinar_regra_deteccao
from .regra import (DETECTA_NADA, construir_agente, detector_nulo,
                    gerar_regra_deteccao, materializar)

__all__ = ["DETECTA_NADA", "aplicar_detectores", "construir_agente",
           "construir_mascara", "detector_nulo", "gerar_regra_deteccao",
           "materializar", "refinar_regra_deteccao"]
