# ceaf_core/modules/mcl_engine/__init__.py (VERSÃO CORRIGIDA)

"""MCL Engine Module"""

# NÃO importe mais CeafSelfRepresentation daqui.

from .mcl_engine import MCLEngine

# Opcional: você pode importar o self_state_analyzer se ele for usado externamente
from . import self_state_analyzer

__all__ = [
    'self_state_analyzer',
    'MCLEngine'  # Exporte apenas o que este módulo realmente define.
]