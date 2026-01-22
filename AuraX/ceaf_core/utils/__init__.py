# ceaf_core/utils/__init__.py

"""
Torna o diretório 'utils' um pacote Python, expondo suas funções e classes mais importantes
para facilitar a importação em outros módulos do sistema CEAF.

Este arquivo age como o índice do pacote, permitindo importações mais limpas e centralizadas.
"""

# De common_utils.py
from .common_utils import (
    sanitize_text_for_logging,
    extract_json_from_text,
    pydantic_to_json_str,
    parse_llm_json_output,
    create_successful_tool_response,
    create_error_tool_response,
)

# De embedding_utils.py
from .embedding_utils import (
    get_embedding_client,
    compute_adaptive_similarity,
    EmbeddingClient,
    cosine_similarity_np,
)

# De observability_types.py (NOVO)
from .observability_types import (
    ObservabilityManager,
    ObservationType,
    Observation,
)

# De context_utils.py (EXISTENTE)
from .context_utils import get_mbs_from_context

# De adk_helpers.py (EXISTENTE)
from .adk_helpers import configure_adk_warnings

# A variável __all__ define a API pública do pacote 'utils'.
# Isso especifica quais nomes são importados quando se usa 'from ceaf_core.utils import *'.
__all__ = [
    # common_utils
    "sanitize_text_for_logging",
    "extract_json_from_text",
    "pydantic_to_json_str",
    "parse_llm_json_output",
    "create_successful_tool_response",
    "create_error_tool_response",

    # embedding_utils
    "get_embedding_client",
    "compute_adaptive_similarity",
    "EmbeddingClient",
    "cosine_similarity_np",

    # observability_types
    "ObservabilityManager",
    "ObservationType",
    "Observation",

    # context_utils
    "get_mbs_from_context",

    # adk_helpers
    "configure_adk_warnings",
]