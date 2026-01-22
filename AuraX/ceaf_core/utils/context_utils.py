# ceaf_core/utils/context_utils.py

import logging
from typing import Optional, Any, cast, TYPE_CHECKING

from google.adk.tools import ToolContext

# --- CORREÇÃO DO IMPORT CIRCULAR (PARTE 1) ---
# Usamos TYPE_CHECKING para que o Python saiba o que é MBSMemoryService
# apenas durante a análise estática (IDE), mas NÃO importe durante a execução.
if TYPE_CHECKING:
    from ceaf_core.services.mbs_memory_service import MBSMemoryService

logger = logging.getLogger(__name__)


def get_mbs_from_context(tool_context: ToolContext) -> Optional['MBSMemoryService']:
    """
    Recupera a instância do MBSMemoryService do ToolContext do ADK.

    Correções aplicadas:
    1. Importação Local para evitar Circular Import.
    2. Remoção de dependência de 'main.py'.
    """
    if tool_context is None:
        logger.error("ContextUtils: tool_context recebido é None!")
        return None

    memory_service_candidate: Any = None
    ic = None  # InvocationContext

    # 1. Tenta extrair do InvocationContext (Estrutura padrão do ADK)
    if hasattr(tool_context, 'invocation_context') and tool_context.invocation_context is not None:
        ic = tool_context.invocation_context

        # Tentativa A: Acesso via runner._services (padrão novo)
        if hasattr(ic, 'runner') and hasattr(ic.runner, '_services') and isinstance(ic.runner._services, dict):
            memory_service_candidate = ic.runner._services.get('memory_service')

        # Tentativa B: Acesso direto no runner (padrão antigo)
        if not memory_service_candidate and hasattr(ic, 'runner') and hasattr(ic.runner, 'memory_service'):
            memory_service_candidate = ic.runner.memory_service

        # Tentativa C: Acesso direto no contexto
        if not memory_service_candidate and hasattr(ic, 'memory_service'):
            memory_service_candidate = ic.memory_service

        # Tentativa D: Acesso via dicionário de services no contexto
        if not memory_service_candidate and hasattr(ic, 'services') and isinstance(ic.services, dict):
            memory_service_candidate = ic.services.get('memory_service')

    # Se não encontrou o candidato, retorna None imediatamente.
    # (Removemos a tentativa de importar de 'main' aqui, pois causava erro)
    if not memory_service_candidate:
        return None

    # --- CORREÇÃO DO IMPORT CIRCULAR (PARTE 2) ---
    # Importamos a classe AQUI dentro. Isso garante que o módulo 'mbs_memory_service'
    # já tenha sido carregado parcialmente antes de tentarmos usá-lo aqui.
    try:
        from ceaf_core.services.mbs_memory_service import MBSMemoryService
        MBS_Class = MBSMemoryService
    except ImportError:
        # Se não conseguir importar (ex: durante testes unitários isolados),
        # definimos como None para cair no fallback de duck-typing
        MBS_Class = None

    # Validação de Tipo
    if MBS_Class is not None and isinstance(memory_service_candidate, MBS_Class):
        return cast('MBSMemoryService', memory_service_candidate)

    # Duck-typing Fallback: Se parece um MBS (tem os métodos certos), aceitamos.
    # Útil se houver problemas de importação ou mocks.
    elif (hasattr(memory_service_candidate, 'search_raw_memories') and
          hasattr(memory_service_candidate, 'add_specific_memory')):
        return cast('MBSMemoryService', memory_service_candidate)

    return None