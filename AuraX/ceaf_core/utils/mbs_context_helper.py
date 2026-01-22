import logging
from typing import Any, Optional, cast

from google.adk.tools import ToolContext

# Attempt to import the real MBSMemoryService class.
# If this import fails, the application's main.py lifespan function should
# catch the ImportError and prevent startup in a degraded state.
from ceaf_core.services.mbs_memory_service import MBSMemoryService

logger = logging.getLogger(__name__)

def get_mbs_from_context(tool_context: ToolContext) -> Optional['MBSMemoryService']:
    """
    Retrieves the MBSMemoryService instance from the ADK ToolContext.

    This function consolidates logic to find the MBS instance, which can be located
    in various places within the ADK's `ToolContext` or via global fallbacks.
    It performs type checking against the `MBSMemoryService` class.

    Args:
        tool_context: The Google ADK ToolContext object provided to a tool function.

    Returns:
        An instance of `MBSMemoryService` if found and it matches the expected interface,
        otherwise `None`.
    """
    logger.debug(f"Attempting to retrieve MBS from tool_context (type: {type(tool_context)})")
    if tool_context is None:
        logger.error("get_mbs_from_context received tool_context as None!")
        return None

    memory_service_candidate: Any = None
    ic = None

    # Try to get invocation_context first
    if hasattr(tool_context, 'invocation_context') and tool_context.invocation_context is not None:
        ic = tool_context.invocation_context
        logger.debug(f"Found tool_context.invocation_context: {type(ic)}")
    else:
        logger.warning("ToolContext has no 'invocation_context' or it is None. Trying global fallback early.")
        # Fallback directly to main's adk_components if invocation_context is missing or None.
        # This is a critical fallback for scenarios where the context isn't fully formed.
        try:
            # We import main here to avoid a circular dependency at the top level
            # if main also imports tools that use this helper.
            # This is generally acceptable for a utility that needs access to the main app's components.
            from ceaf_project.main import adk_components as main_adk_components_module_level
            memory_service_candidate = main_adk_components_module_level.get('memory_service')
            if memory_service_candidate:
                logger.info("Retrieved memory_service_candidate from main.adk_components (early fallback).")
                # Perform basic duck-typing or type check immediately for the fallback.
                if isinstance(memory_service_candidate, MBSMemoryService):
                    return cast(MBSMemoryService, memory_service_candidate)
                elif (hasattr(memory_service_candidate, 'search_raw_memories') and
                      hasattr(memory_service_candidate, 'add_specific_memory')):
                    logger.warning("Using duck-typing fallback for MBSMemoryService in early global retrieval.")
                    return cast(MBSMemoryService, memory_service_candidate)
                logger.error(f"Early global fallback found memory_service_candidate (type: {type(memory_service_candidate)}) but it doesn't match expected MBSMemoryService interface.")
                return None
        except ImportError:
            logger.debug("main.adk_components not found or importable for early global fallback.")
            pass # Continue to other checks if early fallback fails

    # If invocation_context was found, try its internal attributes
    if ic:
        # Attempt 1: Direct access from invocation_context if explicitly set (common in testing/mocking)
        if hasattr(ic, 'memory_service'):
            memory_service_candidate = ic.memory_service
            if memory_service_candidate:
                logger.debug("Found memory_service_candidate via ic.memory_service (direct).")

        # Attempt 2: Standard ADK runner services access (common in production runtime)
        if not memory_service_candidate and hasattr(ic, 'runner'):
            if hasattr(ic.runner, '_services'): # Newer ADK versions might store services in _services dict
                memory_service_candidate = ic.runner._services.get('memory_service')
                if memory_service_candidate:
                    logger.debug("Found memory_service_candidate via ic.runner._services.")
            if not memory_service_candidate and hasattr(ic.runner, 'memory_service'): # Older ADK or direct attribute
                memory_service_candidate = ic.runner.memory_service
                if memory_service_candidate:
                    logger.debug("Found memory_service_candidate via ic.runner.memory_service (direct on runner).")

        # Attempt 3: Check if services are stored differently on invocation_context (less common)
        if not memory_service_candidate and hasattr(ic, 'services') and isinstance(ic.services, dict):
            memory_service_candidate = ic.services.get('memory_service')
            if memory_service_candidate:
                logger.debug("Found memory_service_candidate via ic.services (dict).")

    # Final fallback: If still no candidate, try accessing main's adk_components again as a last resort
    # (This path is less ideal as it implies the context wasn't properly passed, but provides robustness)
    if not memory_service_candidate:
        logger.warning("Could not find memory_service_candidate through common context paths. Trying global main.adk_components fallback (last resort).")
        try:
            from main import adk_components as main_adk_components_module_level # type: ignore
            memory_service_candidate = main_adk_components_module_level.get('memory_service')
            if memory_service_candidate:
                logger.info("Retrieved memory_service_candidate from main.adk_components (last resort fallback).")
        except ImportError:
            logger.debug("main.adk_components not found or importable for last resort fallback.")
            pass # Fall through if import fails

    if not memory_service_candidate:
        logger.error("MBSMemoryService instance completely not found in any context path or fallback.")
        return None

    # Final type validation before returning the candidate
    # This checks if the candidate object behaves like MBSMemoryService, either by explicit type
    # or by duck-typing (checking for key methods).
    if isinstance(memory_service_candidate, MBSMemoryService):
        logger.info("Validated memory service instance successfully against MBSMemoryService type.")
        return cast(MBSMemoryService, memory_service_candidate)
    elif (hasattr(memory_service_candidate, 'search_raw_memories') and
            hasattr(memory_service_candidate, 'add_specific_memory') and
            hasattr(memory_service_candidate, 'get_memory_by_id')): # Add other critical methods for duck typing if needed
        logger.warning(
            "Using DUCK-TYPING for MBSMemoryService as strict isinstance check failed. This typically means the application is running in a potentially degraded state or the environment is not fully set up.")
        return cast(MBSMemoryService, memory_service_candidate)
    else:
        logger.error(f"Found memory_service_candidate (type: {type(memory_service_candidate)}) "
                     f"but it doesn't match expected MBSMemoryService interface via strict type or duck-typing. "
                     f"Candidate has search_raw_memories: {hasattr(memory_service_candidate, 'search_raw_memories')}, "
                     f"add_specific_memory: {hasattr(memory_service_candidate, 'add_specific_memory')}, "
                     f"get_memory_by_id: {hasattr(memory_service_candidate, 'get_memory_by_id')}.")
        return None

