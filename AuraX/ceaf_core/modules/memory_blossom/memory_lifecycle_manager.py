# ceaf_core/modules/memory_blossom/memory_lifecycle_manager.py

import time
import logging
from typing import List, Optional, Tuple, Dict, Any
import math  # For decay functions like exponential

# Assuming memory_types.py is in the same parent directory (modules/memory_blossom/)
try:
    from .memory_types import BaseMemory, MemorySalience, AnyMemoryType, MemorySourceType

    MEMORY_TYPES_LOADED_SUCCESSFULLY = True
except ImportError:
    logging.error("MemoryLifecycleManager: Could not import CEAF memory_types. Using placeholders.")
    MEMORY_TYPES_LOADED_SUCCESSFULLY = False


    # Define minimal placeholders if import fails, to allow type hinting and basic structure
    class BaseMemory:  # type: ignore
        def __init__(self, memory_id: str, timestamp: float, **kwargs):
            self.memory_id = memory_id
            self.timestamp = timestamp
            self.last_accessed_ts: Optional[float] = None
            self.access_count: int = 0
            self.dynamic_salience_score: float = 0.5
            self.decay_rate: float = 0.01
            self.salience: str = "medium"  # Placeholder for MemorySalience enum
            self.source_type: str = "unknown"  # Placeholder for MemorySourceType enum
            self.__dict__.update(kwargs)

        def mark_accessed(self): self.last_accessed_ts = time.time(); self.access_count += 1


    class MemorySalience:
        MEDIUM = "medium";
        LOW = "low";
        HIGH = "high";
        CRITICAL = "critical"  # type: ignore


    class MemorySourceType:
        USER_INTERACTION = "user_interaction";
        ORA_RESPONSE = "ora_response";
        INTERNAL_REFLECTION = "internal_reflection";
        TOOL_OUTPUT = "tool_output";
        SYNTHESIZED_SUMMARY = "synthesized_summary";
        EXTERNAL_INGESTION = "external_ingestion";
        NCF_DIRECTIVE = "ncf_directive"  # type: ignore


    AnyMemoryType = BaseMemory  # type: ignore

logger = logging.getLogger("MemoryLifecycleManager")

# --- Configuration for Lifecycle Management ---
SALIENCE_UPDATE_CONFIG = {
    "access_retrieval_boost": 0.05,  # Boost on simple retrieval
    "access_commit_boost": 0.2,  # Boost when explicitly committed/created
    "positive_reinforcement_base": 0.15,  # Base boost for positive signals
    "negative_reinforcement_base": -0.1,  # Base penalty for negative signals
    "max_salience": 1.0,
    "min_salience": 0.0
}

# How many seconds represent one "unit" for decay. E.g., decay applied per day.
DECAY_TIME_UNIT_SECONDS = 86400  # 1 day
DEFAULT_ARCHIVE_SALIENCE_THRESHOLD = 0.1
EPHEMERAL_SOURCE_TYPES_FOR_DELETION = [
    MemorySourceType.TOOL_OUTPUT,  # Often transient
    # MemorySourceType.SYNTHESIZED_SUMMARY, # Can be regenerated
]


def initialize_dynamic_salience(memory: BaseMemory) -> None:
    """
    Initializes the dynamic_salience_score based on the initial MemorySalience enum.
    This should be called when a memory is first created or loaded.
    """
    if not hasattr(memory, 'dynamic_salience_score'):
        # If the attribute isn't even there, add it.
        # This helps if old memory objects are loaded that don't have it.
        setattr(memory, 'dynamic_salience_score', 0.5)  # Default starting point

    if MEMORY_TYPES_LOADED_SUCCESSFULLY:
        salience_map = {
            MemorySalience.CRITICAL: 0.95,
            MemorySalience.HIGH: 0.8,
            MemorySalience.MEDIUM: 0.6,
            MemorySalience.LOW: 0.3,
        }
        # Handle if memory.salience is string or enum
        initial_salience_value = memory.salience
        if isinstance(memory.salience, MemorySalience):
            initial_salience_value = memory.salience.value  # Get the string value

        memory.dynamic_salience_score = salience_map.get(initial_salience_value, 0.5)  # type: ignore
    else:  # Fallback for placeholder types
        salience_map_str = {
            "critical": 0.95, "high": 0.8, "medium": 0.6, "low": 0.3,
        }
        memory.dynamic_salience_score = salience_map_str.get(str(memory.salience).lower(), 0.5)
    logger.debug(
        f"Memory {memory.memory_id}: Initialized dynamic_salience_score to {memory.dynamic_salience_score:.2f} from static salience '{memory.salience}'.")


def update_dynamic_salience(
        memory: BaseMemory,
        access_type: str = "retrieval",  # "retrieval", "commit", "positive_feedback", "negative_feedback"
        reinforcement_signal: Optional[float] = None,  # Value between -1.0 and 1.0 for feedback
        config: Dict[str, Any] = SALIENCE_UPDATE_CONFIG
) -> None:
    """
    Updates the dynamic salience score of a memory.
    """
    if not hasattr(memory, 'dynamic_salience_score'):
        initialize_dynamic_salience(memory)  # Ensure it's initialized

    current_salience = memory.dynamic_salience_score
    change = 0.0

    if access_type == "retrieval":
        change = config.get("access_retrieval_boost", 0.05)
        memory.mark_accessed()  # Also update access count and timestamp
    elif access_type == "commit":
        change = config.get("access_commit_boost", 0.2)
        memory.mark_accessed()
    elif access_type == "positive_feedback" and reinforcement_signal is not None:
        # reinforcement_signal should be 0 to 1 for positive
        change = config.get("positive_reinforcement_base", 0.15) * max(0, min(1, reinforcement_signal))
        memory.mark_accessed()
    elif access_type == "negative_feedback" and reinforcement_signal is not None:
        # reinforcement_signal should be 0 to 1 for magnitude of negative feedback (penalty = base * signal)
        change = config.get("negative_reinforcement_base", -0.1) * max(0, min(1, reinforcement_signal))
        memory.mark_accessed()
    else:
        logger.warning(
            f"Memory {memory.memory_id}: Unknown access_type '{access_type}' or missing reinforcement_signal for salience update.")

    new_salience = current_salience + change
    memory.dynamic_salience_score = max(config.get("min_salience", 0.0),
                                        min(config.get("max_salience", 1.0), new_salience))

    logger.debug(
        f"Memory {memory.memory_id}: Salience updated from {current_salience:.2f} to {memory.dynamic_salience_score:.2f} (Change: {change:.2f}, Type: {access_type}).")


def apply_decay_to_memory(memory: BaseMemory, current_time: Optional[float] = None) -> None:
    """Applies decay to a single memory's dynamic_salience_score."""
    if not hasattr(memory, 'dynamic_salience_score') or not hasattr(memory, 'decay_rate'):
        logger.warning(f"Memory {memory.memory_id} missing dynamic_salience_score or decay_rate. Skipping decay.")
        return

    if not current_time:
        current_time = time.time()

    last_event_time = memory.last_accessed_ts or memory.timestamp
    time_since_last_event = current_time - last_event_time

    if time_since_last_event <= 0:
        return  # No decay if accessed now or in the future

    # Simple linear decay for now, could be exponential
    # decay_units = time_since_last_event / DECAY_TIME_UNIT_SECONDS
    # decay_amount = memory.decay_rate * decay_units

    # Exponential decay: S_new = S_old * e^(-lambda * t)
    # where lambda is related to decay_rate and t is time in units.
    # For simplicity, let's use a slightly different exponential model: S_new = S_old * (1 - decay_rate_per_unit)^decay_units
    decay_units = time_since_last_event / DECAY_TIME_UNIT_SECONDS
    decay_factor_per_unit = (1.0 - memory.decay_rate)  # Assumes decay_rate is per DECAY_TIME_UNIT_SECONDS

    # To prevent issues with very small decay_factor_per_unit or large decay_units leading to 0
    if decay_factor_per_unit <= 0:
        new_salience = SALIENCE_UPDATE_CONFIG.get("min_salience", 0.0)
    else:
        try:
            # Ensure the base for pow is not negative if decay_units is fractional
            effective_decay_factor = math.pow(max(0, decay_factor_per_unit), decay_units)
            new_salience = memory.dynamic_salience_score * effective_decay_factor
        except ValueError:  # math domain error
            logger.warning(f"Memory {memory.memory_id}: Math error during decay calculation. Clamping salience.")
            new_salience = SALIENCE_UPDATE_CONFIG.get("min_salience", 0.0)

    old_salience = memory.dynamic_salience_score
    memory.dynamic_salience_score = max(SALIENCE_UPDATE_CONFIG.get("min_salience", 0.0), new_salience)

    if old_salience != memory.dynamic_salience_score:
        logger.debug(
            f"Memory {memory.memory_id}: Decayed salience from {old_salience:.3f} to {memory.dynamic_salience_score:.3f} "
            f"(time_delta: {time_since_last_event:.0f}s, decay_units: {decay_units:.2f})."
        )


def apply_decay_to_all_memories(memory_store: List[AnyMemoryType]) -> None:
    """
    Periodically called. Reduces dynamic_salience_score based on decay_rate and last_accessed_ts.
    Modifies memories in-place.
    """
    logger.info(f"Applying decay to {len(memory_store)} memories.")
    current_time = time.time()
    for memory in memory_store:
        apply_decay_to_memory(memory, current_time)
    logger.info("Decay application complete.")


def archive_or_forget_low_salience_memories(
        memory_store: List[AnyMemoryType],
        archive_threshold: float = DEFAULT_ARCHIVE_SALIENCE_THRESHOLD,
        ephemeral_sources: Optional[List[Any]] = None  # List of MemorySourceType values or strings
) -> Tuple[List[AnyMemoryType], List[AnyMemoryType], List[AnyMemoryType]]:
    """
    Identifies memories below a dynamic_salience_score threshold.
    Returns three lists: memories_to_keep, memories_to_archive, memories_to_forget.
    Does NOT modify the input memory_store directly, caller handles that.
    """
    if ephemeral_sources is None:
        ephemeral_sources = EPHEMERAL_SOURCE_TYPES_FOR_DELETION

    memories_to_keep: List[AnyMemoryType] = []
    memories_to_archive: List[AnyMemoryType] = []
    memories_to_forget: List[AnyMemoryType] = []

    logger.info(f"Checking {len(memory_store)} memories for archiving/forgetting (threshold: {archive_threshold}).")

    for memory in memory_store:
        if not hasattr(memory, 'dynamic_salience_score'):
            logger.warning(f"Memory {memory.memory_id} missing dynamic_salience_score. Keeping it by default.")
            memories_to_keep.append(memory)
            continue

        if memory.dynamic_salience_score < archive_threshold:
            # Check if memory.source_type is in ephemeral_sources
            # Handle both enum and string representations for source_type
            source_type_val = memory.source_type
            if MEMORY_TYPES_LOADED_SUCCESSFULLY and isinstance(memory.source_type, MemorySourceType):
                source_type_val = memory.source_type.value  # type: ignore

            is_ephemeral = False
            for ephemeral_type in ephemeral_sources:
                ephemeral_type_val = ephemeral_type
                if MEMORY_TYPES_LOADED_SUCCESSFULLY and isinstance(ephemeral_type, MemorySourceType):
                    ephemeral_type_val = ephemeral_type.value  # type: ignore

                if source_type_val == ephemeral_type_val:
                    is_ephemeral = True
                    break

            if is_ephemeral:
                memories_to_forget.append(memory)
                logger.debug(
                    f"Memory {memory.memory_id} (Salience: {memory.dynamic_salience_score:.2f}, Source: {source_type_val}) marked for FORGETTING.")
            else:
                memories_to_archive.append(memory)
                logger.debug(
                    f"Memory {memory.memory_id} (Salience: {memory.dynamic_salience_score:.2f}, Source: {source_type_val}) marked for ARCHIVING.")
        else:
            memories_to_keep.append(memory)

    logger.info(
        f"Memory lifecycle: Keep: {len(memories_to_keep)}, Archive: {len(memories_to_archive)}, Forget: {len(memories_to_forget)}."
    )
    return memories_to_keep, memories_to_archive, memories_to_forget


MEMORY_LIFECYCLE_MANAGER_LOADED = MEMORY_TYPES_LOADED_SUCCESSFULLY
