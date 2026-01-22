# ceaf_core/modules/memory_blossom/__init__.py
"""Memory Blossom Module"""

# Import key classes to make them available at the module level
from .memory_types import (
    BaseMemory,
    GoalRecord,
    GoalStatus,
    ExplicitMemory,
    ExplicitMemoryContent,
    MemorySourceType,
    MemorySalience,
    KGEntityRecord,
    KGRelationRecord,
    EmotionalMemory,
    ProceduralMemory,
    AnyMemoryType
)

from .memory_lifecycle_manager import (
    initialize_dynamic_salience,
    update_dynamic_salience,
    apply_decay_to_all_memories,
    archive_or_forget_low_salience_memories,
    DEFAULT_ARCHIVE_SALIENCE_THRESHOLD,
    EPHEMERAL_SOURCE_TYPES_FOR_DELETION
)

__all__ = [
    # Memory types
    'BaseMemory',
    'GoalRecord',
    'GoalStatus',
    'ExplicitMemory',
    'ExplicitMemoryContent',
    'MemorySourceType',
    'MemorySalience',
    'KGEntityRecord',
    'KGRelationRecord',
    'EmotionalMemory',
    'ProceduralMemory',
    'AnyMemoryType',
    # Lifecycle functions
    'initialize_dynamic_salience',
    'update_dynamic_salience',
    'apply_decay_to_all_memories',
    'archive_or_forget_low_salience_memories',
    'DEFAULT_ARCHIVE_SALIENCE_THRESHOLD',
    'EPHEMERAL_SOURCE_TYPES_FOR_DELETION'
]