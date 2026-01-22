# Memory Types
# ceaf_project/ceaf_core/modules/memory_blossom/memory_types.py
import re
import time
import uuid
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
from pydantic import model_validator


# --- Enums for Categorization ---

class MemorySalience(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"  # For flashbulb-like memories


class MemorySourceType(Enum):
    REASONING_MEMORY = None
    USER_INTERACTION = "user_interaction"
    ORA_RESPONSE = "ora_response"
    TOOL_OUTPUT = "tool_output"
    INTERNAL_REFLECTION = "internal_reflection"  # e.g., from MCL or VRE
    NCF_DIRECTIVE = "ncf_directive"
    EXTERNAL_INGESTION = "external_ingestion"  # For knowledge loaded from outside CEAF
    SYNTHESIZED_SUMMARY = "synthesized_summary"  # Memory created by MemoryBlossom itself
    FOUNDATIONAL_DIALOGUE = "foundational_dialogue"
    FOUNDATIONAL_TRAINING = "foundational_training"


class EmotionalTag(Enum):
    # Basic Plutchik-like emotions, can be expanded
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"
    CONFUSION = "confusion"  # Useful for AI state
    CURIOSITY = "curiosity"
    SATISFACTION = "satisfaction"  # e.g., task completion
    FRUSTRATION = "frustration"  # e.g., repeated errors


# --- Base Memory Model ---

class BaseMemory(BaseModel):
    memory_id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex}")
    timestamp: float = Field(default_factory=time.time)
    last_accessed_ts: Optional[float] = None
    access_count: int = 0

    source_turn_id: Optional[str] = None  # Invocation ID of the turn it originated from
    source_interaction_id: Optional[str] = None  # Broader interaction/session ID
    source_type: MemorySourceType
    source_agent_name: Optional[str] = None  # e.g., "ORA", "MCL_Agent", "user"

    salience: MemorySalience = MemorySalience.MEDIUM
    emotional_tags: List[EmotionalTag] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)  # For keyword-based retrieval

    # For narrative linking
    related_memory_ids: List[str] = Field(default_factory=list)
    narrative_thread_id: Optional[str] = None  # To group memories into a coherent story/topic

    dynamic_salience_score: float = Field(0.5, ge=0.0, le=1.0)  # Calculated, reflects current importance
    decay_rate: float = Field(0.01)  # How quickly it loses salience if not accessed/reinforced

    # Embedding - will be a vector, store as list of floats or a reference
    # For simplicity here, we'll assume it might be stored elsewhere or handled by a vector DB.
    # If storing directly: embedding: Optional[List[float]] = None
    embedding_reference: Optional[str] = None  # e.g., ID in a vector database
    outcome_value: float = Field(0.0, description="Outcome of the experience, from -1.0 (failure) to 1.0 (success).")
    learning_value: float = Field(0.0, ge=0.0, le=1.0, description="How much was learned from this experience.")
    metadata: Dict[str, Any] = Field(default_factory=dict)  # For any other custom data

    def mark_accessed(self):
        self.last_accessed_ts = time.time()
        self.access_count += 1

    class Config:
        use_enum_values = True  # Store enum values as strings




class GoalStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class GoalRecord(BaseMemory): # Goals are a form of memory
    memory_type: Literal["goal_record"] = "goal_record"
    goal_description: str
    parent_goal_id: Optional[str] = None
    sub_goal_ids: List[str] = Field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    priority: int = Field(5, ge=1, le=10) # 1 highest
    motivation_level: float = Field(0.7, ge=0.0, le=1.0) # Internal drive
    origin_narrative_thread_id: Optional[str] = None # Link to narrative that spawned it
    linked_procedural_memory_id: Optional[str] = None # If a procedure exists to achieve it

    # --- Specific Memory Types ---
class ExplicitMemoryContent(BaseModel):
    # Content can be text, structured data, or reference to a file/artifact
    text_content: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None  # e.g., JSON object
    artifact_reference: Optional[str] = None




    @model_validator(mode='after')
    def check_content_present(self) -> 'ExplicitMemoryContent':
         if not self.text_content and not self.structured_data and not self.artifact_reference:
             raise ValueError("At least one of text_content, structured_data, or artifact_reference must be provided.")
         return self


class ExplicitMemory(BaseMemory):
    memory_type: Literal["explicit"] = "explicit"
    content: ExplicitMemoryContent
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)  # Confidence in the accuracy of the memory
    is_core_value: bool = Field(False, description="True if this memory represents a core value or belief.")
    explains_procedure_step: Optional[str] = Field(None,
                                                   description="ID of a ProceduralStep this memory elaborates on.")
    provides_evidence_for_goal: Optional[str] = Field(None, description="ID of a GoalRecord this memory supports.")
    failure_pattern: Optional[str] = Field(None, description="If this memory represents a failure, this is its classification.")


class KGEntityType(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"
    PRODUCT = "product"
    ARTIFACT = "artifact" # e.g., a report, a piece of code
    SYSTEM_COMPONENT = "system_component" # e.g., ORA, MBS, NCF
    USER = "user"
    ISSUE = "issue"
    OTHER = "other"

class KGEntityRecord(BaseMemory):
    """
    Represents a node (entity) in the knowledge graph.
    Stored as a distinct memory type in MemoryBlossom.
    """
    memory_type: Literal["kg_entity_record"] = "kg_entity_record"
    entity_id_str: str = Field(..., description="A unique string identifier for this entity (e.g., 'user_john_doe', 'concept_ai_ethics'). Should be canonicalized.")
    label: str = Field(..., description="Human-readable label for the entity (e.g., 'John Doe', 'AI Ethics').")
    entity_type: KGEntityType = Field(KGEntityType.OTHER, description="The type of the entity.")
    description: Optional[str] = Field(None, description="A brief description of the entity.")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Key-value pairs describing the entity's properties.")
    aliases: List[str] = Field(default_factory=list, description="Alternative names or identifiers for this entity.")

    @model_validator(mode='before')
    @classmethod
    def ensure_entity_id_from_label(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # If entity_id_str is not provided, attempt to create one from label and type
        # This is a simple placeholder; robust canonicalization is complex.
        if not values.get('entity_id_str') and values.get('label'):
            label_slug = re.sub(r'\s+', '_', values['label'].lower())
            label_slug = re.sub(r'\W+', '', label_slug) # Remove non-alphanumeric
            entity_type_val = values.get('entity_type', KGEntityType.OTHER)
            if isinstance(entity_type_val, KGEntityType):
                entity_type_val = entity_type_val.value
            values['entity_id_str'] = f"{entity_type_val}_{label_slug[:50]}" # Ensure not too long
        elif not values.get('entity_id_str') and not values.get('label'):
            raise ValueError("KGEntityRecord requires at least a label to derive an entity_id_str if not provided.")
        return values


class EmotionalContext(BaseModel):
    triggering_event_summary: Optional[str] = None
    associated_stimuli: List[str] = Field(default_factory=list)
    intensity: float = Field(0.5, ge=0.0, le=1.0)  # Normalized intensity


class EmotionalMemory(BaseMemory):
    memory_type: Literal["emotional"] = "emotional"
    primary_emotion: EmotionalTag  # The dominant emotion recorded
    context: EmotionalContext


class ProceduralStep(BaseModel):
    step_number: int
    description: str
    expected_inputs: Optional[List[str]] = None
    expected_outputs: Optional[List[str]] = None
    tool_to_use: Optional[str] = None  # Name of ADK tool
    sub_procedure_id: Optional[str] = None  # Link to another ProceduralMemory


class ProceduralMemory(BaseMemory):
    memory_type: Literal["procedural"] = "procedural"
    procedure_name: str
    goal_description: str
    steps: List[ProceduralStep]
    trigger_conditions: List[str] = Field(default_factory=list)  # Conditions that might activate this procedure


class FlashbulbMemory(BaseMemory):
    memory_type: Literal["flashbulb"] = "flashbulb"
    event_summary: str  # A concise summary of the highly salient event
    vividness_score: float = Field(ge=0.0, le=1.0)
    personal_significance: Optional[str] = None  # Why it was significant to CEAF
    # Flashbulb memories inherently have high salience, so it's set in BaseMemory
    content_details: ExplicitMemoryContent  # Detailed account of the event


class SomaticMarker(BaseModel):
    marker_type: str  # e.g., "positive_anticipation", "negative_avoidance", "error_signal"
    associated_state_pattern: Dict[str, Any]  # Abstract representation of internal state associated with this marker
    intensity: float = Field(ge=0.0, le=1.0)


class SomaticMemory(BaseMemory):
    """
    Represents "gut feelings" or internal state associations with certain situations or outcomes.
    These are learned markers that can quickly guide decision-making without full deliberation.
    """
    memory_type: Literal["somatic"] = "somatic"
    triggering_context_summary: str  # Summary of the context that elicits/elicited this marker
    marker: SomaticMarker


class LiminalThoughtFragment(BaseModel):
    fragment_text: Optional[str] = None
    image_url_reference: Optional[str] = None  # If it's a visual fragment
    conceptual_tags: List[str] = Field(default_factory=list)
    connection_strength_to_conscious: float = Field(ge=0.0, le=1.0)  # How close it is to surfacing


class LiminalMemory(BaseMemory):
    """
    Represents "pre-conscious" fragments, ideas, or associations that are not yet fully formed
    or integrated into explicit thought. The "dream-like" content.
    """
    memory_type: Literal["liminal"] = "liminal"
    fragments: List[LiminalThoughtFragment]
    potential_significance: Optional[str] = None  # Why these fragments might be important


class ReasoningStep(BaseModel):
    step_number: int
    description: str
    reasoning: str # Por que este passo foi tomado

class ReasoningMemory(BaseMemory):
    memory_type: Literal["reasoning"] = "reasoning"
    task_description: str # Um resumo da tarefa ou query que iniciou o raciocínio
    strategy_summary: str # A estratégia de alto nível (vinda do AgencyModule)
    reasoning_steps: List[ReasoningStep] # Os passos tomados (pode ser extraído da deliberação)
    outcome: Literal["success", "failure", "partial_success"]
    outcome_reasoning: str # Por que foi um sucesso ou falha

class GenerativeSeed(BaseModel):
    seed_type: str  # e.g., "prompt_template", "style_guide", "core_concept_map"
    content: Union[str, Dict[str, Any]]  # The actual seed data
    usage_instructions: Optional[str] = None


class GenerativeMemory(BaseMemory):
    """
    Memories that help CEAF generate new content, ideas, or behaviors.
    These are less about recalling past events and more about providing templates or starting points.
    """
    memory_type: Literal["generative"] = "generative"
    seed_name: str
    seed_data: GenerativeSeed
    applicability_contexts: List[str] = Field(default_factory=list)  # When is this seed useful?

class KGRelationRecord(BaseMemory):
    """
    Represents an edge (relationship) in the knowledge graph.
    Stored as a distinct memory type in MemoryBlossom.
    """
    memory_type: Literal["kg_relation_record"] = "kg_relation_record"
    relation_id_str: str = Field(default_factory=lambda: f"rel_{uuid.uuid4().hex}", description="A unique string identifier for this relationship instance.")
    source_entity_id_str: str = Field(..., description="The entity_id_str of the source/subject entity.")
    target_entity_id_str: str = Field(..., description="The entity_id_str of the target/object entity.")
    relation_label: str = Field(..., description="Human-readable label for the relationship (e.g., 'reported', 'affects', 'is_a', 'created_by').")
    description: Optional[str] = Field(None, description="A brief description or context for this relationship.")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Key-value pairs describing properties of the relationship itself (e.g., 'date_reported', 'confidence').")
    directed: bool = Field(True, description="Is the relationship directed from source to target?")


class InteroceptivePredictionMemory(ExplicitMemory):
    """
    Uma memória explícita que captura o resultado de um ciclo de predição interoceptiva.
    Representa a "surpresa" ou o "erro de predição" do sistema sobre seu próprio estado.
    O conteúdo principal é armazenado em structured_data.
    """
    memory_type: Literal["interoceptive_prediction"] = "interoceptive_prediction"

    @model_validator(mode='before')
    @classmethod
    def ensure_content_structure(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Garante que o conteúdo seja formatado corretamente para este tipo de memória."""
        content = values.get('content')
        if not isinstance(content, ExplicitMemoryContent) or not content.structured_data:
            raise ValueError("InteroceptivePredictionMemory deve ser inicializada com structured_data em seu conteúdo.")

        # Define um text_content padrão se não for fornecido, para fins de embedding.
        if not content.text_content:
            error_data = content.structured_data.get("prediction_error_signal", {})
            total_error = error_data.get("total_error", 0.0)
            summary = f"Reflexão sobre auto-predição: Experimentei um erro de predição de {total_error:.2f}."
            content.text_content = summary

        values['content'] = content
        return values


# --- Union Type for all Memory Types ---
# This allows functions to accept any specific memory type.
AnyMemoryType = Union[
    ExplicitMemory,
    EmotionalMemory,
    ProceduralMemory,
    FlashbulbMemory,
    SomaticMemory,
    LiminalMemory,
    GenerativeMemory,
    GoalRecord,
    KGEntityRecord,
    KGRelationRecord,
    InteroceptivePredictionMemory,
    ReasoningMemory,
    BaseMemory
]
