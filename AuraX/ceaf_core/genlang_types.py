# ceaf_core/genlang_types.py
import time
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field

# --- Biological States (Persisted in Redis) ---

class VirtualBodyState(BaseModel):
    cognitive_fatigue: float = Field(0.0, ge=0.0, le=1.0)
    information_saturation: float = Field(0.0, ge=0.0, le=1.0)
    last_updated: float = Field(default_factory=time.time)
    phenomenological_report: str = "Feeling balanced and alert."

class DriveState(BaseModel):
    intensity: float = Field(0.5, ge=0.0, le=1.0)
    texture: Optional[str] = None
    conflict: Optional[str] = None
    momentum: float = Field(0.0)
    effectiveness_score: float = Field(0.5, ge=0.0, le=2.0)

class MotivationalDrives(BaseModel):
    curiosity: DriveState = Field(default_factory=DriveState)
    mastery: DriveState = Field(default_factory=DriveState)
    connection: DriveState = Field(default_factory=DriveState)
    consistency: DriveState = Field(default_factory=DriveState)
    last_updated: float = Field(default_factory=time.time)

# --- Cognitive States (Ephemeral per Turn) ---

class TurnMetrics(BaseModel):
    turn_id: str
    agency_score: float = 0.0
    used_mycelial_path: bool = False
    vre_rejection_count: int = 0
    vre_flags: List[str] = Field(default_factory=list)
    final_confidence: float = 0.0
    relevant_memories_count: int = 0
    new_memories_created: int = 0  # Added for EmbodimentModule

class UserRepresentation(BaseModel):
    emotional_state: str = "neutral"
    communication_style: str = "neutral"
    known_preferences: List[str] = Field(default_factory=list)
    knowledge_level: str = "unknown"
    last_update_reason: str = "Initial model."

class InternalStateReport(BaseModel):
    cognitive_strain: float = 0.0
    cognitive_flow: float = 0.0
    epistemic_discomfort: float = 0.0
    ethical_tension: float = 0.0
    social_resonance: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class GenlangVector(BaseModel):
    vector: List[float]
    source_text: Optional[str] = None
    model_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IntentPacket(BaseModel):
    query_vector: GenlangVector
    intent_vector: Optional[GenlangVector] = None
    emotional_valence_vector: Optional[GenlangVector] = None
    entity_vectors: List[GenlangVector] = Field(default_factory=list)
    metadata: Dict[str, Any]

class GuidancePacket(BaseModel):
    coherence_vector: GenlangVector
    novelty_vector: GenlangVector
    goal_alignment_vector: Optional[GenlangVector] = None
    safety_avoidance_vector: Optional[GenlangVector] = None

class CommonGroundTracker(BaseModel):
    agent_statement_counts: Dict[str, int] = Field(default_factory=dict)
    user_acknowledged_topics: List[str] = Field(default_factory=list)

    def record_agent_statement(self, statement_type: str):
        self.agent_statement_counts[statement_type] = self.agent_statement_counts.get(statement_type, 0) + 1

    def check_statement_count(self, statement_type: str) -> int:
        return self.agent_statement_counts.get(statement_type, 0)

    def is_becoming_repetitive(self, statement_type: str, threshold: int = 2) -> bool:
        return self.check_statement_count(statement_type) >= threshold

class ToolOutputPacket(BaseModel):
    tool_name: str
    status: Literal["success", "error"]
    summary_vector: Optional[GenlangVector] = None
    raw_output: Optional[str] = None

class CognitiveStatePacket(BaseModel):
    original_intent: IntentPacket
    identity_vector: GenlangVector
    relevant_memory_vectors: List[GenlangVector] = Field(default_factory=list)
    common_ground: CommonGroundTracker = Field(default_factory=CommonGroundTracker)
    guidance_packet: Optional[GuidancePacket] = None  # Optional as it's filled later
    deliberation_history: List[str] = Field(default_factory=list)
    tool_outputs: List[ToolOutputPacket] = Field(default_factory=list)
    ethical_assessment_summary: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ResponsePacket(BaseModel):
    content_summary: str
    response_emotional_tone: str = "neutral"
    confidence_score: float = 0.8
    supporting_memory_snippets: List[str] = Field(default_factory=list)
    ethical_assessment_summary: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Helper types for refinement (can remain basic)
class AdjustmentVector(BaseModel):
    vector: GenlangVector
    description: str
    weight: float = 0.5

class RefinementPacket(BaseModel):
    adjustment_vectors: List[AdjustmentVector] = Field(default_factory=list)
    textual_recommendations: List[str] = Field(default_factory=list)