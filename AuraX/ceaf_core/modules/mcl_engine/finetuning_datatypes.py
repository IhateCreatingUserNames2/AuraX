# ceaf_core/modules/mcl_engine/finetuning_datatypes.py

import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class FinetuningDataPoint(BaseModel):
    """
    Represents a single data point logged for potential ORA fine-tuning,
    focusing on the ORA's self-correction loop involving VRE.
    """
    log_id: str = Field(default_factory=lambda: f"ft_log_{time.time_ns()}")
    timestamp: float = Field(default_factory=time.time)
    session_id: Optional[str] = None
    turn_id: Optional[str] = None # Invocation ID of the ORA turn

    user_query: Optional[str] = None
    ncf_context_summary: Optional[str] = Field(None, description="A summary or key parts of the NCF used by ORA.")

    ora_initial_draft_response: str = Field(..., description="ORA's first attempt at a response, before VRE review.")
    vre_critique_json: Optional[str] = Field(None, description="The JSON output from the VRE assessment of the initial draft.")
    vre_overall_alignment: Optional[str] = Field(None, description="VRE's overall alignment assessment (e.g., 'aligned', 'minor_concerns').")
    vre_recommendations_applied: Optional[List[str]] = Field(None, description="Specific VRE recommendations ORA attempted to apply.")

    ora_refined_response: str = Field(..., description="ORA's final response after considering VRE critique.")

    # Optional: Other contextual data
    mcl_eoc_assessment_at_turn_start: Optional[str] = None # EoC state before this interaction loop
    active_ncf_parameters: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing this data point (e.g., 'ethical_dilemma', 'factual_correction').")

    class Config:
        str_strip_whitespace = True
