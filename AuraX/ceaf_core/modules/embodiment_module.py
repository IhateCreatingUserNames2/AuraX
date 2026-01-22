# ceaf_core/modules/embodiment_module.py
import time
import logging
from typing import Dict, Any

from ceaf_core.genlang_types import VirtualBodyState
from ceaf_core.models import BodyConfig
from ceaf_core.services.state_manager import StateManager

logger = logging.getLogger("CEAFv3_Embodiment")


class EmbodimentModule:
    """
    Manages the "virtual body" of the agent via Redis.
    Calculates fatigue, saturation, and recovery based on activity metrics.
    """

    def __init__(self, config: BodyConfig = None):
        self.config = config or BodyConfig()
        self.state_manager = StateManager()

    async def process_turn_effects(self, agent_id: str, metrics: Dict[str, Any]) -> VirtualBodyState:
        """
        Fetches current state from Redis, applies biological logic, and saves back.
        Returns the updated state for the current turn context.
        """
        # 1. Fetch State (Nervous System)
        current_state = await self.state_manager.get_body_state(agent_id)

        # 2. Apply Biological Logic
        updated_state = self._calculate_state_update(current_state, metrics)

        # 3. Persist State
        await self.state_manager.save_body_state(agent_id, updated_state)

        return updated_state

    def _calculate_state_update(self, body_state: VirtualBodyState, metrics: Dict[str, Any]) -> VirtualBodyState:
        """Pure logic calculation for state update."""
        updated_state = body_state.model_copy(deep=True)

        # --- 1. Cognitive Fatigue (Strain) ---
        strain = metrics.get("cognitive_strain", 0.0)
        fatigue_increase = strain * self.config.fatigue_accumulation_multiplier
        updated_state.cognitive_fatigue += fatigue_increase

        if fatigue_increase > 0.01:
            logger.debug(f"Embodiment: Strain {strain:.2f} -> Fatigue +{fatigue_increase:.2f}")

        # --- 2. Information Saturation (New Memories) ---
        new_memories = metrics.get("new_memories_created", 0)
        saturation_increase = new_memories * self.config.saturation_accumulation_per_memory
        updated_state.information_saturation += saturation_increase

        # --- 3. Passive Recovery (Decay) ---
        current_time = time.time()
        time_delta_seconds = current_time - updated_state.last_updated
        time_delta_hours = time_delta_seconds / 3600.0

        # Apply recovery only if time has passed
        if time_delta_seconds > 0:
            fatigue_recovery = self.config.fatigue_recovery_rate * time_delta_hours
            updated_state.cognitive_fatigue -= fatigue_recovery

            saturation_recovery = self.config.saturation_recovery_rate * time_delta_hours
            updated_state.information_saturation -= saturation_recovery

        # --- 4. Normalization & Alerts ---
        updated_state.cognitive_fatigue = max(0.0, min(1.0, updated_state.cognitive_fatigue))
        updated_state.information_saturation = max(0.0, min(1.0, updated_state.information_saturation))
        updated_state.last_updated = current_time

        if updated_state.cognitive_fatigue > self.config.fatigue_warning_threshold:
            logger.warning(f"⚠️ EMBODIMENT ALERT: Critical Cognitive Fatigue ({updated_state.cognitive_fatigue:.2f})")

        return updated_state