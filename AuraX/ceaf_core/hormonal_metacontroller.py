# ceaf_core/hormonal_metacontroller.py
import logging
from typing import Dict, Any, List

from ceaf_core.genlang_types import MotivationalDrives
from ceaf_core.services.state_manager import StateManager

logger = logging.getLogger("AuraV4_Endocrine")


class HormonalMetacontroller:
    """
    PASSO 3: Endocrine System (Redis-Backed).
    Translates Epistemic Tension (xi) into behavioral 'dosages' and updates persistent hormonal baselines.
    """

    def __init__(self):
        self.state_manager = StateManager()

    async def process_hormonal_response(self, agent_id: str, xi: float) -> Dict[str, Any]:
        """
        Calculates steering based on Epistemic Tension (xi) and current biological state.
        Updates persistent hormonal baselines in Redis.
        """
        # 1. Fetch current biological context
        drives = await self.state_manager.get_drives(agent_id)
        current_hormones = await self.state_manager.get_hormonal_baseline(agent_id)

        # 2. Calculate Response
        steering_result = self._calculate_dosage(xi, drives, current_hormones)

        # 3. Update Baselines (e.g., High tension increases baseline cortisol)
        new_cortisol = current_hormones.get("cortisol", 0.0)

        if xi > 0.6:
            # Acute stress increases cortisol baseline slightly
            new_cortisol = min(1.0, new_cortisol + 0.05)
        else:
            # Recovery
            new_cortisol = max(0.0, new_cortisol - 0.01)

        await self.state_manager.update_hormonal_baseline(agent_id, {"cortisol": new_cortisol})

        return steering_result

    def _calculate_dosage(self, xi: float, drives: MotivationalDrives, hormones: Dict[str, float]) -> Dict[str, str]:
        """
        Pure logic for determining steering injection.
        """
        result = {
            "hormonal_injection": "[STEERING: NEUTRAL]",
            "state_label": "STABLE"
        }

        cortisol_level = hormones.get("cortisol", 0.0)

        # Logic: Cortisol (Stress/Rigidity)
        # Combined effect of immediate tension (xi) and chronic stress (cortisol)
        effective_stress = xi + (cortisol_level * 0.5)

        if effective_stress > 0.6:
            result["state_label"] = "STRESSED"
            result["hormonal_injection"] = "[STEERING: CAUTION - MODE: RIGID_TRUTH]"

        # Logic: Dopamine (Flow/Curiosity)
        elif effective_stress < 0.3 and drives.curiosity.intensity > 0.7:
            result["state_label"] = "FLOW"
            result["hormonal_injection"] = "[STEERING: CURIOSITY - MODE: EXPLORATIVE]"

        # Logic: Oxytocin (Social Resonance)
        if drives.connection.intensity > 0.8:
            result["hormonal_injection"] += "\n[STEERING: EMPATHY - MODE: WARM]"

        return result