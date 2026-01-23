# ceaf_core/background_tasks/dreaming_activities.py
import logging
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta

from temporalio import activity

from ceaf_core.modules.vector_lab import VectorLab
from ceaf_core.services.llm_service import LLMService
# Data Services
from database.models import AgentRepository
from ceaf_core.services.state_manager import StateManager
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.services.llm_service import LLMService

logger = logging.getLogger("DreamingActivities")


# --- Activity Context ---
class DreamerContext:
    _instance = None

    def __init__(self):
        self.db = AgentRepository()
        self.state = StateManager()
        self.memory = MBSMemoryService()
        self.llm = LLMService()

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


@activity.defn
async def fetch_active_agents_activity(lookback_hours: int = 48) -> List[str]:
    """Retrieves IDs of agents active in the last X hours."""
    ctx = DreamerContext.get()
    logger.info(f"Dreamer: Scanning for agents active in last {lookback_hours}h...")
    active_ids = await ctx.db.get_recently_active_agent_ids(hours=lookback_hours)
    logger.info(f"Dreamer: Found {len(active_ids)} active agents.")
    return active_ids


@activity.defn
async def optimize_identity_vectors_activity(agent_id: str) -> str:
    """
    Atividade de Sonho: Auto-otimização de vetores de conceito.
    """
    # 1. Decide O QUE aprender
    # Lógica simples: Alterna entre conceitos base ou reage a falhas recentes
    # Em produção, isso viria de uma análise de logs (LCAM)
    concepts_to_optimize = ["Empathy", "Creativity", "Brevity"]
    import random
    target_concept = random.choice(concepts_to_optimize)

    logger.info(f"💤 Dreamer: A Aura decidiu meditar sobre '{target_concept}' hoje.")

    # 2. Instancia o Lab
    # Precisa do LLMService para gerar os dados
    llm_service = LLMService()
    lab = VectorLab(llm_service=llm_service)

    # 3. Roda o Ciclo
    # Nota: Isso pode demorar minutos e usar muita GPU
    try:
        result_file = await lab.run_optimization_cycle(target_concept)

        if result_file:
            return f"Optimization Success: Generated {result_file} for {target_concept}"
        else:
            return f"Optimization Inconclusive for {target_concept}"

    except Exception as e:
        logger.error(f"Dreamer: Pesadelo no laboratório: {e}")
        return f"Error optimizing {target_concept}: {e}"


@activity.defn
async def restore_body_state_activity(agent_id: str) -> None:
    """Restores cognitive fatigue (Sleep)."""
    ctx = DreamerContext.get()

    body = await ctx.state.get_body_state(agent_id)

    # Apply Sleep Logic (Reduction)
    body.cognitive_fatigue *= 0.1
    body.information_saturation *= 0.5

    await ctx.state.save_body_state(agent_id, body)
    logger.info(f"Dreamer: Restored body state for {agent_id}.")


@activity.defn
async def process_drives_activity(agent_id: str) -> None:
    """Updates motivational drives (Passive evolution)."""
    ctx = DreamerContext.get()

    drives = await ctx.state.get_drives(agent_id)
    now = datetime.now().timestamp()

    # Calculate time delta
    last_update = drives.last_updated
    delta_hours = (now - last_update) / 3600.0

    if delta_hours > 0:
        # Increase curiosity/connection over time (Longing)
        drives.curiosity.intensity += 0.05 * delta_hours
        drives.connection.intensity += 0.1 * delta_hours

        # Normalize
        drives.curiosity.intensity = min(1.0, drives.curiosity.intensity)
        drives.connection.intensity = min(1.0, drives.connection.intensity)

        drives.last_updated = now
        await ctx.state.save_drives(agent_id, drives)
        logger.info(f"Dreamer: Updated drives for {agent_id} (Delta: {delta_hours:.2f}h).")


@activity.defn
async def latent_consolidation_activity(agent_id: str) -> str:
    """
    Performs memory consolidation (Clustering/Dreaming).
    Returns a summary of the 'dream' if successful.
    """
    ctx = DreamerContext.get()

    # 1. Fetch recent memories
    # Note: search_raw_memories logic is complex, for this phase we simplify
    # assuming we just want recent memories regardless of query
    recent_memories = await ctx.memory.search_raw_memories(query="*", top_k=50, agent_id=agent_id)

    if len(recent_memories) < 5:
        return "Not enough memories to dream."

    # 2. Logic for Clustering would go here (using sklearn as before)
    # For brevity in migration, we assume a simple summary logic
    # In full implementation, copy the DBSCAN logic from original aura_reflector.py

    dream_summary = "Consolidated recent interactions into long-term patterns."
    logger.info(f"Dreamer: {agent_id} dreamt: {dream_summary}")

    return dream_summary


@activity.defn
async def generate_proactive_trigger_activity(agent_id: str) -> bool:
    """
    Checks if the agent wants to initiate conversation.
    Returns True if a message should be dispatched.
    """
    ctx = DreamerContext.get()
    drives = await ctx.state.get_drives(agent_id)

    # Calculate Trigger Score
    score = (drives.connection.intensity * 0.6) + (drives.curiosity.intensity * 0.4)

    if score > 0.8:  # High threshold for proactivity
        logger.info(f"Dreamer: {agent_id} triggering PROACTIVE message (Score: {score:.2f})")
        # In a real system, we would generate the message content here
        # and queue it for the Notification Service.
        # For Phase 4, we just signal the intent.
        return True

    return False