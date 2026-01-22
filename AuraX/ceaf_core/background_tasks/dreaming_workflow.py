# ceaf_core/background_tasks/dreaming_workflow.py
import asyncio
from datetime import timedelta
from typing import List

from temporalio import workflow
from temporalio.common import RetryPolicy

# Import Activities
with workflow.unsafe.imports_passed_through():
    from ceaf_core.background_tasks.dreaming_activities import (
        fetch_active_agents_activity,
        restore_body_state_activity,
        process_drives_activity,
        latent_consolidation_activity,
        generate_proactive_trigger_activity
    )


@workflow.defn
class DreamingWorkflow:
    """
    The Subconscious Loop.
    Runs periodically to maintain agent homeostasis and memory consolidation.
    """

    @workflow.run
    async def run(self) -> None:
        workflow.logger.info("Starting Dreaming Cycle...")

        retry_policy = RetryPolicy(maximum_attempts=3)

        # 1. Find who needs to dream
        active_agents: List[str] = await workflow.execute_activity(
            fetch_active_agents_activity,
            args=[48],  # Look back 48h
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=retry_policy
        )

        if not active_agents:
            workflow.logger.info("No active agents found. Sleeping.")
            return

        # 2. Process each agent (Parallel execution)
        # In Temporal, we can spawn child workflows or just loop activities.
        # For simplicity, we loop activities asynchronously.

        futures = []
        for agent_id in active_agents:
            futures.append(self._process_single_agent(agent_id))

        await asyncio.gather(*futures)

        workflow.logger.info("Dreaming Cycle Complete.")

    async def _process_single_agent(self, agent_id: str):
        """Helper to orchestrate the dream sequence for one agent."""
        activity_opts = {
            "start_to_close_timeout": timedelta(seconds=60),
            "retry_policy": RetryPolicy(maximum_attempts=3)
        }

        # Biological Maintenance
        await workflow.execute_activity(restore_body_state_activity, args=[agent_id], **activity_opts)
        await workflow.execute_activity(process_drives_activity, args=[agent_id], **activity_opts)

        # Memory Consolidation
        await workflow.execute_activity(latent_consolidation_activity, args=[agent_id], **activity_opts)

        # Proactivity Check
        # If true, this activity would internally trigger a notification dispatch via API
        # or we could return the result and handle dispatch here via another activity.
        await workflow.execute_activity(generate_proactive_trigger_activity, args=[agent_id], **activity_opts)