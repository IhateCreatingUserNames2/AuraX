# ceaf_core/workflows.py
from datetime import timedelta
from typing import Dict, Any, List

from temporalio import workflow
from temporalio.common import RetryPolicy

# Import Activities
with workflow.unsafe.imports_passed_through():
    from ceaf_core.activities import (
        perception_activity,
        investigation_activity,
        hormonization_activity,
        agency_activity,
        synthesis_activity,
        evolution_activity,
        logging_activity
    )


@workflow.defn
class CognitiveCycleWorkflow:
    @workflow.run
    async def run(
            self,
            agent_id: str,
            session_id: str,
            query: str,
            chat_history: List[Dict[str, str]],
            persistence_path: str,
            user_id: str = "default_user"
    ) -> Dict[str, Any]:

        state_dict = {
            "agent_id": agent_id,
            "session_id": session_id,
            "user_id": user_id,
            "identity_glyph": [],  # Would be fetched in a real scenario
            "metadata": {"chat_history": chat_history}
        }

        # Activity Options
        retry_policy = RetryPolicy(maximum_attempts=3)
        activity_opts = {
            "start_to_close_timeout": timedelta(seconds=600),
            "retry_policy": retry_policy
        }
        long_activity_opts = {
            "start_to_close_timeout": timedelta(seconds=2500),
            "retry_policy": retry_policy
        }

        # --- Step 1: Perception ---
        perception_result = await workflow.execute_activity(
            perception_activity,
            args=[state_dict, query],
            **activity_opts
        )
        intent_data = perception_result["intent_packet"]
        xi = perception_result["xi"]

        if "identity_glyph" in perception_result:
            state_dict["identity_glyph"] = perception_result["identity_glyph"]

        # --- Step 2: Investigation ---
        investigation_result = await workflow.execute_activity(
            investigation_activity,
            args=[state_dict, intent_data],
            **long_activity_opts  # RLM can take time
        )
        memory_context = investigation_result["memory_context"]

        structured_memories_list = investigation_result.get("structured_memories", [])

        # --- Step 3: Hormonization ---
        hormonal_result = await workflow.execute_activity(
            hormonization_activity,
            args=[state_dict, xi],
            **activity_opts
        )

        # --- Step 4: Agency ---
        agency_result = await workflow.execute_activity(
            agency_activity,
            args=[state_dict, intent_data, hormonal_result, memory_context],
            **long_activity_opts  # Deliberation takes time
        )
        strategy_data = agency_result["strategy"]

        # --- Step 5: Synthesis ---
        response_text = await workflow.execute_activity(
            synthesis_activity,
            args=[
                state_dict,
                intent_data,
                strategy_data,
                hormonal_result,
                memory_context,
                structured_memories_list
            ],
            **long_activity_opts
        )

        # --- Step 6: Evolution (Fire-and-forget logic, but awaited in workflow) ---

        await workflow.execute_activity(
            logging_activity,
            args=[
                state_dict,
                intent_data,
                response_text,
                hormonal_result,
                strategy_data,
                persistence_path
            ],
            **activity_opts
        )

        return {
            "response": response_text,
            "xi": xi,
            "agent_id": agent_id,
            "session_id": session_id
        }