# ceaf_core/system.py
import asyncio
import logging
import os
import uuid
import time
from datetime import timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

from temporalio.client import Client

logger = logging.getLogger("AuraV4_System_Client")

# Configuration
TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")
TASK_QUEUE = "ceaf-cognitive-queue-v2"


class CEAFSystem:
    """
    CEAF System V4 (Distributed Client).

    This class serves as the entry point for the API to interact with the
    distributed cognitive engine. It does not run logic locally anymore.
    Instead, it orchestrates the 'CognitiveCycleWorkflow' via Temporal.io.
    """

    def __init__(self, config: Dict[str, Any], agent_manager=None, db_repo=None):
        self.config = config
        self.agent_id = config.get("agent_id", "default_agent")
        # Persistence path is kept for compatibility, though storage is now handled by microservices
        self.persistence_path = Path(config.get("persistence_path", f"./agent_data/{self.agent_id}"))

        self.temporal_client: Optional[Client] = None
        self._client_lock = asyncio.Lock()

    async def _ensure_client(self):
        """Lazy initialization of the Temporal Client."""
        if not self.temporal_client:
            async with self._client_lock:
                if not self.temporal_client:
                    try:
                        logger.info(f"Connecting to Temporal Server at {TEMPORAL_HOST}...")
                        self.temporal_client = await Client.connect(TEMPORAL_HOST)
                        logger.info("✅ Connected to Temporal.")
                    except Exception as e:
                        logger.critical(f"❌ Failed to connect to Temporal: {e}")
                        raise RuntimeError("Cognitive Engine Unavailable") from e

    async def process(self, query: str, session_id: str, **kwargs) -> Dict[str, Any]:
        """
        Inits the Cognitive Cycle.

        Args:
            query: The user's input text.
            session_id: The active session identifier.
            **kwargs: Additional context, primarily 'chat_history'.

        Returns:
            A dictionary containing the agent's response and metadata,
            formatted to match the expected API output schema.
        """
        await self._ensure_client()

        path_str = str(self.persistence_path.resolve())
        chat_history = kwargs.get("chat_history", [])

        # Unique Workflow ID to prevent duplication of the same turn
        # We use a hash of the query + timestamp to ensure uniqueness per turn attempt
        workflow_id = f"turn-{session_id}-{int(time.time())}"

        logger.info(f"🚀 Triggering Cognitive Workflow: {workflow_id} for Agent {self.agent_id}")

        try:
            # Execute the workflow and await the result
            # The logic inside 'CognitiveCycleWorkflow' will run on a separate Worker node.
            user_id = self.config.get("user_id", "default_user")

            result = await self.temporal_client.execute_workflow(
                "CognitiveCycleWorkflow",
                args=[
                    self.agent_id,
                    session_id,
                    query,
                    chat_history,
                    path_str,
                    user_id
                ],
                id=workflow_id,
                task_queue=TASK_QUEUE,
                execution_timeout=timedelta(seconds=18000)
            )

            logger.info(f"✅ Workflow {workflow_id} completed successfully.")

            # Construct the response packet expected by the API routes
            response_payload = {
                "response": result["response"],
                "xi": result.get("xi", 0.0),
                "organism_state": "ACTIVE",  # Placeholder, real detailed state is in Redis
                "chat_history": chat_history + [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": result["response"]}
                ],
                "status": "success"
            }

            return response_payload

        except Exception as e:
            logger.error(f"❌ Cognitive Workflow Failed: {e}", exc_info=True)

            # Fallback response so the API doesn't crash
            return {
                "response": "I encountered an internal error while processing your request. Please try again later.",
                "xi": 1.0,  # High tension due to error
                "organism_state": "ERROR",
                "status": "error",
                "chat_history": chat_history
            }

    # --- Utility Methods (Legacy Compatibility) ---

    # These methods might still be called by legacy parts of the API (e.g., proactive triggers).
    # In a full migration, these should also move to Workflows or direct Service calls.
    # For Phase 3, we keep them as stubs or direct calls to the new services if needed.

    async def _ensure_self_model(self):
        """Deprecated: Self model logic is now handled inside the Synthesis Activity."""
        from ceaf_core.models import CeafSelfRepresentation
        return CeafSelfRepresentation()

    async def reload_cognitive_profile(self, profile):
        """Hot-reload logic. In distributed systems, this requires a Signal to the Worker."""
        # TODO: Implement Temporal Signal to update worker cache
        logger.info("Reload profile signal received (Not yet implemented for distributed workers).")