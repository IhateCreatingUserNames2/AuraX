# worker.py
import os
from dotenv import load_dotenv

load_dotenv()
import asyncio
import logging
from temporalio.client import Client
from temporalio.worker import Worker

# Import Workflows
from ceaf_core.workflows import CognitiveCycleWorkflow
from ceaf_core.background_tasks.dreaming_workflow import DreamingWorkflow  # New

# Import Activities
from ceaf_core.activities import (
    perception_activity,
    investigation_activity,
    hormonization_activity,
    agency_activity,
    synthesis_activity,
    evolution_activity
)
from ceaf_core.background_tasks.dreaming_activities import (
    fetch_active_agents_activity,
    restore_body_state_activity,
    process_drives_activity,
    latent_consolidation_activity,
    generate_proactive_trigger_activity
)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TemporalWorker")

TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")


async def main():
    logger.info(f"Connecting to Temporal Server at {TEMPORAL_HOST}...")
    client = await Client.connect(TEMPORAL_HOST)

    # Initialize the worker
    worker = Worker(
        client,
        task_queue="ceaf-cognitive-queue",
        workflows=[
            CognitiveCycleWorkflow,
            DreamingWorkflow  # Register the Dreamer
        ],
        activities=[
            # Cognitive Cycle
            perception_activity,
            investigation_activity,
            hormonization_activity,
            agency_activity,
            synthesis_activity,
            evolution_activity,

            # Dreaming / Subconscious
            fetch_active_agents_activity,
            restore_body_state_activity,
            process_drives_activity,
            latent_consolidation_activity,
            generate_proactive_trigger_activity
        ],
    )

    logger.info("Worker started. Listening on 'ceaf-cognitive-queue'...")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())