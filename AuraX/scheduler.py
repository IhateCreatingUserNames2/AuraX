# scheduler.py
import asyncio
import os
from temporalio.client import Client, Schedule, ScheduleActionStartWorkflow, ScheduleSpec, ScheduleIntervalSpec

TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")


async def main():
    client = await Client.connect(TEMPORAL_HOST)

    # Define the schedule (Every 10 minutes)
    schedule_id = "ceaf-dreaming-schedule"

    try:
        # Create or Update Schedule
        await client.create_schedule(
            id=schedule_id,
            schedule=Schedule(
                action=ScheduleActionStartWorkflow(
                    "DreamingWorkflow",
                    args=[],
                    id="dreaming-workflow-job",
                    task_queue="ceaf-cognitive-queue",
                ),
                spec=ScheduleSpec(
                    intervals=[ScheduleIntervalSpec(every=600)]  # 600s = 10m
                ),
            ),
        )
        print(f"✅ Schedule '{schedule_id}' created successfully.")
    except Exception as e:
        print(f"ℹ️ Schedule might already exist or failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())