# scheduler_reset_force_simple.py
import asyncio
import os
import logging
from datetime import timedelta

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleSpec,
    ScheduleIntervalSpec,
    ScheduleOverlapPolicy,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ForceReset")

TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")
NEW_TASK_QUEUE = "ceaf-cognitive-queue-v2"

async def main():
    logger.info(f"🔌 Conectando ao Temporal em {TEMPORAL_HOST}...")
    client = await Client.connect(TEMPORAL_HOST)

    # 1. LISTAR E DELETAR TODOS OS AGENDAMENTOS ANTIGOS
    logger.info("🔍 Varrendo agendamentos ativos...")
    try:
        schedule_iterator = await client.list_schedules()
        async for schedule_entry in schedule_iterator:
            sid = schedule_entry.id
            if "dreaming" in sid or "ceaf" in sid:
                logger.warning(f"⚠️ Deletando agendamento: '{sid}'...")
                try:
                    handle = client.get_schedule_handle(sid)
                    await handle.delete()
                except Exception as e:
                    logger.error(f"Erro ao deletar {sid}: {e}")
    except Exception as e:
        logger.warning(f"Não foi possível listar agendamentos: {e}")

    # 2. CRIAR O NOVO AGENDAMENTO (SEM ESPECIFICAR POLICY EXPLICITAMENTE)
    target_id = "ceaf-dreaming-schedule-v4-clean"
    interval_minutes = 10

    logger.info(f"🆕 Criando schedule '{target_id}' na fila '{NEW_TASK_QUEUE}'...")

    try:
        # Opção 1: Sem especificar policy (usa padrões do SDK)
        await client.create_schedule(
            id=target_id,
            schedule=Schedule(
                action=ScheduleActionStartWorkflow(
                    "DreamingWorkflow",
                    args=[],
                    id="dreaming-workflow-job",
                    task_queue=NEW_TASK_QUEUE,
                ),
                spec=ScheduleSpec(
                    intervals=[ScheduleIntervalSpec(every=timedelta(minutes=interval_minutes))]
                ),
                # Não especificamos 'policy' - o SDK usará os padrões
            ),
        )
        logger.info(f"🚀 SUCESSO! Agendamento limpo criado. Próxima execução em {interval_minutes} min.")

    except Exception as e:
        logger.error(f"❌ Falha ao criar agendamento: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())