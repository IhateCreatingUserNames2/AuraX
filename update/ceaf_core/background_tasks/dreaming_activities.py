import logging
import asyncio
import random
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from temporalio import activity

from ceaf_core.modules.vector_lab import VectorLab
from ceaf_core.services.llm_service import LLMService
from ceaf_core.utils.common_utils import extract_json_from_text
from database.models import AgentRepository
from ceaf_core.services.state_manager import StateManager
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.modules.data_extractor import TrainingDataExtractor
from ceaf_core.modules.dream_trainer import DreamMachine
from pathlib import Path


logger = logging.getLogger("DreamingActivities")


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


# --- [NOVA LÓGICA] DIAGNÓSTICO ESTRATÉGICO ---
async def _diagnose_strategic_concept(ctx: DreamerContext, agent_id: str) -> Optional[str]:
    """
    Analisa memórias de falha (LCAM) para determinar qual conceito
    resolveria os problemas mais frequentes do agente.
    """
    logger.info(f"🕵️ Dreamer ({agent_id}): Analisando falhas recentes para diagnóstico...")

    # 1. Buscar memórias de erro/falha no MBS
    # O LCAM marca memórias com keywords específicas
    query_text = "failure prediction_error lcam_lesson mistake negative_feedback"

    try:
        # Busca as top 10 falhas mais relevantes/recentes
        failed_memories = await ctx.memory.search_raw_memories(
            query=query_text,
            top_k=10,
            agent_id=agent_id,
            min_score=0.4  # Só queremos falhas relevantes
        )
    except Exception as e:
        logger.error(f"Erro ao buscar memórias de falha: {e}")
        return None

    if not failed_memories:
        logger.info(f"✅ Nenhuma falha significativa encontrada para {agent_id}. Sistema saudável.")
        return None

    # 2. Preparar dossiê para o LLM
    failure_log = []
    for mem_obj, score in failed_memories:
        # Extrai o texto da memória (assumindo que MBS retorna objetos Memory)
        content = getattr(mem_obj.content, 'text_content', str(mem_obj))
        failure_log.append(f"- {content[:300]}")  # Trunca para economizar tokens

    failures_text = "\n".join(failure_log)

    # 3. Consultar o LLM "Psicólogo"
    prompt = f"""
    Você é um Engenheiro de Cognição de IA (Dreamer Module).
    Analise o seguinte log de falhas operacionais recentes de um agente:

    --- LOG DE FALHAS ---
    {failures_text}
    --- FIM DO LOG ---

    **SUA TAREFA:**
    Identifique **UMA ÚNICA** característica comportamental, traço de personalidade ou conceito abstrato (Steering Vector) que, se injetado no modelo, preveniria essas falhas.

    Exemplos:
    - Se o erro foi "resposta muito longa/prolixa" -> Conceito: "Extreme_Conciseness"
    - Se o erro foi "frieza com o usuário" -> Conceito: "Warm_Empathy"
    - Se o erro foi "alucinação de fatos" -> Conceito: "Strict_Factuality"
    - Se o erro foi "falta de criatividade" -> Conceito: "Divergent_Thinking"

    Responda APENAS com um JSON:
    {{
        "concept_name": "Nome_Do_Conceito_Em_Ingles",
        "reasoning": "Breve motivo do diagnóstico."
    }}
    """

    try:
        # Usa o modelo 'smart' para diagnóstico preciso
        response = await ctx.llm.ainvoke(ctx.llm.config.smart_model, prompt, temperature=0.4)
        data = extract_json_from_text(response)

        if data and "concept_name" in data:
            concept = data["concept_name"].replace(" ", "_")  # Normaliza
            reason = data.get("reasoning", "Diagnosis logic")
            logger.critical(f"💊 DIAGNÓSTICO: O agente precisa de '{concept}'. Motivo: {reason}")
            return concept

    except Exception as e:
        logger.error(f"Erro no diagnóstico LLM: {e}")

    return None


@activity.defn
async def train_neural_physics_activity(agent_id: str, persistence_path: str) -> str:
    """
    Atividade de Sonho: Treinamento da Física Neural (Forward/Inverse Models).
    Lê o histórico do SQLite e atualiza os pesos da rede neural.
    """
    logger.info(f"🧠 Dreamer ({agent_id}): Iniciando Treino Neural (Física Cognitiva)...")

    # 1. Caminho do Banco de Dados
    db_path = Path(persistence_path) / "cognitive_turn_history.sqlite"
    if not db_path.exists():
        return "Skipped: No history database found."

    # 2. Extração de Dados (State, Action, Next_State)
    extractor = TrainingDataExtractor(str(db_path))
    training_data = await extractor.extract_vectors()

    if not training_data:
        return "Skipped: No valid training triplets found."

    # 3. Treinamento (PyTorch)
    trainer = DreamMachine()  # Usa defaults (384 dim)
    trainer.load_brains()  # Carrega estado anterior

    # Executa o treino (síncrono, pois PyTorch ocupa a thread, mas Activity roda em thread separada no Worker)
    result = trainer.train_cycle(training_data, epochs=30)

    return result


@activity.defn
async def optimize_identity_vectors_activity(agent_id: str) -> str:
    """
    Atividade de Sonho: Otimização de Identidade Vetorial.
    Baseada em evidências reais de falha (LCAM) ou rotação de manutenção.
    Atualiza o Mapa Endócrino do agente com novos reflexos comportamentais.
    """
    ctx = DreamerContext.get()

    # 1. Tenta diagnosticar uma necessidade real baseada em falhas passadas
    target_concept = await _diagnose_strategic_concept(ctx, agent_id)

    # 2. Se não houver falhas críticas, usa rotação de manutenção (Gym Mental)
    is_maintenance = False
    if not target_concept:
        is_maintenance = True
        concepts_library = [
            "Extreme_Brevity",
            "High_Empathy",
            "Socratic_Questioning",
            "Creative_Chaos",
            "Absolute_Honesty",
            "Stoic_Calmness"
        ]
        target_concept = random.choice(concepts_library)
        logger.info(f"💤 Dreamer: Modo Manutenção. Reforçando traço base: '{target_concept}'.")

    # Adiciona sufixo de versão para evolução histórica
    version_suffix = datetime.now().strftime("%m%d")
    unique_concept_name = f"{target_concept}_{version_suffix}"

    # Escolha da camada (Random Search segura entre camadas intermediárias/altas)
    target_layer = random.randint(14, 24)

    # 3. Instancia o Laboratório
    lab = VectorLab(llm_service=ctx.llm)

    # 4. Executa o Ciclo de Aprendizado (Geração de Dados + Calibração Remota)
    try:
        result_message = await lab.run_optimization_cycle(
            concept_name=unique_concept_name,
            target_layer=target_layer
        )

        # 5. Se o vetor foi criado com sucesso, integramos ao sistema
        if "SUCESSO" in result_message:

            # A. Registro de Memória (Self-Documentation)
            # Se for um diagnóstico de falha, registramos o "tratamento" no MBS
            if not is_maintenance:
                from ceaf_core.modules.memory_blossom.memory_types import ExplicitMemory, ExplicitMemoryContent, \
                    MemorySourceType, MemorySalience

                treatment_mem = ExplicitMemory(
                    content=ExplicitMemoryContent(
                        text_content=f"Dreamer System Update: Generated steering vector '{unique_concept_name}' to address detected operational failures."
                    ),
                    source_type=MemorySourceType.INTERNAL_REFLECTION,
                    salience=MemorySalience.HIGH,
                    keywords=["system_update", "dreamer", "vector_steering", "self_correction"],
                    agent_id=agent_id
                )
                await ctx.memory.add_specific_memory(treatment_mem, agent_id=agent_id)

            # B. Atualização do Mapa Endócrino (Evolving Reflexes)
            # Classifica o novo vetor para associá-lo a um hormônio
            hormone_type = "dopamine"  # Default
            concept_lower = unique_concept_name.lower()

            if any(x in concept_lower for x in ["calm", "stoic", "defense", "honesty", "brevity"]):
                hormone_type = "cortisol"  # Resposta de defesa/estabilidade
            elif any(x in concept_lower for x in ["empathy", "love", "care", "connection"]):
                hormone_type = "oxytocin"  # Resposta social
            elif any(x in concept_lower for x in ["curiosity", "question", "creative", "chaos"]):
                hormone_type = "dopamine"  # Resposta exploratória

            # Atualiza o Mapa no Redis para uso imediato (Hot-Steering)
            await ctx.state.update_endocrine_link(agent_id, hormone_type, unique_concept_name)

            logger.info(
                f"🧬 Reflexos Evoluídos: Resposta de {hormone_type.upper()} atualizada para usar '{unique_concept_name}'")

        logger.info(f"💤 Dreamer Concluído: {result_message}")
        return result_message

    except Exception as e:
        logger.error(f"🔥 Erro crítico no sonho vetorial: {e}", exc_info=True)
        return f"Error: {str(e)}"

# --- OUTRAS ATIVIDADES (Mantidas para integridade do arquivo) ---

@activity.defn
async def fetch_active_agents_activity(lookback_hours: int = 48) -> List[str]:
    ctx = DreamerContext.get()
    logger.info(f"Dreamer: Scanning for agents active in last {lookback_hours}h...")
    active_ids = await ctx.db.get_recently_active_agent_ids(hours=lookback_hours)
    return active_ids


@activity.defn
async def restore_body_state_activity(agent_id: str) -> None:
    ctx = DreamerContext.get()
    body = await ctx.state.get_body_state(agent_id)
    body.cognitive_fatigue *= 0.1
    body.information_saturation *= 0.5
    await ctx.state.save_body_state(agent_id, body)
    logger.info(f"Dreamer: Restored body state for {agent_id}.")


@activity.defn
async def process_drives_activity(agent_id: str) -> None:
    ctx = DreamerContext.get()
    drives = await ctx.state.get_drives(agent_id)
    now = datetime.now().timestamp()
    delta_hours = (now - drives.last_updated) / 3600.0
    if delta_hours > 0:
        drives.curiosity.intensity = min(1.0, drives.curiosity.intensity + (0.05 * delta_hours))
        drives.connection.intensity = min(1.0, drives.connection.intensity + (0.1 * delta_hours))
        drives.last_updated = now
        await ctx.state.save_drives(agent_id, drives)


@activity.defn
async def latent_consolidation_activity(agent_id: str) -> str:
    ctx = DreamerContext.get()
    # Em um sistema real, aqui chamaria o AuraReflector para fazer o clustering
    return f"Consolidation placeholder for {agent_id}"


@activity.defn
async def generate_proactive_trigger_activity(agent_id: str) -> bool:
    ctx = DreamerContext.get()
    drives = await ctx.state.get_drives(agent_id)
    score = (drives.connection.intensity * 0.6) + (drives.curiosity.intensity * 0.4)
    if score > 0.8:
        logger.info(f"Dreamer: {agent_id} triggering PROACTIVE message (Score: {score:.2f})")
        return True
    return False