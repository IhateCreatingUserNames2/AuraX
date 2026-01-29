# ceaf_core/activities.py
import logging
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List

from ceaf_core.services.cognitive_log_service import CognitiveLogService
from ceaf_core.services.user_profiling_service import UserProfilingService
from pathlib import Path
import numpy as np
from temporalio import activity

from ceaf_core.services.state_manager import StateManager
# Import Core Logic Modules
from ceaf_core.translators.human_to_genlang import HumanToGenlangTranslator
from ceaf_core.rlm_investigator import RLMInvestigator
from ceaf_core.hormonal_metacontroller import HormonalMetacontroller
from ceaf_core.agency_module import AgencyModule
from ceaf_core.translators.genlang_to_human import GenlangToHumanTranslator
from ceaf_core.birag_validator import BiRAGValidator
from ceaf_core.v4_sensors import AuraMonitor
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.services.llm_service import LLMService
from ceaf_core.modules.ncim_engine.ncim_module import NCIMModule
from ceaf_core.modules.vre_engine.vre_engine import VREEngineV3
from ceaf_core.modules.mcl_engine.mcl_engine import MCLEngine
from ceaf_core.modules.lcam_module import LCAMModule
from ceaf_core.modules.embodiment_module import EmbodimentModule
from ceaf_core.modules.geometric_brain import GeometricBrain
from ceaf_core.modules.ncim_engine.ncim_module import SELF_MODEL_MEMORY_ID


# Import Types
from ceaf_core.genlang_types import (
    CognitiveStatePacket, IntentPacket, ResponsePacket,
    GenlangVector, GuidancePacket
)
from ceaf_core.monadic_base import AuraState
from ceaf_core.models import SystemPrompts, MCLConfig, BodyConfig, LLMConfig

logger = logging.getLogger("CEAF_Activities")


# --- Service Initialization Helper ---
# In a production worker, these might be initialized once per worker process or dependency injected.
# For simplicity here, we initialize them lazily or assume singleton behavior where appropriate.

class ActivityContext:
    _instance = None

    def __init__(self):
        self.llm_service = LLMService()
        self.memory_service = MBSMemoryService()
        self.htg = HumanToGenlangTranslator(llm_config=self.llm_service.config)
        self.monitor = AuraMonitor()
        self.rlm = RLMInvestigator()  # API Key from env
        self.hormonal = HormonalMetacontroller()
        self.vre = VREEngineV3()
        self.lcam = LCAMModule(self.memory_service)
        self.mcl = MCLEngine(
            config={},
            agent_config={},
            lcam_module=self.lcam,
            llm_service=self.llm_service
        )
        self.agency = AgencyModule(
            self.llm_service,
            self.vre,
            self.mcl,
            available_tools_summary=""
        )
        self.gth = GenlangToHumanTranslator(llm_service=self.llm_service)
        self.birag = BiRAGValidator(self.llm_service)
        self.ncim = NCIMModule(
            self.llm_service,
            self.memory_service,
            persistence_path=None  # Handled via MBS
        )
        self.geometric_brain = GeometricBrain()
        self.state_manager = StateManager()
        self.embodiment = EmbodimentModule()
        self.user_profiler = UserProfilingService(self.memory_service)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# --- 1. Perception (HTG) ---

@activity.defn
async def perception_activity(state_dict: Dict, query: str) -> Dict:
    ctx = ActivityContext.get()
    agent_id = state_dict["agent_id"]
    user_id = state_dict.get("user_id", "default_user")

    # Imports necessários para a correção
    from ceaf_core.modules.ncim_engine.ncim_module import SELF_MODEL_MEMORY_ID
    from ceaf_core.models import CeafSelfRepresentation
    from ceaf_core.modules.memory_blossom.memory_types import ExplicitMemory, ExplicitMemoryContent, MemorySourceType, \
        MemorySalience

    # --- CORREÇÃO: Carregar ou Gerar Identidade (Glifo) ---
    identity_glyph = []

    try:
        # 1. Tenta buscar a memória do Self Model no MBS
        self_mem = await ctx.memory_service.get_memory_by_id(SELF_MODEL_MEMORY_ID)

        if self_mem and hasattr(self_mem, 'embedding') and self_mem.embedding:
            # Cenário Ideal: Memória existe e já tem vetor
            identity_glyph = self_mem.embedding

        elif self_mem:
            # Memória existe mas vetor não veio no objeto (dependendo da implementação do MBS)
            # Tenta pegar do cache ou regenerar
            text = getattr(self_mem.content, 'text_content', "Agente Aura")
            identity_glyph = await ctx.llm_service.embedding_client.get_embedding(text, context_type="kg_entity_record")

        else:
            # --- SELF-HEALING (A Cura) ---
            # A memória não existe. Vamos criá-la agora baseada no Agent Profile (fallback).
            logger.warning(
                f"⚠️ Identidade ({SELF_MODEL_MEMORY_ID}) não encontrada no MBS. Iniciando Bootstrapping de Identidade...")

            # Cria um Self Model padrão (ou tente carregar do disco se possível, aqui usamos padrão)
            # Em produção, você poderia ler o agent_config.json aqui, mas vamos fazer um boot rápido.
            bootstrapped_model = CeafSelfRepresentation(
                perceived_capabilities=["processamento cognitivo", "memória vetorial", "raciocínio ético"],
                known_limitations=["conhecimento limitado ao treino", "janela de contexto"],
                persona_attributes={"tone": "analítico e prestativo", "style": "claro"}
            )

            # Gera o texto da identidade
            identity_text = f"Valores: {bootstrapped_model.dynamic_values_summary_for_turn}. Persona: {bootstrapped_model.persona_attributes}."

            # Gera o vetor
            identity_glyph = await ctx.llm_service.embedding_client.get_embedding(identity_text,
                                                                                  context_type="kg_entity_record")

            # Salva no MBS para que na próxima vez exista!
            logger.info("💾 Salvando Identidade Bootstrapped no MBS...")
            new_mem = ExplicitMemory(
                memory_id=SELF_MODEL_MEMORY_ID,
                content=ExplicitMemoryContent(
                    text_content=identity_text,
                    structured_data=bootstrapped_model.model_dump()
                ),
                memory_type="explicit",
                source_type=MemorySourceType.INTERNAL_REFLECTION,
                salience=MemorySalience.CRITICAL,
                keywords=["self-model", "identity", "bootstrap"],
                embedding=identity_glyph  # Passa o vetor já calculado se seu MBS suportar, senão ele recalcula
            )
            await ctx.memory_service.add_specific_memory(new_mem, agent_id=agent_id)

    except Exception as e:
        logger.error(f"🔥 Falha Crítica no Identity System: {e}", exc_info=True)
        # Último recurso para não quebrar o pipeline, mas o Dreamer vai ignorar isso
        identity_glyph = [0.001] * 384  # Usamos 0.001 em vez de 0.0 para tentar passar filtros matemáticos simples

        # 1. Recuperar Estado Atual (WM e Expectativa)
    wm = await ctx.state_manager.get_working_memory(agent_id)
    # Supondo que o último vetor da WM seja a "expectativa" do momento anterior
    last_vector = wm[0]['vector'] if wm else None

    # 2. Calcular Surpresa (Active Inference) - Se tiver implementado o WorldModel
    # surprise = await ctx.world_model.calculate_surprise(query, last_vector)
    surprise = 0.5  # Placeholder se não criar o world_model.py agora

    # 3. Geometric Gating (O Cérebro Matemático)
    # Extrai apenas os vetores para o cálculo
    wm_vectors = [np.array(item['vector']) for item in wm]

    # O gating retorna a Tensão (xi) e a Ação (ACCEPT/REINFORCE)
    # IMPORTANTE: Desempacota 4 valores (incluindo o target_idx que pode ser None)
    action, xi, new_vec_np, target_idx = ctx.geometric_brain.compute_gating(query, wm_vectors)
    new_vec = new_vec_np.tolist()

    # 4. Modulação de Energia baseada na Surpresa
    # Se a surpresa é alta, a energia inicial é maior (Flashbulb effect)
    base_energy = 1.0 + (surprise * 2.0)

    # 5. Atualizar Memória de Trabalho (Redis)
    if action == "ACCEPT":
        # Nova memória entra na WM
        wm.append({
            'text': query,
            'vector': new_vec,
            'energy': base_energy,
            'timestamp': datetime.now().timestamp()
        })
    elif action == "REINFORCE" and target_idx is not None:
        # Memória existente é reforçada (Reconsolidação)
        wm[target_idx]['energy'] += base_energy
        # Atualiza o timestamp para "agora"
        wm[target_idx]['timestamp'] = datetime.now().timestamp()

    # 6. Salvar e Aplicar Corte (Top-7)
    await ctx.state_manager.save_working_memory(agent_id, wm)

    # 7. Retorno do Pacote de Percepção V4 (Usando Classes Pydantic)

    # A. Cria o vetor tipado
    query_genlang_vector = GenlangVector(
        source_text=query,
        vector=new_vec,
        model_name="all-MiniLM-L6-v2"  # Nome do modelo usado pelo GeometricBrain
    )

    # B. Cria o pacote de intenção tipado
    # Preenchemos campos opcionais com valores vazios/None para satisfazer o Pydantic
    intent_packet = IntentPacket(
        query_vector=query_genlang_vector,
        intent_vector=None,  # Será preenchido na próxima etapa se necessário
        emotional_valence_vector=None,  # Será preenchido na próxima etapa
        entity_vectors=[],  # Lista vazia
        metadata={
            "xi": float(xi),
            "surprise": float(surprise),
            "gating_action": action
        }
    )

    user_emotion = intent_packet.metadata.get("emotional_tone_description", "neutral")

    # Atualiza assincronamente (fire and forget ou await)
    await ctx.user_profiler.update_user_profile(user_id, {
        "emotional_state": user_emotion
        # Futuro: extrair estilo de comunicação da query
    })

    return {
        "xi": float(xi),
        "surprise": float(surprise),
        "gating_action": action,
        "wm_snapshot": [i['text'] for i in wm],
        # Serializa o objeto Pydantic para um dicionário compatível com Temporal
        "intent_packet": intent_packet.model_dump(),
        "user_id": user_id,
        "identity_glyph": identity_glyph
    }

# --- 2. Investigation (RLM) ---

@activity.defn
async def investigation_activity(state_dict: Dict[str, Any], intent_data: Dict[str, Any]) -> Dict[str, Any]:
    ctx = ActivityContext.get()
    intent = IntentPacket(**intent_data)

    logger.info("Step 2: Investigation (RLM/MBS)")

    # Search Memories
    # memories é uma lista de tuplas (memory_object, score)
    memories_result = await ctx.memory_service.search_raw_memories(intent, top_k=15)

    text_fragments = []
    # Vamos serializar os objetos de memória para passar entre atividades do Temporal
    serialized_memories = []

    for m, score in memories_result:
        # Extrai texto para o contexto bruto (fallback)
        if hasattr(m, 'content') and hasattr(m.content, 'text_content') and m.content.text_content:
            text_fragments.append(str(m.content.text_content))
        elif hasattr(m, 'label'):
            text_fragments.append(f"Entity: {m.label}")

        # Serializa o objeto para enviar para a próxima etapa
        # O método model_dump() do Pydantic é perfeito para isso
        if hasattr(m, 'model_dump'):
            serialized_memories.append(m.model_dump())

    raw_text = "\n".join(text_fragments) if text_fragments else "No relevant memories."

    # Retorna tanto o texto (para compatibilidade) quanto os objetos estruturados
    return {
        "memory_context": raw_text,
        "structured_memories": serialized_memories
    }


# --- 3. Hormonization ---

@activity.defn
async def hormonization_activity(state_dict: Dict[str, Any], xi: float) -> Dict[str, Any]:
    ctx = ActivityContext.get()
    agent_id = state_dict["agent_id"]

    logger.info(f"Step 3: Hormonization (Xi: {xi:.2f})")

    # This now uses the Redis-backed Metacontroller
    steering_result = await ctx.hormonal.process_hormonal_response(agent_id, xi)

    # Update physiological state (Fatigue)
    # We create a dummy metrics dict here, real metrics come from Agency later
    metrics_preview = {"cognitive_strain": xi}
    await ctx.embodiment.process_turn_effects(agent_id, metrics_preview)

    return steering_result


# --- 4. Agency (Deliberation) ---

@activity.defn
async def agency_activity(
        state_dict: Dict[str, Any],
        intent_data: Dict[str, Any],
        hormonal_data: Dict[str, Any],
        context_data: str
) -> Dict[str, Any]:
    ctx = ActivityContext.get()
    agent_id = state_dict["agent_id"]
    intent = IntentPacket(**intent_data)

    logger.info("Step 4: Agency Deliberation")

    # Reconstruct inputs
    # Note: In a real implementation, we would construct full CognitiveStatePacket here
    # For now, we mock the necessary parts for the decide_next_step call

    # Mocking CognitiveStatePacket for Agency Module
    # We need to fetch relevant memories vectors again or pass them.
    # For efficiency in this refactor, we assume context_data string is enough for the prompt,
    # but the AgencyModule expects vectors.
    # In a full migration, we'd pass memory IDs and fetch vectors here.

    identity_vec = GenlangVector(
        vector=state_dict.get("identity_glyph", []),
        source_text="Self",
        model_name="v4"
    )

    # Placeholder vectors for guidance
    vec_zero = GenlangVector(vector=[0.0] * 384, source_text="neutral", model_name="init")
    guidance = GuidancePacket(coherence_vector=vec_zero, novelty_vector=vec_zero)

    cognitive_state = CognitiveStatePacket(
        original_intent=intent,
        identity_vector=identity_vec,
        guidance_packet=guidance,
        metadata=state_dict.get("metadata", {})
    )

    # Configure Guidance from Hormonal Data
    mcl_guidance = {
        "advice": hormonal_data.get("hormonal_injection", ""),
        "agent_name": "Aura",
        "cognitive_state_name": hormonal_data.get("state_label", "STABLE"),
        "mcl_analysis": {"agency_score": 5.0},  # Placeholder
        "biases": {"coherence_bias": 0.5, "novelty_bias": 0.5}
    }

    # Fetch Drives for context
    drives = await ctx.state_manager.get_drives(agent_id)  # Using state manager from context
    # Note: ctx.agency.decide_next_step doesn't take drives directly, but we might use them later.

    # Execute Decision
    # We pass an empty observer for now or a simple logger wrapper
    from ceaf_core.utils.observability_types import ObservabilityManager
    observer = ObservabilityManager(state_dict["session_id"])

    sim_config = {"reality_score": 0.75, "simulation_trust": 0.75}
    chat_history = state_dict.get("metadata", {}).get("chat_history", [])

    winning_strategy = await ctx.agency.decide_next_step(
        cognitive_state=cognitive_state,
        mcl_guidance=mcl_guidance,
        observer=observer,
        sim_calibration_config=sim_config,
        chat_history=chat_history
    )

    return {
        "strategy": winning_strategy.model_dump(mode='json'),
        "mcl_guidance": mcl_guidance  # Pass forward for GTH
    }


# --- 5. Synthesis (GTH) ---

@activity.defn
async def synthesis_activity(
        state_dict: Dict[str, Any],
        intent_data: Dict[str, Any],
        strategy_data: Dict[str, Any],
        hormonal_data: Dict[str, Any],
        memory_context: str,
        structured_memories_data: List[Dict[str, Any]] = []
) -> str:
    ctx = ActivityContext.get()
    agent_id = state_dict["agent_id"]
    user_id = state_dict.get("user_id", "default_user")

    logger.info("Step 5: Synthesis (GTH)")

    user_model = await ctx.user_profiler.get_user_profile(user_id)

    # 1. Rehydrate Strategy
    from ceaf_core.agency_module import WinningStrategy
    strategy = WinningStrategy(**strategy_data)

    # 2. Fetch Self Model (Identity) via NCIM (O CORRETO)
    # O NCIM sabe qual ID buscar e como reconstruir o objeto.
    # Se ele tiver cache interno no futuro, isso será transparente.
    try:
        # Precisamos de um método no NCIM para buscar o modelo atual.
        # Se não existir, usamos o fallback do MBS, mas encapsulado.
        if hasattr(ctx.ncim, 'get_current_self_model'):
             self_model = await ctx.ncim.get_current_self_model()
        else:
             # Fallback temporário direto no MBS se o método não existir no NCIM ainda
             # Mas o ideal é adicionar esse método no NCIM.
             # Vamos assumir o padrão antigo por segurança se o NCIM não tiver o método exposto.
             mem = await ctx.memory_service.get_memory_by_id("ceaf_self_model_singleton_v1")
             if mem and hasattr(mem, 'content') and mem.content.structured_data:
                 from ceaf_core.models import CeafSelfRepresentation
                 self_model = CeafSelfRepresentation(**mem.content.structured_data)
             else:
                 from ceaf_core.models import CeafSelfRepresentation
                 self_model = CeafSelfRepresentation()
    except Exception as e:
        logger.error(f"Erro ao buscar Self Model: {e}")
        from ceaf_core.models import CeafSelfRepresentation
        self_model = CeafSelfRepresentation()

    # 3. Fetch Biological State
    body_state = await ctx.state_manager.get_body_state(agent_id)
    drives = await ctx.state_manager.get_drives(agent_id)

    chat_history = state_dict.get("metadata", {}).get("chat_history", [])

    turn_context = {
        "operational_advice": hormonal_data.get("hormonal_injection"),
        "xi": state_dict.get("xi", 0.0),
        "active_steering": hormonal_data.get("active_steering")
    }

    # 4. Reidratar Memórias de Suporte
    supporting_memories_objects = []
    if structured_memories_data:
        for mem_dict in structured_memories_data:
            try:
                mem_obj = ctx.memory_service._reconstruct_memory_object(mem_dict)
                if mem_obj:
                    supporting_memories_objects.append(mem_obj)
            except Exception as e:
                logger.warning(f"Failed to rehydrate memory object: {e}")

    # 5. Executar Tradução
    response_text = await ctx.gth.translate(
        winning_strategy=strategy,
        supporting_memories=supporting_memories_objects,
        user_model=user_model,
        self_model=self_model,
        agent_name="Aura",
        memory_service=ctx.memory_service,
        chat_history=chat_history,
        body_state=body_state,
        drives=drives,
        turn_context=turn_context,
        original_user_query=intent_data.get("query_vector", {}).get("source_text")
    )

    return response_text


# --- 6. Evolution (BiRAG) ---

@activity.defn
async def evolution_activity(
        state_dict: Dict[str, Any],
        response_text: str,
        evidence_text: str
) -> None:
    ctx = ActivityContext.get()
    logger.info("Step 6: Evolution (BiRAG)")

    entailment = await ctx.birag.validate_entailment(response_text, evidence_text)

    if entailment > 0.7:
        logger.info(f"BiRAG: Validated knowledge (score: {entailment}). Storing.")
        # Store new memory
        from ceaf_core.modules.memory_blossom.memory_types import ExplicitMemory, ExplicitMemoryContent, \
            MemorySourceType, MemorySalience
        new_mem = ExplicitMemory(
            content=ExplicitMemoryContent(text_content=response_text),
            source_type=MemorySourceType.ORA_RESPONSE,
            salience=MemorySalience.HIGH,
            metadata={"entailment_score": entailment},
            agent_id=state_dict["agent_id"]
        )
        await ctx.memory_service.add_specific_memory(new_mem, agent_id=state_dict["agent_id"])


@activity.defn
async def logging_activity(
        state_dict: Dict[str, Any],
        intent_data: Dict[str, Any],
        response_text: str,
        mcl_guidance: Dict[str, Any],
        strategy_data: Dict[str, Any],
        persistence_path_str: str
) -> None:
    """
    Atividade dedicada para persistir o turno no SQLite.
    CORRIGIDA: Garante a gravação do Identity Vector.
    """
    try:
        path = Path(persistence_path_str)
        log_service = CognitiveLogService(persistence_path=path)

        # 1. Recupera/Garante o Vetor de Identidade
        identity_glyph = state_dict.get("identity_glyph", [])

        # Se estiver vazio, tenta um fallback de emergência para não perder o treino
        if not identity_glyph:
            logger.warning(
                "LoggingActivity: Identity Glyph vazio no state_dict. Usando vetor zerado para manter estrutura.")
            identity_glyph = [0.0] * 384  # Dimensão padrão

        # 2. Constrói o Cognitive State Packet EXPLICITAMENTE
        # Isso garante que o DataExtractor encontre o que precisa
        cognitive_packet = {
            "original_intent": intent_data,
            "deliberation_history": state_dict.get("metadata", {}).get("deliberation_history", []),

            # --- O CAMPO QUE FALTAVA ---
            "identity_vector": {
                "vector": identity_glyph,
                "source_text": "Identity Snapshot",
                "model_name": "default"
            }
            # ---------------------------
        }

        # 3. Reconstrói o Response Packet
        response_packet = {
            "content_summary": response_text,
            "confidence_score": 0.85,
            "response_emotional_tone": "neutral"
        }

        # 4. Tenta extrair o vetor de ação da estratégia
        action_vector = strategy_data.get("action_vector")

        # 5. Grava no disco
        log_service.log_turn(
            turn_id=f"turn_{state_dict['session_id']}_{int(time.time())}",
            session_id=state_dict['session_id'],
            cognitive_state_packet=cognitive_packet,  # Agora com Identity!
            response_packet=response_packet,
            mcl_guidance=mcl_guidance,
            action_vector=action_vector
        )
        logger.info(f"💾 LOG SALVO (Com Identity Vector {len(identity_glyph)} dims) em: {path}")

    except Exception as e:
        logger.error(f"❌ Erro crítico ao salvar log: {e}", exc_info=True)