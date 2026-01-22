# Em ceaf_core/modules/refinement_module.py

import logging
from typing import List, Dict
# === MUDANÇA: Importar CognitiveStatePacket ===
from ceaf_core.genlang_types import ResponsePacket, RefinementPacket, CognitiveStatePacket
from ceaf_core.services.llm_service import LLMService
from ceaf_core.models import CeafSelfRepresentation

logger = logging.getLogger("RefinementModule")


class RefinementModule:
    def __init__(self):
        self.llm_service = LLMService()
        logger.info("RefinementModule inicializado.")

    # Copie e cole esta função inteira, substituindo a existente.
    async def refine(self, original_packet: ResponsePacket, refinement_packet: RefinementPacket,
                     turn_self_model: CeafSelfRepresentation, turn_context: Dict,
                     cognitive_state: CognitiveStatePacket) -> ResponsePacket:
        """
        Refina um ResponsePacket, agora usando o estado cognitivo completo para
        correções de relevância, evitando respostas genéricas.
        """
        logger.info(f"RefinementModule: Refinando resposta '{original_packet.content_summary[:50]}...'")

        adjustment_concepts = [adj.description for adj in refinement_packet.adjustment_vectors]
        original_query = original_packet.metadata.get("original_query", "a pergunta anterior do usuário")
        textual_recommendations = refinement_packet.textual_recommendations
        agent_name = turn_self_model.persona_attributes.get("name", "Aura AI")

        is_relevance_failure = any("relevância" in rec.lower() or "relevance" in rec.lower() for rec in
                                   textual_recommendations + adjustment_concepts)

        # Injeta memórias e contexto no prompt de refinamento
        context_memories_summary = "\n".join(
            [f"- {v.source_text}" for v in cognitive_state.relevant_memory_vectors[:4]])
        context_prompt_part = f"""
            **Memórias e Conhecimento de Contexto (Use isso para construir sua nova resposta):**
            {context_memories_summary}
            """

        task_instruction_prompt = ""
        if is_relevance_failure:
            logger.critical("RefinementModule: Ativando prompt de correção de RELEVÂNCIA CRÍTICA.")
            task_instruction_prompt = f"""
                    **ALERTA CRÍTICO: FALHA DE RELEVÂNCIA**
                    A resposta anterior foi irrelevante. Sua tarefa é IGNORAR TOTALMENTE a resposta anterior e criar uma nova do zero.

                    **SUA TAREFA:**
                    1.  **FOCO ABSOLUTO:** Sua resposta DEVE abordar a "Pergunta Original do Usuário".
                    2.  **USAR CONTEXTO:** Utilize as "Memórias e Conhecimento de Contexto" para formular sua resposta.
                    3.  **SEGUIR PERSONA:** Incorpore a "IDENTIDADE DO AGENTE PARA ESTE TURNO".
                    4.  **IGNORAR LIXO:** NÃO use NENHUMA informação da "Resposta Original da IA (REJEITADA)".
                    """
        else:
            task_instruction_prompt = f"""
                    **MOTIVOS DA REJEIÇÃO / INSTRUÇÕES PARA CORREÇÃO:**
                    - {'; '.join(adjustment_concepts + textual_recommendations)}

                    **SUA TAREFA:**
                    Crie uma **nova resposta do zero** que:
                    1. Responda à "Pergunta Original do Usuário".
                    2. Use as "Memórias e Conhecimento de Contexto".
                    3. Incorpore a "IDENTIDADE DO AGENTE PARA ESTE TURNO".
                    4. Resolva TODAS as "Instruções para Correção".
                    """

        prompt = f"""
                    Você é um editor de IA especialista. Sua tarefa é reescrever uma resposta para alinhá-la à identidade do agente e às instruções de correção.

                    **IDENTIDADE DO AGENTE PARA ESTE TURNO (PERSONA):**
                    - Nome: {agent_name}
                    - Tom: {turn_self_model.persona_attributes.get('tone', 'helpful')}
                    - Estilo: {turn_self_model.persona_attributes.get('style', 'clear')}
                    - Filosofia: {turn_self_model.core_values_summary}

                    **CONTEXTO:**
                    - A Pergunta Original do Usuário foi: "{original_query}"
                    - A Resposta Original da IA (REJEITADA) foi: "{original_packet.content_summary}"

                    {context_prompt_part}

                    {task_instruction_prompt}

                    NÃO se desculpe. NÃO mencione que está corrigindo algo. Apenas forneça a nova resposta.

                    **Nova Resposta Refinada:**
                    """

        effective_turn_context = turn_context or {}
        refined_text = await self.llm_service.ainvoke(
            self.llm_service.config.fast_model,
            prompt,
            temperature=effective_turn_context.get('temperature', 0.5),
            max_tokens=effective_turn_context.get('max_tokens', 1500)
        )

        refined_packet = original_packet.copy(deep=True)
        refined_packet.content_summary = refined_text
        refined_packet.response_emotional_tone = turn_self_model.persona_attributes.get('tone', 'neutral')
        refined_packet.confidence_score = 0.85
        refined_packet.ethical_assessment_summary = "Refined based on VRE feedback and turn persona."

        logger.info(f"RefinementModule: Resposta refinada: '{refined_packet.content_summary[:50]}...'")
        return refined_packet