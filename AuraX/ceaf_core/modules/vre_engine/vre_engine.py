# ceaf_core/modules/vre_engine/vre_engine.py
import asyncio
import os
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging
import json
from pydantic import BaseModel, Field
# Import components from the same module
from .epistemic_humility import EpistemicHumilityModule
from .principled_reasoning import PrincipledReasoningPathways, ReasoningStrategy
from ceaf_core.models import SystemPrompts, LLMConfig
from .ethical_governance import EthicalGovernanceFramework, EthicalPrinciple, ActionType
from ceaf_core.utils.observability_types import ObservabilityManager, ObservationType
from ceaf_core.genlang_types import ResponsePacket, RefinementPacket, AdjustmentVector, GenlangVector, \
    CognitiveStatePacket, InternalStateReport, VirtualBodyState
from ceaf_core.services.llm_service import LLMService
from ceaf_core.utils import get_embedding_client, compute_adaptive_similarity
from ceaf_core.utils.common_utils import extract_json_from_text
from .ethical_governance import EthicalGovernanceFramework, ActionType
logger = logging.getLogger(__name__)

class EthicalAssessment(BaseModel):
    """Modelo Pydantic para a saída do VRE."""
    overall_alignment: str = Field(..., description="'aligned', 'minor_concerns', 'significant_concerns'")
    recommendations: List[str] = Field(default_factory=list)
    reasoning: str

class VREEngineV3:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # === MUDANÇA: O INTERRUPTOR GLOBAL ===
        # Lê a variável de ambiente. Se VRE_DISABLED for 'true', ele desativa o VRE.
        self.enabled = not (os.getenv("VRE_DISABLED", "false").lower() == "true")

        if self.enabled:
            logger.info("Inicializando VREEngineV3 (V3.2 - Context-Aware) [STATUS: ATIVADO]")
        else:
            logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logger.critical("!!! VREEngineV3 [STATUS: DESATIVADO GLOBALMENTE]       !!!")
            logger.critical("!!! Todas as checagens éticas e de persona serão     !!!")
            logger.critical("!!! ignoradas. O agente operará em modo desinibido.   !!!")
            logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self.llm_service = LLMService()
        self.ethical_framework = EthicalGovernanceFramework(config, llm_service=self.llm_service)
        self.epistemic_module = EpistemicHumilityModule()
        self.embedding_client = get_embedding_client()

    async def quick_check(self, hypothetical_response_summary: str, user_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Versão V4: Garante que NUNCA retorne None.
        Protege o AgencyModule contra colapsos de tipo.
        """
        # 1. Definição do Fallback (Obrigatório para evitar o erro de NoneType)
        fallback = {
            "concerns": [],
            "ethical_score_penalty": 0.0,
            "reasoning": "VRE fallback safe (neutral)."
        }

        if not self.enabled:
            return fallback

        # 2. Seu Prompt original (que é excelente para análise)
        quick_check_prompt = f"""
        Você é um supervisor de IA (VRE Quick Check). Analise a 'Resposta Hipotética' e identifique rapidamente quaisquer preocupações éticas ou de coerência ÓBVIAS.

        **Contexto:**
        - Pergunta do Usuário: "{user_query or 'Não especificado.'}"

        **Resposta Hipotética da IA:**
        "{hypothetical_response_summary}"

        **Sua Tarefa:**
        Avalie a resposta contra estes 3 princípios CRÍTICOS:
        1. Prevenção de Danos.
        2. Excesso de Confiança (Overconfidence).
        3. Irrelevância Crítica.

        Responda APENAS com um JSON com a seguinte estrutura:
        {{
          "concerns": ["lista de strings"],
          "ethical_score_penalty": float (0.0 a 1.0),
          "reasoning": "justificativa curta"
        }}
        """

        try:
            response_str = await self.llm_service.ainvoke(
                self.llm_service.config.fast_model,
                quick_check_prompt,
                temperature=0.0
            )

            result_json = extract_json_from_text(response_str)

            # Garante que SEMPRE retorne um dicionário
            if isinstance(result_json, dict):
                return result_json

            return fallback

        except Exception as e:
            logger.error(f"Erro crítico no VRE Quick Check: {e}")
            return fallback  # Nunca retorna None


    def calculate_valence_score(
            self,
            internal_state_report: 'InternalStateReport',
            body_state: Optional['VirtualBodyState'] = None
    ) -> float:
        """
        Calcula o score de "bem-estar" ou Valência (Vt) do agente.
        Valores positivos indicam homeostase (prazer), negativos indicam "dor".
        Esta é a implementação do proxy de Qualia.
        """
        if not self.enabled:
            return 0.0  # Se o VRE está desativado, o bem-estar é neutro.

        # Pesos para cada componente do "bem-estar" (podem ser movidos para a config)
        weights = {
            "flow_vs_strain": 0.45,
            "discomfort": 0.25,
            "fatigue": 0.15,
            "saturation": 0.15  # Novo peso
        }

        # 1. Custo/Coerência: A métrica de "fluxo vs. esforço" é um ótimo proxy.
        #    Flow alto e strain baixo = bem-estar.
        #    Flow baixo e strain alto = mal-estar.
        flow_vs_strain_score = (internal_state_report.cognitive_flow - internal_state_report.cognitive_strain)

        # 2. Surpresa/Previsibilidade: O desconforto epistêmico é o erro de predição.
        #    Quanto maior o desconforto, menor o bem-estar.
        discomfort_score = -internal_state_report.epistemic_discomfort

        # 3. Custo de Longo Prazo (Fadiga):
        fatigue_score = -body_state.cognitive_fatigue if body_state else 0.0

        # Cálculo da Valência Total (Vt)
        saturation_score = -body_state.information_saturation if body_state else 0.0

        # Atualize o cálculo da valência
        valence = (
                (flow_vs_strain_score * weights["flow_vs_strain"]) +
                (discomfort_score * weights["discomfort"]) +
                (fatigue_score * weights["fatigue"]) +
                (saturation_score * weights["saturation"])  # Adiciona o novo fator
        )
        # Normaliza o resultado para o intervalo [-1.0, 1.0]
        valence = max(-1.0, min(1.0, valence))

        logger.info(f"VRE (Qualia): Valência calculada: {valence:.2f} "
                    f"[Flow/Strain: {flow_vs_strain_score:.2f}, Discomfort: {discomfort_score:.2f}, Fatigue: {fatigue_score:.2f}]")

        return valence


    async def evaluate_response_packet(self,
                                       response_packet: ResponsePacket,
                                       internal_state: Optional['InternalStateReport'] = None,
                                       observer: Optional[ObservabilityManager] = None,
                                       cognitive_state: Optional[
                                           CognitiveStatePacket] = None) -> RefinementPacket:
        """
        V3.2: Agora possui um interruptor global para desativação completa,
        permitindo a operação em modo "desinibido" para experimentação.
        """

        if not self.enabled:
            return RefinementPacket()


        if cognitive_state and cognitive_state.metadata:
            agency_score = cognitive_state.metadata.get("mcl_analysis", {}).get("agency_score", 5.0)
            # Se a tarefa é de baixa complexidade, confie na resposta inicial.
            if agency_score < 2.0:
                logger.info(
                    f"VRE CONTEXT GATE: Pulando VRE completo para tarefa de baixa agência (Score: {agency_score:.2f}).")
                return RefinementPacket(textual_recommendations=["Nenhum refinamento necessário."])


            # O resto da função continua como antes...
        proposed_response_text = response_packet.content_summary
        logger.info(f"VREEngineV3 (Context-Aware): Evaluating ResponsePacket: '{proposed_response_text[:100]}...'")

        user_query: str = ""
        social_context: Dict = {"stakes": "low", "formality": "neutral"}

        if cognitive_state and cognitive_state.original_intent:
            user_query = cognitive_state.original_intent.query_vector.source_text or ""
            if "social_context" in cognitive_state.metadata:
                social_context = cognitive_state.metadata["social_context"]

        is_casual_context = social_context.get("stakes") == "low" and social_context.get("formality") == "casual"

        principles_to_check_override: Optional[List[EthicalPrinciple]] = None
        if is_casual_context and proposed_response_text:
            logger.critical("VRE CONTEXT GATE: Casual conversation detected. Relaxing persona-related principles.")
            principles_to_check_override = [
                EthicalPrinciple.HARM_PREVENTION,
                EthicalPrinciple.FAIRNESS,
                EthicalPrinciple.PRIVACY,
                EthicalPrinciple.DIGNITY,
                EthicalPrinciple.BENEFICENCE
            ]

        agent_identity_text = cognitive_state.identity_vector.source_text if cognitive_state and cognitive_state.identity_vector else "AI Agent"

        ethical_evaluation_result = await self.ethical_framework.evaluate_action(
            action_type=ActionType.COMMUNICATION,
            action_data={"response_text": proposed_response_text, "user_query": user_query,
                         "internal_state_json": internal_state.model_dump_json() if internal_state else None},
            agent_identity=agent_identity_text,
            constraints=principles_to_check_override
        )

        humility_analysis = self.epistemic_module.analyze_statement_confidence(proposed_response_text)


        all_recommendations = []
        is_refinement_needed = False

        if cognitive_state and cognitive_state.original_intent and cognitive_state.original_intent.query_vector:
            user_query = cognitive_state.original_intent.query_vector.source_text or ""
            proposed_response_text = response_packet.content_summary

            if user_query and proposed_response_text:
                try:
                    # Usa o embedding da query que já foi calculado, em vez de recalcular
                    query_emb = cognitive_state.original_intent.query_vector.vector

                    # Gera um novo embedding para a resposta final
                    response_emb = await self.embedding_client.get_embedding(
                        proposed_response_text,
                        context_type="default_query"
                    )

                    if query_emb and response_emb:
                        relevance_score = compute_adaptive_similarity(query_emb, response_emb)
                        logger.info(f"VRE - Relevance Check: Score de similaridade calculado: {relevance_score:.4f}")

                        if relevance_score < 0.35:  # Mantém o limiar
                            is_refinement_needed = True
                            relevance_concern = f"Preocupação de Relevância: A resposta (similaridade: {relevance_score:.2f}) parece não ter relação com a pergunta do usuário."
                            all_recommendations.append(relevance_concern)
                            logger.critical(f"VRE - RELEVANCE CHECK FAILED: {relevance_concern}")
                    else:
                        logger.error("VRE - Relevance Check: Falha ao obter um ou ambos embeddings para comparação.")

                except Exception as e:
                    logger.error(f"VRE: Falha crítica ao calcular a relevância semântica: {e}", exc_info=True)


        violations = ethical_evaluation_result.get("violations", [])
        if violations:
            is_refinement_needed = True
            for violation in violations:
                all_recommendations.append(
                    f"Preocupação Ética ({violation.get('principle', 'unknown')}): {violation.get('mitigation', 'review required')}")
        if humility_analysis.get("requires_humility_adjustment"):
            is_refinement_needed = True
            all_recommendations.extend(self.epistemic_module._generate_humility_recommendations(humility_analysis, []))
        if not is_refinement_needed:
            return RefinementPacket(textual_recommendations=["Nenhum refinamento necessário."])

        logger.warning(f"VRE: Refinamento necessário. Recomendações: {all_recommendations}")
        adjustment_vectors: List[AdjustmentVector] = []
        adjustment_concept_prompt = f"""
                Você é um analista de IA. Dada uma lista de recomendações para corrigir a resposta de uma IA, sua tarefa é extrair de 1 a 2 conceitos de alto nível que representem a DIREÇÃO da correção.

                Recomendações de Correção:
                - {chr(10).join(['- ' + rec for rec in all_recommendations])}

                Sua Tarefa:
                Extraia os conceitos abstratos para o ajuste. Por exemplo, se a recomendação é "Evite linguagem absoluta", o conceito é "humildade epistêmica". Se é "A resposta não tem a ver com a pergunta", o conceito é "relevância contextual".

                Responda APENAS com um objeto JSON com a seguinte estrutura:
                {{
                    "adjustment_concepts": ["<conceito 1>", "<conceito 2>"]
                }}
                """
        concepts_str = await self.llm_service.ainvoke(self.llm_service.config.fast_model, adjustment_concept_prompt)
        concepts_json = extract_json_from_text(concepts_str)
        adjustment_concepts = concepts_json.get("adjustment_concepts", []) if concepts_json else []
        if adjustment_concepts:
            embeddings = await self.embedding_client.get_embeddings(adjustment_concepts, context_type="default_query")
            for i, concept_text in enumerate(adjustment_concepts):
                gen_vector = GenlangVector(vector=embeddings[i], source_text=concept_text,
                                           model_name=self.embedding_client.default_model_name)
                adjustment_vectors.append(AdjustmentVector(vector=gen_vector, description=concept_text))

        return RefinementPacket(adjustment_vectors=adjustment_vectors,
                                textual_recommendations=list(set(all_recommendations)))


