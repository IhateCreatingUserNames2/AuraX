# ceaf_core/modules/lcam_module.py
import logging
import re
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING

from ceaf_core.genlang_types import CognitiveStatePacket, RefinementPacket
from ceaf_core.utils.embedding_utils import get_embedding_client, compute_adaptive_similarity
from ceaf_core.modules.memory_blossom.memory_types import ReasoningMemory, ReasoningStep, ExplicitMemory, \
    ExplicitMemoryContent, MemorySourceType, MemorySalience

# --- CORREÇÃO DO IMPORT CIRCULAR ---
if TYPE_CHECKING:
    # Estes imports só acontecem durante checagem de tipo, não na execução
    from ceaf_core.agency_module import WinningStrategy
    from ceaf_core.services.mbs_memory_service import MBSMemoryService
# -----------------------------------

logger = logging.getLogger("CEAFv3_LCAM")


class LCAMModule:
    """
    Loss Cataloging and Analysis Module (V3).
    Identifica interações de 'falha' e cria memórias sobre elas para aprendizado futuro.
    """

    # Note as aspas em 'MBSMemoryService'. Isso é um "Forward Reference".
    def __init__(self, memory_service: 'MBSMemoryService'):
        self.memory = memory_service
        self.embedding_client = get_embedding_client()
        logger.info("LCAMModule (V3) inicializado.")

    def predict_turn_outcome(self, cognitive_state: CognitiveStatePacket, mcl_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gera uma previsão sobre o resultado do turno, agindo como o sistema de expectativa do cérebro.
        """
        logger.info("LCAM Prediction: Gerando previsão do resultado do turno...")

        agency_score = mcl_params.get("mcl_analysis", {}).get("agency_score", 5.0)
        coherence_bias = mcl_params.get("biases", {}).get("coherence_bias", 0.7)

        # 1. Complexidade Percebida
        complexity_factor = 1.0 - (min(agency_score, 10.0) / 10.0)

        # 2. Coerência do Contexto
        coherence_factor = 0.5
        memory_vectors = [np.array(vec.vector) for vec in cognitive_state.relevant_memory_vectors if vec.vector]
        if len(memory_vectors) > 1:
            similarities = []
            for i in range(len(memory_vectors)):
                for j in range(i + 1, len(memory_vectors)):
                    sim = compute_adaptive_similarity(memory_vectors[i].tolist(), memory_vectors[j].tolist())
                    similarities.append(sim)
            if similarities:
                coherence_factor = np.mean(similarities)

        # 3. Alinhamento com a Persona
        alignment_factor = 1.0
        if agency_score > 5.0 and coherence_bias > 0.7:
            alignment_factor = 0.8
        elif agency_score < 3.0 and coherence_bias < 0.5:
            alignment_factor = 0.9

        expected_confidence = (complexity_factor * 0.5) + (coherence_factor * 0.3) + (alignment_factor * 0.2)

        prediction = {
            "expected_final_confidence": max(0.0, min(1.0, expected_confidence)),
            "expected_vre_rejections": 0,
            "prediction_timestamp": time.time()
        }

        return prediction

    async def analyze_and_catalog_loss(self,
                                       turn_prediction: Dict[str, Any],
                                       turn_metrics: Dict[str, Any],
                                       cognitive_state: CognitiveStatePacket,
                                       winning_strategy: 'WinningStrategy',
                                       final_response: str
                                       ):
        """
        Versão 2.1 (GCSL-NF): Aprende tanto com o que deu certo quanto com o que deu errado.
        """
        expected_confidence = turn_prediction.get("expected_final_confidence", 0.5)
        actual_confidence = turn_metrics.get("final_confidence", 0.5)
        confidence_error = actual_confidence - expected_confidence

        rejection_count = turn_metrics.get("vre_rejection_count", 0) + turn_metrics.get("user_feedback_rejections", 0)
        rejection_error = -1.0 * rejection_count

        prediction_error_signal = (confidence_error * 0.4) + (rejection_error * 0.6)

        # 1. FEEDBACK POSITIVO
        await self._create_positive_feedback_memory(
            initial_state=cognitive_state,
            trajectory=winning_strategy,
            actual_outcome_metrics=turn_metrics,
            final_response=final_response
        )

        # 2. FEEDBACK NEGATIVO
        if prediction_error_signal < -0.25:
            logger.critical(f"LCAM (Negative Feedback): Falha inesperada! Sinal: {prediction_error_signal:.2f}")
            await self._create_negative_feedback_memory(
                initial_state=cognitive_state,
                trajectory=winning_strategy,
                expected_outcome=turn_prediction,
                actual_outcome_metrics=turn_metrics,
                prediction_error=prediction_error_signal
            )

        # 3. FEEDBACK DE REFORÇO
        elif prediction_error_signal > 0.25:
            logger.critical(f"LCAM (Positive Reinforcement): Sucesso inesperado! Sinal: {prediction_error_signal:.2f}")

    async def _create_positive_feedback_memory(self, initial_state: CognitiveStatePacket, trajectory: 'WinningStrategy',
                                               actual_outcome_metrics: Dict, final_response: str):
        outcome_status = "success" if actual_outcome_metrics.get("vre_rejection_count", 0) == 0 else "failure"

        reasoning_mem = ReasoningMemory(
            task_description=initial_state.original_intent.query_vector.source_text,
            strategy_summary=trajectory.strategy_description or f"Uso de ferramenta: {trajectory.tool_call_request}",
            reasoning_steps=[
                ReasoningStep(step_number=1, description="Estratégia", reasoning=trajectory.reasoning)],
            outcome=outcome_status,
            outcome_reasoning=f"Confiança: {actual_outcome_metrics.get('final_confidence', 0):.2f}",
            source_type=MemorySourceType.INTERNAL_REFLECTION,
            source_turn_id=actual_outcome_metrics.get("turn_id"),
            salience=MemorySalience.MEDIUM
        )
        await self.memory.add_specific_memory(reasoning_mem)

    async def _create_negative_feedback_memory(self, initial_state: CognitiveStatePacket, trajectory: 'WinningStrategy',
                                               expected_outcome: Dict, actual_outcome_metrics: Dict,
                                               prediction_error: float):
        loss_content_text = f"""
        Lição Aprendida (Falha de Predição):
        - Tarefa: "{initial_state.original_intent.query_vector.source_text}"
        - Estratégia: "{trajectory.strategy_description or 'Tool Call'}"
        - Esperado: {expected_outcome.get('expected_final_confidence', 0.5):.2f} / Real: {actual_outcome_metrics.get('final_confidence', 0.5):.2f}
        - Erro: {prediction_error:.2f}. Evitar esta trajetória em contextos similares.
        """

        loss_memory = ExplicitMemory(
            content=ExplicitMemoryContent(text_content=loss_content_text),
            memory_type="explicit",
            source_type=MemorySourceType.INTERNAL_REFLECTION,
            salience=MemorySalience.HIGH,
            keywords=["failure", "prediction_error", "lcam_lesson"],
            failure_pattern="prediction_error",
            source_turn_id=actual_outcome_metrics.get("turn_id"),
            learning_value=abs(prediction_error)
        )
        await self.memory.add_specific_memory(loss_memory)

    async def get_insights_on_potential_failure(
            self,
            current_query: str,
            similarity_threshold: float = 0.80
    ) -> Optional[Dict[str, Any]]:
        """Busca no MBS por memórias de falhas semanticamente similares."""
        lcam_search_query = f"falha erro lição_aprendida {current_query}"
        potential_failures = await self.memory.search_raw_memories(lcam_search_query, top_k=3)

        if not potential_failures:
            return None

        try:
            query_embedding = await self.embedding_client.get_embedding(current_query, context_type="default_query")
        except Exception:
            return None

        best_match: Optional[Tuple[ExplicitMemory, float]] = None

        for mem_obj, score in potential_failures:
            # Filtro básico
            if "falha" not in mem_obj.keywords and "erro" not in mem_obj.keywords:
                continue

            # Busca embedding no cache do serviço de memória (dependência interna)
            # Em arquitetura pura, deveríamos pedir ao serviço para calcular a similaridade,
            # mas aqui acessamos o cache por performance se disponível
            mem_embedding = getattr(self.memory, '_embedding_cache', {}).get(mem_obj.memory_id)

            # Se não tiver cache local, confiamos no score do qdrant (que já é similaridade)
            similarity = score
            if mem_embedding:
                similarity = compute_adaptive_similarity(query_embedding, mem_embedding)

            if similarity > similarity_threshold:
                if best_match is None or similarity > best_match[1]:
                    best_match = (mem_obj, similarity)

        if best_match:
            matched_memory, match_similarity = best_match
            failure_reason = "Motivo não especificado"
            if matched_memory.content and matched_memory.content.text_content:
                match = re.search(r"Motivo da Falha:\s*(.*)", matched_memory.content.text_content, re.IGNORECASE)
                if match: failure_reason = match.group(1).strip()

            return {
                "warning_level": "high" if match_similarity > 0.9 else "medium",
                "message": f"Cuidado: Situação {match_similarity:.0%} similar a uma falha passada.",
                "past_failure_memory_id": matched_memory.memory_id,
                "recommendation": "Aumente a revisão ética e cautela."
            }

        return None