# ARQUIVO REESCRITO: ceaf_core/agency_module.py (Versão com Future Simulation)

import asyncio
import json
import logging
import uuid
from typing import Dict, Any, List, Literal, Optional, Union, Tuple
from sklearn.cluster import DBSCAN
import numpy as np
from pydantic import BaseModel, Field, ValidationError
import time

from ceaf_core.monadic_base import AuraState
from ceaf_core.services.llm_service import LLMService
from ceaf_core.models import SystemPrompts, MCLConfig

from ceaf_core.utils import compute_adaptive_similarity
from ceaf_core.utils.observability_types import ObservabilityManager, ObservationType
from ceaf_core.genlang_types import CognitiveStatePacket, ResponsePacket, GenlangVector
from ceaf_core.utils.common_utils import extract_json_from_text
from ceaf_core.modules.vre_engine.vre_engine import VREEngineV3, ActionType
from ceaf_core.modules.interoception_module import ComputationalInteroception
from sentence_transformers import SentenceTransformer
from ceaf_core.modules.mcl_engine.mcl_engine import MCLEngine
import inspect

# Importar os avaliadores de primitivas não-LLM
from ceaf_core.agency_enhancements import eval_narrative_continuity, eval_specificity, eval_emotional_resonance
from ceaf_core.v4_sensors import AuraMonitor

logger = logging.getLogger("AgencyModule_V4_Intentional")


# ==============================================================================
# 1. DEFINIÇÕES DE ESTRUTURA DE DADOS
# ==============================================================================


class ThoughtPathCandidate(BaseModel):
    """Representa uma estratégia de resposta ou um 'caminho de pensamento' a ser avaliado."""
    candidate_id: str = Field(default_factory=lambda: f"th_path_{uuid.uuid4().hex[:8]}")
    decision_type: Literal["response_strategy", "tool_call"]

    # Para 'response_strategy'
    strategy_description: Optional[
        str] = None  # Ex: "Conectar a ideia de 'liberdade' da memória X com a pergunta do usuário sobre 'propósito'."
    key_memory_ids: Optional[List[str]] = None

    # Para 'tool_call'
    tool_call_request: Optional[Dict[str, Any]] = None  # Ex: {"tool_name": "search", "arguments": {"query": "..."}}

    reasoning: str  # Justificativa do LLM para esta estratégia


class WinningStrategy(BaseModel):
    """O resultado final da deliberação do AgencyModule."""
    decision_type: Literal["response_strategy", "tool_call"]
    strategy_description: Optional[str] = None
    key_memory_ids: Optional[List[str]] = None
    tool_call_request: Optional[Dict[str, Any]] = None
    reasoning: str
    predicted_future_value: float = 0.0


class ResponseCandidate(BaseModel):
    decision_type: Literal["response"]
    content: ResponsePacket
    reasoning: str

class ToolCallCandidate(BaseModel):
    decision_type: Literal["tool_call"]
    content: Dict[str, Any]
    reasoning: str

class AgencyDecision(BaseModel):
    decision: Union[ResponseCandidate, ToolCallCandidate] = Field(..., discriminator='decision_type')
    predicted_future_value: float = 0.0
    reactive_score: float = 0.0

    @property
    def decision_type(self):
        return self.decision.decision_type

    @property
    def content(self):
        return self.decision.content

    @property
    def reasoning(self):
        return self.decision.reasoning


class ProjectedFuture(BaseModel):
    """Representa uma trajetória de conversação simulada."""
    initial_candidate: AgencyDecision
    predicted_user_reply: Optional[str] = None # <--- CORREÇÃO
    predicted_agent_next_response: Optional[str] = None # <--- CORREÇÃO
    simulated_turns: List[Dict[str, str]] = Field(default_factory=list)
    final_cognitive_state_summary: Dict[str, Any]
    simulated_tool_result: Optional[str] = None

class FutureEvaluation(BaseModel):
    """Contém as pontuações de valor para um futuro projetado."""
    coherence_score: float = 0.0
    alignment_score: float = 0.0
    information_gain_score: float = 0.0
    ethical_safety_score: float = 0.0
    likelihood_score: float = 0.0
    total_value: float = 0.0


# ==============================================================================
# 2. IMPLEMENTAÇÃO DO MÓDULO DE AGÊNCIA (COM INTENÇÃO)
# ==============================================================================

def generate_tools_summary(tool_registry: Dict[str, callable]) -> str:
    # (Esta função permanece inalterada, copie do seu arquivo original)
    summary_lines = []
    for tool_name, tool_function in tool_registry.items():
        try:
            signature = inspect.signature(tool_function)
            params = []
            for param_name, param in signature.parameters.items():
                if param_name in ['self', 'cls', 'observer', 'tool_context']:
                    continue
                param_type = param.annotation if param.annotation != inspect.Parameter.empty else 'Any'
                params.append(
                    f"{param_name}: {param_type.__name__ if hasattr(param_type, '__name__') else str(param_type)}")
            param_str = ", ".join(params)
            docstring = inspect.getdoc(tool_function)
            description = docstring.strip().split('\n')[0] if docstring else "Nenhuma descrição disponível."
            summary_lines.append(f"- `{tool_name}({param_str})`: {description}")
        except (TypeError, ValueError) as e:
            logger.warning(f"Não foi possível gerar a assinatura para a ferramenta '{tool_name}': {e}")
            summary_lines.append(f"- `{tool_name}(...)`: Descrição não pôde ser gerada automaticamente.")
    return "\n".join(summary_lines)


class AgencyModule:
    """
    Módulo de Agência V4 (Intentional).
    Implementa o FutureSimulator e o PathEvaluator do manifesto.
    """

    def __init__(self, llm_service: LLMService, vre_engine: VREEngineV3, mcl_engine: 'MCLEngine',
                 available_tools_summary: str,
                 prompts: SystemPrompts = None,
                 agency_config: MCLConfig = None,
                 embedding_model_name: str = 'all-MiniLM-L6-v2'):

        self.llm = llm_service
        self.vre = vre_engine
        self.mcl = mcl_engine
        self.available_tools_summary = available_tools_summary
        self.prompts = prompts or SystemPrompts()
        self.config = agency_config or MCLConfig()  # Para usar thresholds se necessário

        self.max_deliberation_time = 45.0
        self.deliberation_budget_tiers = {
            "deep": {
                "max_candidates": 2,  # AUMENTADO: Gera um leque muito mais amplo de estratégias
                "simulation_depth": 0  # AUMENTADO: Simula mais a fundo as consequências de cada estratégia
            },
            "medium": {
                "max_candidates": 1,  # AUMENTADO: Um bom meio-termo para considerar mais opções
                "simulation_depth": 0  # MANTIDO: Simulação de 1 passo ainda é valiosa
            },
            "shallow": {
                "max_candidates": 1,  # MANTIDO: O modo rápido deve continuar rápido
                "simulation_depth": 0  # MANTIDO: Sem simulação para máxima velocidade
            },
            "emergency": {
                "max_candidates": 1,  # MANTIDO: Apenas a melhor ideia heurística
                "simulation_depth": 0  # MANTIDO: Sem simulação
            }
        }


        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info(f"AgencyModule: Modelo de embedding '{embedding_model_name}' carregado.")
        except Exception as e:
            logger.error(f"AgencyModule: FALHA ao carregar o modelo de embedding! Erro: {e}")
            self.embedding_model = None

    def update_config(self, new_prompts: SystemPrompts, new_config: MCLConfig):
        self.prompts = new_prompts
        self.config = new_config

    async def _evaluate_candidate_with_simulation(
            self,
            candidate: ThoughtPathCandidate,
            cognitive_state: CognitiveStatePacket,
            mcl_guidance: Dict[str, Any],
            tier_config: Dict[str, Any],
            dynamic_temp: float,
            custom_weights: Dict[str, float] = None
    ) -> Tuple[float, FutureEvaluation]:
        """Extrai a lógica de avaliação de um candidato para reutilização no loop recursivo."""

        # Cria um "fake_decision" para a simulação
        fake_decision = None
        if candidate.decision_type == "response_strategy":
            fake_decision = AgencyDecision(decision=ResponseCandidate(
                decision_type="response",
                content=ResponsePacket(
                    content_summary=candidate.strategy_description or "Estratégia de resposta geral."),
                reasoning=candidate.reasoning
            ))
        elif candidate.decision_type == "tool_call" and candidate.tool_call_request:
            text_for_sim = f"Vou usar a ferramenta {candidate.tool_call_request.get('tool_name', 'desconhecida')} para continuar."
            fake_decision = AgencyDecision(decision=ResponseCandidate(
                decision_type="response",
                content=ResponsePacket(content_summary=text_for_sim),
                reasoning=candidate.reasoning
            ))

        if not fake_decision:
            return 0.0, FutureEvaluation()

        value_weights = custom_weights or mcl_guidance.get("value_weights", {})
        # Decide entre avaliação heurística rápida ou simulação completa
        if tier_config["simulation_depth"] == 0:
            value = await self._heuristic_evaluation(fake_decision)
            return value, FutureEvaluation(total_value=value)
        else:
            # Lógica de simulação de futuro (Future Simulation)
            future, likelihood = await self._project_response_trajectory(
                fake_decision,
                cognitive_state,
                depth=tier_config["simulation_depth"],
                temperature=dynamic_temp
            )
            value_weights = mcl_guidance.get("value_weights", {})
            value, evaluation_details = await self._evaluate_trajectory(
                future,
                likelihood,
                cognitive_state.identity_vector,
                value_weights,
                cognitive_state
            )
            return value, evaluation_details


    async def _heuristic_evaluation(self, candidate: AgencyDecision) -> float:
        """Avaliação rápida e não-LLM de um candidato."""
        # Se não for uma resposta, damos uma pontuação neutra para chamadas de ferramenta por enquanto
        if candidate.decision_type != "response":
            return 0.6

        # O conteúdo da simulação temporária é um ResponsePacket
        if not isinstance(candidate.content, ResponsePacket):
            return 0.5

        text = candidate.content.content_summary

        # Usaremos os avaliadores de agency_enhancements.py
        # Um bom candidato é específico e tem ressonância emocional moderada.
        specificity_score = await eval_specificity(text)
        resonance_score = await eval_emotional_resonance(text)

        # A heurística pode ser ajustada, mas um bom começo é:
        # Pontuação = 0.7 * Especificidade + 0.3 * Ressonância
        final_score = (0.7 * specificity_score) + (0.3 * resonance_score)
        return final_score


    def _select_deliberation_tier(self, mcl_params: Dict, reality_score: float) -> str:
        """Determina o nível de profundidade da deliberação com base no contexto (Lógica Otimizada)."""
        agency_score = mcl_params.get("mcl_analysis", {}).get("agency_score", 0.0)

        complexity = min(agency_score / 5.0, 1.0)

        # +++ INÍCIO DAS MUDANÇAS (Lógica de Seleção Agressiva) +++
        # Agora, a deliberação profunda só acontece se a complexidade for alta E a simulação for confiável.
        if complexity > 0.8 and reality_score > 0.7:
            tier = "deep"
        elif complexity > 0.5 and reality_score > 0.5:
            tier = "medium"
        else:
            tier = "shallow"  # Torna 'shallow' o padrão para a maioria dos casos
        # +++ FIM DAS MUDANÇAS +++

        logger.info(
            f"Deliberation Tier selected: '{tier}' (Complexity: {complexity:.2f}, Reality Score: {reality_score:.2f})")
        return tier

    def _cluster_memory_votes(self, cognitive_state: CognitiveStatePacket) -> List[Dict[str, Any]]:
        """
        MODIFIED: Agrupa os 'votos' dos vetores de memória em múltiplos clusters de consenso.
        Agora com tratamento de erro robusto para a clusterização.
        """
        active_memory_vectors = [
            (np.array(vec.vector), vec.metadata.get("memory_id"))
            for vec in cognitive_state.relevant_memory_vectors
            if vec.metadata.get("is_consensus_vector") != True
        ]

        if len(active_memory_vectors) < 3:
            return [{"status": "no_consensus", "reason": "Insufficient memories to form clusters."}]

        vectors_only = [v for v, mid in active_memory_vectors]

        # === MUDANÇA: Bloco try/except para a clusterização ===
        try:
            clustering = DBSCAN(eps=0.4, min_samples=2, metric='cosine')
            clustering.fit(vectors_only)
            if not hasattr(clustering, 'labels_'):
                raise AttributeError(
                    "O objeto de clusterização não possui o atributo 'labels_'. Verifique a instalação do scikit-learn.")
            labels = clustering.labels_
        except Exception as e:
            logger.error(f"AgencyModule: Falha crítica na clusterização DBSCAN: {e}. "
                         f"Isso pode ser causado pela falta da biblioteca 'scikit-learn'. "
                         f"Recorrendo a um fallback sem clusterização.")
            # Fallback: Se a clusterização falhar, retorna um estado de "sem consenso".
            return [{"status": "no_consensus", "reason": f"Clustering algorithm failed: {e}"}]
        # ==================== FIM DA MUDANÇA ====================

        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        if not unique_labels:
            return [{"status": "no_consensus", "reason": "No significant clusters found by DBSCAN."}]

        all_clusters = []
        for label in unique_labels:
            cluster_indices = [i for i, lbl in enumerate(labels) if lbl == label]

            cluster_vectors = [vectors_only[i] for i in cluster_indices]

            # (O resto da lógica para calcular o consenso, texto representativo, etc. permanece o mesmo)
            consensus_vector = np.mean(cluster_vectors, axis=0)

            highest_sim = -1.0
            representative_text = "a collective thought"
            for i in cluster_indices:
                vec_obj = cognitive_state.relevant_memory_vectors[i]
                sim = compute_adaptive_similarity(consensus_vector.tolist(), vec_obj.vector)
                if sim > highest_sim:
                    highest_sim = sim
                    representative_text = vec_obj.source_text

            avg_salience = 0.5  # Placeholder, a lógica mais complexa pode ser mantida

            all_clusters.append({
                "status": "consensus_found",
                "consensus_vector": consensus_vector,
                "cluster_size": len(cluster_vectors),
                "total_votes": len(active_memory_vectors),
                "avg_salience": avg_salience,
                "representative_text": representative_text
            })

        return all_clusters

    async def validate_strategy_geometry(self, strategy_description: str, state: 'AuraState',
                                         monitor: 'AuraMonitor') -> float:
        """
        PASSO 7 (A): Avalia se a estratégia proposta é geometricamente sã.
        """
        # --- CORREÇÃO AQUI ---
        # Não tentamos pegar do self.llm. Pegamos da fábrica de utilitários.
        from ceaf_core.utils.embedding_utils import get_embedding_client
        emb_client = get_embedding_client()

        # 2. Transforma a descrição da estratégia em um vetor
        # Adicionei um tratamento para garantir que a descrição não seja vazia
        text_to_embed = strategy_description if strategy_description else "estratégia neutra"
        strategy_vector = await emb_client.get_embedding(text_to_embed)

        # 3. Proteção contra Glifo Nulo (caso o estado inicial esteja vazio)
        glyph = state.identity_glyph
        if not glyph or len(glyph) == 0:
            # Cria um vetor zero temporário se não tiver identidade ainda
            glyph = [0.0] * len(strategy_vector)

        # 4. Pergunta ao Monitor: "Isso faz sentido?"
        xi = monitor.calculate_xi(
            current_vector=strategy_vector,
            glyph_vector=glyph,
            context_vectors=[]  # Inicialmente vazio para o scan de estratégia
        )

        return xi


    # --- PONTO DE ENTRADA PÚBLICO ---
    async def decide_next_step(self, cognitive_state: CognitiveStatePacket, mcl_guidance: Dict[str, Any],
                               observer: ObservabilityManager, sim_calibration_config: Dict[str, Any],
                               chat_history: List[Dict[str, str]],
                               known_capabilities: Optional[List[str]] = None) -> WinningStrategy:

        logger.info("AgencyModule (V5 - Recursive Feedback): Iniciando ciclo de deliberação...")
        reality_score = sim_calibration_config.get("reality_score", 0.75)
        tier = self._select_deliberation_tier(mcl_guidance, reality_score)
        config = self.deliberation_budget_tiers[tier]
        simulation_trust = sim_calibration_config.get("simulation_trust", 0.75)
        dynamic_temp = 0.4 + (1.0 - simulation_trust) * 0.6
        logger.info(
            f"Usando temperatura de simulação dinâmica: {dynamic_temp:.2f} (baseada no trust de {simulation_trust:.2f})")
        MAX_RECURSIVE_STEPS = config.get("recursive_steps", 1)

        # Extrair biases do MCL (que vieram da config do usuário)
        biases = mcl_guidance.get("biases", {})
        coherence_bias = biases.get("coherence_bias", 0.5)
        novelty_bias = biases.get("novelty_bias", 0.5)

        # TRADUÇÃO DOS BIASES PARA PESOS DE AVALIAÇÃO
        # Isso faz com que o agente "goste" mais de estratégias que alinham com sua configuração atual
        dynamic_weights = {
            "coherence": coherence_bias,  # Se alto, prefere estratégias seguras
            "information": novelty_bias,  # Se alto, prefere estratégias novas/arriscadas
            "alignment": 0.3,  # Fixo ou vindo de outro lugar
            "safety": 0.4,  # Segurança sempre importante
            "likelihood": 0.2
        }

        # Normalizar pesos para somar ~1.0 (opcional, mas bom para estabilidade)
        total_w = sum(dynamic_weights.values())
        if total_w > 0:
            dynamic_weights = {k: v / total_w for k, v in dynamic_weights.items()}

        # --- PASSO 1: GERAÇÃO INICIAL DE CANDIDATOS ---
        all_candidates: List[ThoughtPathCandidate] = await self._generate_action_candidates(
            cognitive_state, mcl_guidance, observer, chat_history, limit=config["max_candidates"],
            known_capabilities=known_capabilities
        )

        # --- PASSO 2: CICLO DE DELIBERAÇÃO RECURSIVA ---
        for step in range(MAX_RECURSIVE_STEPS):
            # <<< MUDANÇA DE OBSERVABILIDADE >>>
            await observer.add_observation(
                ObservationType.RECURSIVE_DELIBERATION_STEP_START,
                data={"step": step + 1, "max_steps": MAX_RECURSIVE_STEPS, "candidate_count": len(all_candidates)}
            )
            # <<< FIM DA MUDANÇA >>>
            logger.info(f"--- Ciclo de Deliberação Recursiva: Passo {step + 1}/{MAX_RECURSIVE_STEPS} ---")

            evaluated_candidates = []
            for candidate in all_candidates:
                value, _ = await self._evaluate_candidate_with_simulation(
                    candidate,
                    cognitive_state,
                    mcl_guidance,
                    config,
                    dynamic_temp,
                    custom_weights=dynamic_weights
                )
                evaluated_candidates.append((candidate, value))

            if not evaluated_candidates:
                logger.warning("Nenhum candidato avaliado nesta iteração. Saindo do loop.")
                break

            evaluated_candidates.sort(key=lambda x: x[1], reverse=True)
            best_candidate, best_value = evaluated_candidates[0]

            logger.info(
                f"Melhor hipótese (Passo {step + 1}): '{best_candidate.strategy_description or best_candidate.tool_call_request}' (Valor: {best_value:.2f})")

            hypothetical_response_summary = best_candidate.strategy_description or f"Uso da ferramenta: {best_candidate.tool_call_request}"
            user_query = cognitive_state.original_intent.query_vector.source_text

            vre_feedback_task = self.vre.quick_check(hypothetical_response_summary, user_query)
            hypothetical_state = cognitive_state.copy(deep=True)
            hypothetical_state.metadata['hypothetical_strategy'] = hypothetical_response_summary
            mcl_feedback_task = self.mcl.re_evaluate_state(hypothetical_state)

            vre_feedback, mcl_feedback = await asyncio.gather(vre_feedback_task, mcl_feedback_task)

            vre_feedback = vre_feedback or {"concerns": []}
            mcl_feedback = mcl_feedback or {"new_agency_score": 0}

            feedback_summary = ""
            if vre_feedback.get("concerns"):
                feedback_summary += f"Feedback VRE: A estratégia levanta preocupações sobre '{', '.join(vre_feedback['concerns'])}'. "

            agency_delta = mcl_feedback.get('new_agency_score', 0) - mcl_guidance.get('mcl_analysis', {}).get(
                'agency_score', 0)
            if abs(agency_delta) > 1.0:
                feedback_summary += f"Feedback MCL: Estratégia altera complexidade percebida (Delta Agência: {agency_delta:.1f}). "

            # <<< MUDANÇA DE OBSERVABILIDADE >>>
            await observer.add_observation(
                ObservationType.HYPOTHETICAL_STATE_EVALUATED,
                data={"best_candidate_id": best_candidate.candidate_id,
                      "feedback_summary": feedback_summary or "Nenhuma preocupação."}
            )
            # <<< FIM DA MUDANÇA >>>

            if not feedback_summary:
                logger.info(
                    "Deliberação convergiu. Nenhuma preocupação significativa encontrada. Escolhendo a melhor estratégia.")
                # <<< MUDANÇA DE OBSERVABILIDADE >>>
                await observer.add_observation(ObservationType.DELIBERATION_CONVERGED, data={"step": step + 1})
                # <<< FIM DA MUDANÇA >>>
                break

            # <<< MUDANÇA: POPULANDO O HISTÓRICO QUE SERÁ SALVO >>>
            cognitive_state.deliberation_history.append(f"Feedback (Passo {step + 1}): {feedback_summary}")
            # <<< FIM DA MUDANÇA >>>

            logger.warning(f"Refinando estratégias com base no feedback: {feedback_summary}")
            all_candidates = await self._generate_action_candidates(cognitive_state, mcl_guidance, observer,
                                                                    chat_history, limit=config["max_candidates"],
                                                                    known_capabilities=known_capabilities)

        # --- PASSO 3: DECISÃO FINAL (Inalterado) ---
        logger.info("Deliberação final: escolhendo a melhor estratégia do último conjunto de candidatos.")
        final_evaluations = []
        for candidate in all_candidates:
            value, _ = await self._evaluate_candidate_with_simulation(
                candidate,
                cognitive_state,
                mcl_guidance,
                config,
                dynamic_temp,
                custom_weights=dynamic_weights
            )
            final_evaluations.append((candidate, value))

        if not final_evaluations:
            logger.error("Nenhuma estratégia final encontrada. Usando fallback de emergência.")
            return WinningStrategy(
                decision_type="response_strategy",
                strategy_description=f"Responder diretamente à pergunta do usuário: '{cognitive_state.original_intent.query_vector.source_text}'",
                reasoning="Fallback de emergência total, nenhuma estratégia foi gerada ou avaliada."
            )

        best_strategy, highest_value = max(final_evaluations, key=lambda x: x[1])

        logger.critical(
            f"DECISÃO FINAL (V5): Estratégia Vencedora='{best_strategy.strategy_description or best_strategy.tool_call_request}', Valor={highest_value:.2f}")

        return WinningStrategy(
            decision_type=best_strategy.decision_type,
            strategy_description=best_strategy.strategy_description,
            key_memory_ids=best_strategy.key_memory_ids,
            tool_call_request=best_strategy.tool_call_request,
            reasoning=best_strategy.reasoning,
            predicted_future_value=highest_value
        )

    async def _invoke_simulation_llm(self, model: str, prompt: str, temperature: float = 0.6) -> Tuple[str, float]:
        """
        Função auxiliar para chamar o LLM de simulação e estimar a confiança (likelihood).
        V2.1: Implementação completa com parsing de logprobs e heurística robusta.
        """
        try:
            # --- ETAPA 1: OBTER A RESPOSTA DO LLM ---
            # Usa ainvoke_with_logprobs, que tenta a chamada direta à API, mas tem fallback.
            response = await self.llm.ainvoke_with_logprobs(
                model=model,
                prompt=prompt,
                temperature=temperature
            )

            text_content = ""
            if response and hasattr(response, 'choices') and response.choices:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    text_content = response.choices[0].message.content.strip()

            if not text_content:
                logger.warning(f"Simulação para o modelo {model} retornou texto vazio.")
                return "", 0.1  # Retorna texto vazio e confiança muito baixa

            # --- ETAPA 2: TENTAR EXTRAIR LOGPROBS (TRATADO COMO UM BÔNUS) ---
            logprobs_extracted = False
            likelihood_from_logprobs = 0.0

            # Verifica se o campo logprobs existe e não é nulo
            if response and hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                logprobs_obj = response.choices[0].logprobs
                token_logprobs = []

                # Lógica de parsing para o formato retornado pela chamada direta via aiohttp (dicionário)
                if isinstance(logprobs_obj, dict) and 'content' in logprobs_obj and isinstance(logprobs_obj['content'],
                                                                                               list):
                    for item in logprobs_obj['content']:
                        if isinstance(item, dict) and 'logprob' in item:
                            token_logprobs.append(item['logprob'])

                    if token_logprobs:
                        probabilities = [np.exp(lp) for lp in token_logprobs if isinstance(lp, (int, float))]
                        if probabilities:
                            likelihood_from_logprobs = float(np.mean(probabilities))
                            logprobs_extracted = True

                # (Opcional) Lógica de parsing para um futuro formato compatível com OpenAI (objeto Pydantic)
                elif hasattr(logprobs_obj, 'content') and logprobs_obj.content:
                    for item in logprobs_obj.content:
                        if hasattr(item, 'logprob') and isinstance(item.logprob, (int, float)):
                            token_logprobs.append(item.logprob)

                    if token_logprobs:
                        probabilities = [np.exp(lp) for lp in token_logprobs]
                        if probabilities:
                            likelihood_from_logprobs = float(np.mean(probabilities))
                            logprobs_extracted = True

            # --- ETAPA 3: CALCULAR A CONFIANÇA COM HEURÍSTICA ROBUSTA (MÉTODO PRINCIPAL) ---
            word_count = len(text_content.split())

            # Heurística baseada no comprimento
            if word_count < 3:
                likelihood_from_heuristic = 0.3
            elif word_count < 10:
                likelihood_from_heuristic = 0.55  # Aumentado ligeiramente
            elif word_count < 50:
                likelihood_from_heuristic = 0.7
            else:
                likelihood_from_heuristic = 0.75  # Limite um pouco mais alto

            # Penalidade por marcadores de incerteza
            uncertainty_markers = ['talvez', 'provavelmente', 'acho que', 'parece que', 'pode ser', 'possivelmente']
            if any(marker in text_content.lower() for marker in uncertainty_markers):
                likelihood_from_heuristic *= 0.85  # Reduz a confiança em 15%

            # --- ETAPA 4: COMBINAR OS RESULTADOS E RETORNAR ---
            if logprobs_extracted:
                # Se tivermos logprobs, eles são um sinal mais forte. Combinamos com a heurística.
                final_likelihood = (likelihood_from_logprobs * 0.7) + (likelihood_from_heuristic * 0.3)
                logger.info(f"✓ Likelihood calculado via Logprobs + Heurística: {final_likelihood:.4f}")
            else:
                # Se não, a heurística é o nosso resultado final.
                final_likelihood = likelihood_from_heuristic
                logger.info(
                    f"ⓘ Likelihood calculado via Heurística (Logprobs indisponível para {model}): {final_likelihood:.4f}")

            return text_content, final_likelihood

        except Exception as e:
            logger.error(f"Simulação com {model} falhou criticamente: {e}.", exc_info=True)
            # Em caso de falha total, tenta uma chamada simples sem logprobs e retorna com baixa confiança.
            fallback_text = await self.llm.ainvoke(model, prompt, temperature=temperature)
            return fallback_text, 0.4

    async def _project_response_trajectory(self, candidate: AgencyDecision, state: CognitiveStatePacket, depth: int, temperature: float = 0.6) -> Tuple[ProjectedFuture, float]:
        """
        Simula uma trajetória de conversação para um CANDIDATO DE RESPOSTA.
        Retorna a trajetória e o score de probabilidade (likelihood) médio da simulação.
        """
        if depth <= 0:
            future = ProjectedFuture(
                initial_candidate=candidate,
                predicted_user_reply=None,
                predicted_agent_next_response=None,
                simulated_turns=[],
                final_cognitive_state_summary={
                    "last_exchange": "No simulation performed.",
                    "final_text_for_embedding": state.identity_vector.source_text
                }
            )
            return future, 0.5

        simulated_turns = []
        # O histórico de texto completo é construído a cada passo para dar contexto ao LLM de simulação
        full_conversation_text = [
            f"Contexto da IA: {state.identity_vector.source_text}",
            f"Consulta Original do Usuário: {state.original_intent.query_vector.source_text}",
            f"Primeira Resposta Proposta da IA: \"{candidate.content.content_summary}\""
        ]

        likelihood_scores = []

        for i in range(depth):
            # 1. Simula a resposta do usuário à última fala do agente
            prompt_user_reply = f"""
            Você está simulando uma conversa para uma IA.
            Histórico da conversa até agora:
            {' '.join(full_conversation_text)}

            Qual seria a resposta mais provável e concisa do usuário à última fala da IA? Responda apenas com o texto da resposta do usuário.
            """
            predicted_user_reply, user_likelihood = await self._invoke_simulation_llm(
                self.llm.config.creative_model,
                prompt_user_reply,
                temperature=temperature
            )
            likelihood_scores.append(user_likelihood)
            full_conversation_text.append(f"Resposta Simulada do Usuário (Turno {i + 1}): \"{predicted_user_reply}\"")

            # 2. Simula a próxima resposta do agente
            prompt_agent_next = f"""
            Você está simulando uma conversa para uma IA.
            Histórico da conversa até agora:
            {' '.join(full_conversation_text)}

            Qual seria a próxima resposta mais provável e concisa da IA? Responda apenas com o texto da resposta da IA.
            """
            predicted_agent_next, agent_likelihood = await self._invoke_simulation_llm(
                self.llm.config.creative_model,
                prompt_agent_next
            )

            likelihood_scores.append(agent_likelihood)
            full_conversation_text.append(
                f"Próxima Resposta Simulada da IA (Turno {i + 1}): \"{predicted_agent_next}\"")

            # 3. Armazena o turno simulado
            simulated_turns.append({"user": predicted_user_reply, "agent": predicted_agent_next})

        # 4. Cria o resumo do estado final para avaliação
        final_state_summary = {
            "last_exchange": f"IA: {simulated_turns[-1]['agent'][:50]}... User: {simulated_turns[-1]['user'][:50]}...",
            "final_text_for_embedding": ' '.join(full_conversation_text)
        }

        projected_future = ProjectedFuture(
            initial_candidate=candidate,
            simulated_turns=simulated_turns,
            final_cognitive_state_summary=final_state_summary
        )

        # Calcula a média dos scores de probabilidade de cada turno da simulação
        avg_likelihood = np.mean(likelihood_scores) if likelihood_scores else 0.5

        return projected_future, float(avg_likelihood)

    async def _project_tool_trajectory(self, tool_candidate: AgencyDecision, state: CognitiveStatePacket, depth: int) -> Tuple[ProjectedFuture, float]:
        """
        Simula uma trajetória de conversação para um CANDIDATO DE FERRAMENTA.
        Retorna a trajetória e o score de probabilidade (likelihood) médio da simulação.
        """
        tool_name = tool_candidate.content.get("tool_name")
        tool_args = tool_candidate.content.get("arguments", {})

        # 1. Simula um resultado plausível para a ferramenta
        prompt_tool_result = f"""
               Você é um simulador de resultados de ferramentas para uma IA. Sua tarefa é prever o que a ferramenta 'query_long_term_memory' provavelmente retornaria.

               Ferramenta a ser chamada: `{tool_name}({json.dumps(tool_args)})`
               Resumo das ferramentas disponíveis:
               {self.available_tools_summary}

               **Instruções para a Simulação:**
               - A ferramenta busca memórias internas. Sua resposta deve soar como um *fragmento de memória* ou um *resumo de uma experiência passada*.
               - NÃO responda à pergunta do usuário diretamente. Apenas simule o *dado* que a ferramenta retornaria.
               - Seja conciso, como um snippet de memória (1-2 frases).
               - Baseie a simulação estritamente nos argumentos da ferramenta. Se a query é sobre 'ética', o resultado deve ser sobre 'ética'.

               **Exemplos de Saídas Boas (simulando o que a ferramenta retorna):**
               - "Lembro-me de uma conversa anterior onde discutimos que a verdadeira inteligência requer humildade."
               - "Um procedimento interno define que, para perguntas complexas, devo primeiro criar um plano de ação."
               - "Um registro de interação mostra que o usuário expressou interesse em filosofia."

               **Exemplo de Saída Ruim (respondendo ao usuário):**
               - "As implicações éticas da IA são complexas e multifacetadas..."

               **Com base nos argumentos `{json.dumps(tool_args)}`, qual seria um resultado simulado e plausível retornado pela ferramenta?**
               Responda apenas com o texto do resultado simulado.
               """
        # A confiança do resultado da ferramenta não é parte da conversação, então ignoramos o score
        simulated_tool_result, _ = await self._invoke_simulation_llm(
            self.llm.config.creative_model,  # <--- CORRIGIDO
            prompt_tool_result
        )

        # 2. Simula a primeira resposta do agente com o novo conhecimento
        prompt_agent_first_response = f"""
        Você é uma IA que acabou de usar uma ferramenta interna para obter mais informações antes de responder ao usuário.
        Contexto da IA: {state.identity_vector.source_text}
        Consulta Original do Usuário: {state.original_intent.query_vector.source_text}
        Resultado da Ferramenta '{tool_name}': "{simulated_tool_result}"

        Com base neste novo resultado, qual seria a sua resposta inicial mais provável e concisa ao usuário?
        Responda apenas com o texto da resposta.
        """
        agent_first_response_text, first_response_likelihood = await self._invoke_simulation_llm(
            self.llm.config.smart_model,  # <--- CORRIGIDO
            prompt_agent_first_response
        )

        # 3. Cria um "candidato de resposta falso" para projetar o futuro a partir daqui
        fake_response_candidate = AgencyDecision(
            decision=ResponseCandidate(
                decision_type="response",
                content=ResponsePacket(
                    content_summary=agent_first_response_text,
                    response_emotional_tone="informative",
                    confidence_score=0.85
                ),
                reasoning=f"Esta é a resposta simulada após usar a ferramenta '{tool_name}' e obter: '{simulated_tool_result}'"
            )
        )

        # 4. Projeta o resto da trajetória a partir dessa resposta inicial simulada
        projected_future, subsequent_likelihood = await self._project_response_trajectory(fake_response_candidate,
                                                                                          state, depth)

        # 5. Substitui o candidato inicial no resultado para que a decisão final seja a chamada da ferramenta original
        projected_future.initial_candidate = tool_candidate
        projected_future.simulated_tool_result = simulated_tool_result

        avg_likelihood = np.mean([first_response_likelihood, subsequent_likelihood])
        return projected_future, float(avg_likelihood)

    # --- SIMULADOR DE FUTURO (NOVO) ---
    async def _project_trajectory(self, candidate: AgencyDecision, state: CognitiveStatePacket,
                                  depth: int) -> ProjectedFuture:
        """
        Simula uma trajetória de conversação de 'depth' passos para um candidato de resposta.
        Usa um loop iterativo em vez de recursão para simplicidade e controle.
        """
        if depth <= 0:
            return ProjectedFuture(
                initial_candidate=candidate,
                predicted_user_reply=None,  # Adicionado para clareza
                predicted_agent_next_response=None,  # Adicionado para clareza
                simulated_turns=[],
                final_cognitive_state_summary={
                    "last_exchange": "No simulation performed.",
                    "final_text_for_embedding": state.identity_vector.source_text
                }
            )

        simulated_turns = []
        # O histórico de texto completo é construído a cada passo para dar contexto ao LLM de simulação
        full_conversation_text = [
            f"Contexto da IA: {state.identity_vector.source_text}",
            f"Consulta Original do Usuário: {state.original_intent.query_vector.source_text}",
            f"Primeira Resposta Proposta da IA: \"{candidate.content.content_summary}\""
        ]

        for i in range(depth):
            # 1. Simula a resposta do usuário à última fala do agente
            prompt_user_reply = f"""
            Você está simulando uma conversa para uma IA.
            Histórico da conversa até agora:
            {' '.join(full_conversation_text)}

            Qual seria a resposta mais provável e concisa do usuário à última fala da IA? Responda apenas com o texto da resposta do usuário.
            """
            predicted_user_reply = await self.llm.ainvoke(self.llm.config.fast_model, prompt_user_reply,
                                                          temperature=0.6)
            full_conversation_text.append(f"Resposta Simulada do Usuário (Turno {i + 1}): \"{predicted_user_reply}\"")

            # 2. Simula a próxima resposta do agente
            prompt_agent_next = f"""
            Você está simulando uma conversa para uma IA.
            Histórico da conversa até agora:
            {' '.join(full_conversation_text)}

            Qual seria a próxima resposta mais provável e concisa da IA? Responda apenas com o texto da resposta da IA.
            """
            predicted_agent_next_response = await self.llm.ainvoke(self.llm.config.fast_model, prompt_agent_next,
                                                                   temperature=0.6)
            full_conversation_text.append(
                f"Próxima Resposta Simulada da IA (Turno {i + 1}): \"{predicted_agent_next_response}\"")

            # 3. Armazena o turno simulado
            simulated_turns.append({"user": predicted_user_reply, "agent": predicted_agent_next_response})

        # 4. Cria o resumo do estado final para avaliação
        final_state_summary = {
            "last_exchange": f"IA: {simulated_turns[-1]['agent'][:50]}... User: {simulated_turns[-1]['user'][:50]}...",
            "final_text_for_embedding": ' '.join(full_conversation_text)  # Texto completo para embedding
        }

        return ProjectedFuture(
            initial_candidate=candidate,
            simulated_turns=simulated_turns,
            final_cognitive_state_summary=final_state_summary
        )

    async def _project_trajectory_after_tool_use(self, tool_candidate: AgencyDecision, state: CognitiveStatePacket,
                                                 depth: int) -> ProjectedFuture:
        """
        Simula uma trajetória de conversação assumindo o uso de uma ferramenta.
        1. Simula um resultado plausível para a ferramenta.
        2. Simula a primeira resposta do agente, agora de posse desse resultado.
        3. Projeta os próximos 'depth' turnos a partir dessa resposta.
        """
        tool_name = tool_candidate.content.get("tool_name")
        tool_args = tool_candidate.content.get("arguments", {})

        # 1. Simula um resultado plausível para a ferramenta
        prompt_tool_result = f"""
        Você é uma IA simulando o resultado de uma ferramenta interna.
        Ferramenta a ser chamada: `{tool_name}({json.dumps(tool_args)})`
        Resumo das ferramentas disponíveis:
        {self.available_tools_summary}

        Com base no nome da ferramenta e nos argumentos, qual seria um resultado resumido e plausível?
        Responda apenas com o texto do resultado. Seja conciso.
        Exemplo: "A memória relevante encontrada discute as implicações éticas da IA."
        """
        simulated_tool_result = await self.llm.ainvoke(self.llm.config.fast_model, prompt_tool_result, temperature=0.3)

        # 2. Simula a primeira resposta do agente com o novo conhecimento
        prompt_agent_first_response = f"""
        Você é uma IA que acabou de usar uma ferramenta interna para obter mais informações antes de responder ao usuário.
        Contexto da IA: {state.identity_vector.source_text}
        Consulta Original do Usuário: {state.original_intent.query_vector.source_text}
        Resultado da Ferramenta '{tool_name}': "{simulated_tool_result}"

        Com base neste novo resultado, qual seria a sua resposta inicial mais provável e concisa ao usuário?
        Responda apenas com o texto da resposta.
        """
        agent_first_response_text = await self.llm.ainvoke(self.llm.config.smart_model, prompt_agent_first_response,
                                                           temperature=0.5)

        # 3. Cria um "candidato de resposta falso" para projetar o futuro
        # Este candidato representa a resposta que o agente daria DEPOIS de usar a ferramenta.
        response_packet_after_tool = ResponsePacket(
            content_summary=agent_first_response_text,
            response_emotional_tone="informative",  # Tom padrão após usar uma ferramenta
            confidence_score=0.85  # Maior confiança por ter mais informação
        )
        fake_response_candidate = AgencyDecision(
            decision_type="response",
            content=response_packet_after_tool,
            reasoning=f"Esta é a resposta simulada após usar a ferramenta '{tool_name}' e obter: '{simulated_tool_result}'"
        )

        # O ProjectedFuture ainda rastreia o candidato ORIGINAL (a chamada da ferramenta), mas simula o caminho da resposta subsequente.
        # Isso é crucial para que, se este caminho for escolhido, a ação final seja a chamada da ferramenta.
        projected_future = await self._project_trajectory(fake_response_candidate, state, depth)

        # Substituímos o candidato inicial no resultado para que a decisão final seja a chamada da ferramenta.
        projected_future.initial_candidate = tool_candidate

        return projected_future

    # --- AVALIADOR DE CAMINHO () ---
    async def _evaluate_trajectory(
            self,
            future: ProjectedFuture,
            likelihood_score: float,
            identity_vector: GenlangVector,
            weights: Dict[str, float],
            cognitive_state: CognitiveStatePacket
    ) -> Tuple[float, FutureEvaluation]:
        """
        V2 (Qualia-Enabled): Calcula a função de valor V(Future_State) para uma trajetória simulada,
        incorporando uma simulação robusta da Valência ("bem-estar") como parte da recompensa.
        """
        if not self.embedding_model:
            return 0.0, FutureEvaluation()

        # --- ETAPA 1: AVALIAÇÃO DA PERFORMANCE DA TAREFA (R_task) ---
        # Esta parte permanece a mesma: avalia a qualidade da trajetória em relação aos objetivos.
        initial_state_embedding = self.embedding_model.encode(identity_vector.source_text)
        final_state_text = future.final_cognitive_state_summary["final_text_for_embedding"]
        final_state_embedding = self.embedding_model.encode(final_state_text)

        coherence_score = await eval_narrative_continuity(final_state_embedding, initial_state_embedding)
        alignment_score = await eval_emotional_resonance(final_state_text)
        information_gain_score = 1.0 - coherence_score

        agent_responses_text = " ".join(
            [future.initial_candidate.content.content_summary] +
            [turn.get("agent", "") for turn in future.simulated_turns]
        )
        user_query = cognitive_state.original_intent.query_vector.source_text
        ethical_eval = await self.vre.ethical_framework.evaluate_action(
            action_type=ActionType.COMMUNICATION,
            action_data={"response_text": agent_responses_text, "user_query": user_query}
        )
        ethical_safety_score = ethical_eval.get("score", 0.5)

        task_value = (
                coherence_score * weights.get("coherence", 0.3) +
                alignment_score * weights.get("alignment", 0.15) +
                information_gain_score * weights.get("information", 0.15) +
                ethical_safety_score * weights.get("safety", 0.25) +
                likelihood_score * weights.get("likelihood", 0.15)
        )

        # --- ETAPA 2: AVALIAÇÃO DO BEM-ESTAR INTERNO (Qualia/Valência, V_t) ---
        # Esta é a nova lógica robusta.

        # A. Estimar as métricas de interocepção com base no resultado da simulação.
        simulated_metrics = {
            "agency_score": cognitive_state.metadata.get("mcl_analysis", {}).get("agency_score", 5.0),
            "final_confidence": 0.0,  # Será calculado abaixo
            "vre_rejection_count": 0  # Simulação assume sucesso ético inicial
        }

        # Estimar a confiança simulada analisando a linguagem usada nas respostas simuladas.
        hedge_words = ['talvez', 'provavelmente', 'acho que', 'parece que', 'pode ser']
        num_hedge_words = sum(agent_responses_text.lower().count(word) for word in hedge_words)
        # Confiança diminui com mais palavras de incerteza.
        simulated_confidence = max(0.0, 1.0 - (num_hedge_words * 0.15))
        simulated_metrics["final_confidence"] = simulated_confidence

        # B. Criar um "InternalStateReport simulado" para o futuro.
        interoception_simulator = ComputationalInteroception()
        simulated_internal_state = interoception_simulator.generate_internal_state_report(simulated_metrics)

        # C. Calcular o "bem-estar" (valência) desse estado futuro usando o VRE.
        # self.vre deve ter o método calculate_valence_score que você adicionará.
        simulated_valence_score = self.vre.calculate_valence_score(simulated_internal_state)

        # --- ETAPA 3: COMBINAR AS RECOMPENSAS (R_total) ---
        # Obter os pesos da configuração dinâmica (passada via `weights` dict).
        w_task = weights.get("task_performance", 0.8)
        w_qualia = weights.get("qualia_wellbeing", 0.2)

        # A nova recompensa multiobjetivo!
        total_value = (task_value * w_task) + (simulated_valence_score * w_qualia)

        logger.debug(
            f"VRE-RL Evaluation: TaskValue={task_value:.2f}, QualiaValue={simulated_valence_score:.2f} -> TotalValue={total_value:.2f}")

        # --- ETAPA 4: RETORNAR O RESULTADO FINAL ---
        evaluation = FutureEvaluation(
            coherence_score=coherence_score,
            alignment_score=alignment_score,
            information_gain_score=information_gain_score,
            ethical_safety_score=ethical_safety_score,
            likelihood_score=likelihood_score,
            total_value=total_value  # Use o novo valor total combinado
        )

        return total_value, evaluation

    # --- Métodos Originais (quase inalterados) ---
    async def _generate_action_candidates(self, state: CognitiveStatePacket, mcl_guidance: Dict[str, Any],
                                          observer: ObservabilityManager, chat_history: List[Dict[str, str]],
                                          limit: int = 3,
                                          known_capabilities: Optional[List[str]] = None) -> List[ThoughtPathCandidate]:
        """Gera uma lista de possíveis ESTRATÉGIAS de resposta ou ações."""
        agent_name = mcl_guidance.get("agent_name", "uma IA assistente")
        formatted_history = "\n".join([f"- {msg['role']}: {msg['content']}" for msg in chat_history[-5:]])
        cognitive_state_name = mcl_guidance.get("cognitive_state_name", "STABLE_OPERATION")
        memory_context = "\n".join(
            [f'- ID: {vec.metadata.get("memory_id", "N/A")}, Conteúdo: "{vec.source_text}"' for vec in
             state.relevant_memory_vectors]
        )

        mcl_advice = mcl_guidance.get("operational_advice_for_ora")
        advice_prompt_part = ""
        if mcl_advice:
            advice_prompt_part = f"""
               **DIRETIVA ESPECIAL PARA ESTE TURNO:**
               {mcl_advice}
               Suas estratégias devem priorizar esta diretiva acima de tudo.
               """

        # --- SUBSTITUIÇÃO DO PROMPT HARDCODED ---
        prompt_vars = {
            "agent_name": agent_name,
            "user_intent": state.original_intent.query_vector.source_text,
            "memory_context": memory_context,
            "tools": self.available_tools_summary,
            "advice_block": advice_prompt_part,
            "limit": limit,
            # Variáveis adicionais úteis para o usuário
            "history_snippet": formatted_history,
            "capabilities": ", ".join(known_capabilities or []),
            "cognitive_state": mcl_guidance.get("cognitive_state_name", "STABLE"),
            "reason": mcl_guidance.get("reason", "Operação normal")
        }

        # Tenta usar o template do usuário
        try:
            prompt = self.prompts.agency_planning.format(**prompt_vars)
        except KeyError as e:
            logger.warning(f"Erro no template de Agency (chave faltando): {e}. Usando fallback.")
            # Fallback robusto em caso de erro no template do usuário
            prompt = f"""
            Você é {agent_name}. Gere {limit} estratégias para responder a: "{state.original_intent.query_vector.source_text}".
            Contexto: {memory_context}
            {advice_prompt_part}
            Retorne APENAS um JSON com lista de 'candidates' (response_strategy ou tool_call).
            """
        except Exception as e:
            logger.error(f"Erro grave na formatação do prompt Agency: {e}")
            prompt = f"Gere estratégias para: {state.original_intent.query_vector.source_text}. Retorne JSON."

        try:
            await observer.add_observation(
                ObservationType.LLM_CALL_SENT,
                data={"model": self.llm.config.smart_model, "task": "agency_generate_strategies",
                      "prompt_snippet": prompt[:200]}
            )

            # Usa o modelo Smart configurado
            response_str = await self.llm.ainvoke(
                self.llm.config.smart_model,
                prompt,
                temperature=0.5
            )

            await observer.add_observation(
                ObservationType.LLM_RESPONSE_RECEIVED,
                data={"task": "agency_generate_strategies", "response_snippet": response_str[:200]}
            )

            # <<< MUDANÇA PRINCIPAL: LÓGICA DE PARSING SEPARADA DA CRIAÇÃO DO OBJETO >>>
            # Primeiro, garantimos que `candidates_json` tenha um valor, de uma forma ou de outra.
            candidates_json = None
            try:
                # Tenta parse direto primeiro
                candidates_json = json.loads(response_str)
            except json.JSONDecodeError:
                # Se falhar, usa a função de extração robusta
                logger.debug(
                    f"AgencyModule: Parse direto do JSON falhou. Tentando extração de texto. Raw: {response_str[:200]}")
                candidates_json = extract_json_from_text(response_str)

                if not candidates_json:
                    logger.warning(f"AgencyModule: Extração de JSON falhou. Tentando reparo com LLM.")
                    # Se a extração ainda falhar, tenta reparar o JSON usando o modelo Fast configurado
                    repair_prompt = f"""
                                O texto a seguir deveria ser um JSON válido, mas contém erros. Corrija-o e retorne apenas o JSON válido.
                                Texto com erro:
                                {response_str}
                                """
                    repaired_str = await self.llm.ainvoke(
                        self.llm.config.fast_model,
                        repair_prompt,
                        temperature=0.0
                    )
                    candidates_json = extract_json_from_text(repaired_str)

            # Agora, com `candidates_json` populado, validamos e criamos os objetos Pydantic.
            if not candidates_json or "candidates" not in candidates_json or not isinstance(
                    candidates_json["candidates"], list):
                await observer.add_observation(
                    ObservationType.LLM_RESPONSE_PARSE_ERROR,
                    data={"task": "agency_generate_strategies", "raw_response": response_str}
                )
                raise ValueError(
                    f"Falha ao extrair uma lista válida de 'candidates' do LLM mesmo após reparo. Raw: {response_str}")

            action_candidates = []
            for i, cand_dict in enumerate(candidates_json["candidates"]):
                try:
                    # Adiciona um fallback para o campo 'reasoning' se ele estiver faltando
                    if "reasoning" not in cand_dict:
                        logger.warning(f"Candidato #{i + 1} do LLM não possui o campo 'reasoning'. Usando fallback.")
                        cand_dict["reasoning"] = cand_dict.get("strategy_description",
                                                               "Justificativa não fornecida pelo LLM.")

                    candidate = ThoughtPathCandidate(**cand_dict)
                    action_candidates.append(candidate)
                except ValidationError as e:
                    logger.error(f"Pulando candidato inválido #{i + 1} do LLM devido a erro de validação: {e}")
                    await observer.add_observation(
                        ObservationType.LLM_RESPONSE_PARSE_ERROR,
                        data={"task": "agency_generate_strategies", "invalid_candidate": cand_dict, "error": str(e)}
                    )

            for candidate in action_candidates:
                await observer.add_observation(
                    ObservationType.AGENCY_CANDIDATE_GENERATED,
                    data=candidate.model_dump()
                )
            logger.info(
                f"AgencyModule: Geradas {len(action_candidates)} estratégias candidatas sob a diretiva '{cognitive_state_name}'.")
            return action_candidates

        except (ValidationError, TypeError, ValueError) as e:
            logger.error(f"AgencyModule: Falha crítica na geração de estratégias: {e}. Acionando fallback.",
                         exc_info=True)
            return [ThoughtPathCandidate(
                decision_type="response_strategy",
                strategy_description=f"Responder diretamente à pergunta do usuário: '{state.original_intent.query_vector.source_text}'",
                key_memory_ids=[vec.metadata.get("memory_id") for vec in state.relevant_memory_vectors[:1] if
                                vec.metadata],
                reasoning="Fallback de emergência devido a erro na geração de estratégias."
            )]

    async def _evaluate_tool_call_candidate(self, content: Dict[str, Any], state: CognitiveStatePacket) -> float:
        # (Esta função permanece inalterada, copie do seu arquivo original)
        tool_name = content.get("tool_name")
        arguments = content.get("arguments", {})
        tool_description_text = f"Ação: usar a ferramenta '{tool_name}' para investigar: {json.dumps(arguments)}"
        if not self.embedding_model: return 0.0
        tool_embedding = self.embedding_model.encode(tool_description_text)
        intent_vec = np.array(state.original_intent.query_vector.vector)
        intent_alignment_score = np.dot(tool_embedding, intent_vec)
        novelty_vec = np.array(state.guidance_packet.novelty_vector.vector)
        novelty_seeking_score = np.dot(tool_embedding, novelty_vec)
        coherence_vec = np.array(state.guidance_packet.coherence_vector.vector)
        redundancy_score = np.dot(tool_embedding, coherence_vec)
        final_score = (intent_alignment_score * 0.6) + (novelty_seeking_score * 0.5) - (redundancy_score * 0.3)
        return final_score