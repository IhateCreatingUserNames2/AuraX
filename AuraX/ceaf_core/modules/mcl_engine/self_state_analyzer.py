# Self State Analyzer
# ceaf_project/ceaf_core/modules/mcl_engine/self_state_analyzer.py

import logging
import time
import statistics  # For potential numerical analysis
from typing import List, Dict, Any, Tuple, Optional
import re # For basic text processing
import numpy as np
import asyncio
from nltk.tokenize import sent_tokenize
from ceaf_core.utils.embedding_utils import get_embedding_client, cosine_similarity_np

from pydantic import json, BaseModel, Field
embedding_client = get_embedding_client()

# Constants from mcl_callbacks might be useful for key names
MCL_OBSERVATIONS_LIST_KEY = "mcl:ora_turn_observations_log"  # From mcl_callbacks.py

logger = logging.getLogger(__name__)

# --- Configuration for Analysis ---
# Thresholds and parameters for heuristics (these are examples, will need tuning)
CONFIG_REPETITIVE_RESPONSE_THRESHOLD = 0.8  # If >80% similar (placeholder metric)
CONFIG_HIGH_TOOL_ERROR_RATE_THRESHOLD = 0.5  # If >50% of tool calls in a turn failed
CONFIG_MAX_FUNCTION_CALLS_PER_TURN_SOFT_LIMIT = 3  # Warning if ORA uses too many tools

# NCF Parameters (example, should align with what ncf_tool and MCL_Agent use)
NCF_CONCEPTUAL_ENTROPY = "conceptual_entropy"
NCF_NARRATIVE_DEPTH = "narrative_depth"
NCF_PHILOSOPHICAL_FRAMING = "philosophical_framing_intensity"


class ORAStateAnalysis(BaseModel):
    """
    Holds the structured analysis of ORA's state for a given turn or period.
    """
    turn_id: str
    assessment_timestamp: float = Field(default_factory=time.time)

    # Overall State Assessment (Qualitative)
    eoc_assessment: str = "indeterminate"
    eoc_confidence: float = 0.0
    summary_notes: List[str] = Field(default_factory=list) # Atributo de classe, não em __init__

    # Quantitative EoC Scores
    novelty_score: Optional[float] = None
    coherence_score: Optional[float] = None
    grounding_score: Optional[float] = None
    eoc_quantitative_score: Optional[float] = None

    # Specific Metrics/Observations
    llm_requests_count: int = 0
    llm_responses_count: int = 0
    tool_calls_attempted: int = 0
    tool_calls_succeeded: int = 0
    tool_errors: List[Dict[str, Any]] = Field(default_factory=list)
    function_calls_in_last_response: List[str] = Field(default_factory=list)
    last_llm_finish_reason: Optional[str] = None
    last_response_text_snippet: Optional[str] = None

    # Heuristic flags
    flags: Dict[str, bool] = Field(default_factory=lambda: {
        "excessive_tool_use": False,
        "high_tool_error_rate": False,
        "potential_loop_behavior": False,
        "response_unusually_short": False,
        "response_unusually_long": False,
        "low_novelty_suspected": False,
        "high_ungroundedness_suspected": False,
        "ncf_params_seem_misaligned": False,
        "low_coherence_suspected": False,
    })

    raw_observations_count: int = 0

    def add_note(self, note: str):
        self.summary_notes.append(note)


async def calculate_novelty_score(
        response_text: str,
        context_texts: List[str],
        previous_response_text: Optional[str] = None
) -> float:
    """
    Calcula a novidade semântica de uma resposta em relação ao seu contexto.
    A novidade é a distância semântica (1 - similaridade) do ponto mais próximo no contexto.
    """
    if not response_text:
        return 0.0

    # Gera todos os embeddings em paralelo para eficiência
    texts_to_embed = [response_text] + context_texts
    if previous_response_text:
        texts_to_embed.append(previous_response_text)

    embeddings = await embedding_client.get_embeddings(texts_to_embed, context_type="default_query")

    response_emb = np.array(embeddings[0])
    context_embs = [np.array(e) for e in embeddings[1:1 + len(context_texts)]]

    # Calcula a similaridade máxima com qualquer parte do contexto
    max_similarity_with_context = 0.0
    if context_embs:
        similarities = [cosine_similarity_np(response_emb, ctx_emb) for ctx_emb in context_embs]
        max_similarity_with_context = max(similarities) if similarities else 0.0

    # A novidade é o inverso da similaridade máxima
    novelty_score = 1.0 - max_similarity_with_context

    # Penaliza fortemente a repetição da resposta anterior (sinal de loop)
    if previous_response_text:
        prev_response_emb = np.array(embeddings[-1])
        similarity_with_prev = cosine_similarity_np(response_emb, prev_response_emb)
        if similarity_with_prev > 0.95:  # Quase idêntico semanticamente
            return 0.0
        # Reduz a novidade proporcionalmente à similaridade com a resposta anterior
        novelty_score *= (1.0 - (similarity_with_prev * 0.5))

    return max(0.0, min(1.0, novelty_score))

async def calculate_coherence_score(
    response_text: str,
    query_text: Optional[str] = None,
    ncf_summary_text: Optional[str] = None
) -> float:
    """
    Calcula a coerência semântica de uma resposta com a consulta do usuário e o contexto geral.
    É uma média ponderada da relevância para a consulta e alinhamento com o tema.
    """
    if not response_text:
        return 0.0

    texts_to_embed = [response_text]
    query_index, ncf_index = -1, -1
    if query_text:
        texts_to_embed.append(query_text)
        query_index = len(texts_to_embed) - 1
    if ncf_summary_text:
        texts_to_embed.append(ncf_summary_text)
        ncf_index = len(texts_to_embed) - 1

    if len(texts_to_embed) == 1:
        return 0.5 # Não há com o que comparar a coerência

    embeddings = await embedding_client.get_embeddings(texts_to_embed, context_type="default_query")
    response_emb = np.array(embeddings[0])

    query_similarity = 0.0
    if query_index != -1:
        query_emb = np.array(embeddings[query_index])
        query_similarity = cosine_similarity_np(response_emb, query_emb)

    ncf_similarity = 0.0
    if ncf_index != -1:
        ncf_emb = np.array(embeddings[ncf_index])
        ncf_similarity = cosine_similarity_np(response_emb, ncf_emb)

    # A coerência é principalmente sobre responder à consulta, mas também se manter no tema.
    # Peso maior para a similaridade com a consulta.
    coherence_score = (query_similarity * 0.7) + (ncf_similarity * 0.3)

    return max(0.0, min(1.0, coherence_score))


async def calculate_grounding_score(
        response_text: str,
        supporting_memory_snippets: Optional[List[str]] = None
) -> float:
    """
    Calcula o quão bem uma resposta é fundamentada ("grounded") em memórias de suporte.
    Verifica, para cada sentença da resposta, se ela é semanticamente suportada por algum snippet de memória.
    """
    if not response_text:
        return 0.0
    if not supporting_memory_snippets:
        return 0.3  # Não se pode verificar, então a confiança é baixa

    response_sentences = sent_tokenize(response_text)
    if not response_sentences:
        return 0.0

    # Gera embeddings para todas as sentenças e snippets de uma vez
    all_texts = response_sentences + supporting_memory_snippets
    all_embeddings = await embedding_client.get_embeddings(all_texts,
                                                           context_type="explicit")  # Memórias são "explicit"

    sentence_embs = [np.array(e) for e in all_embeddings[:len(response_sentences)]]
    snippet_embs = [np.array(e) for e in all_embeddings[len(response_sentences):]]

    grounded_sentences_count = 0
    grounding_threshold = 0.75  # Limiar de similaridade para considerar uma sentença "suportada"

    for sent_emb in sentence_embs:
        # Encontra a maior similaridade desta sentença com qualquer um dos snippets de memória
        max_similarity_with_snippets = 0.0
        if snippet_embs:
            similarities = [cosine_similarity_np(sent_emb, snip_emb) for snip_emb in snippet_embs]
            max_similarity_with_snippets = max(similarities) if similarities else 0.0

        if max_similarity_with_snippets >= grounding_threshold:
            grounded_sentences_count += 1

    grounding_ratio = grounded_sentences_count / len(response_sentences)
    return grounding_ratio


async def analyze_ora_turn_observations(
        turn_id: str,
        turn_observations: List[Dict[str, Any]],
        # --- New arguments for quantitative scoring ---
        ora_response_text: Optional[str] = None,
        user_query_text: Optional[str] = None,
        ncf_text_summary: Optional[str] = None, # Summary or full NCF text
        retrieved_memory_snippets: Optional[List[str]] = None,
        previous_ora_response_text: Optional[str] = None, # For novelty comparison
        # --- End new arguments ---
        previous_analysis: Optional[ORAStateAnalysis] = None,
        current_ncf_params: Optional[Dict[str, Any]] = None
) -> ORAStateAnalysis:
    """
    Analyzes a list of observations for a specific ORA turn to assess its state.
    """
    analysis = ORAStateAnalysis(turn_id=turn_id)
    analysis.raw_observations_count = len(turn_observations)
    if ora_response_text:
        analysis.last_response_text_snippet = ora_response_text[:200] # Store snippet

    logger.info(
        f"MCL Analyzer: Starting analysis for turn '{turn_id}' with {analysis.raw_observations_count} observations.")

    if not turn_observations and not ora_response_text: # Need at least some data
        analysis.add_note("No observations or response text provided for this turn.")
        analysis.eoc_assessment = "indeterminate_no_data"
        return analysis

    last_llm_request_tools = []
    OBS_LLM_REQUEST = "llm_call_sent"
    OBS_LLM_RESPONSE = "llm_response_received"
    OBS_TOOL_ATTEMPT = "tool_call_attempted"
    OBS_TOOL_SUCCEEDED = "tool_call_succeeded"
    OBS_TOOL_FAILED = "tool_call_failed"  # NOVO
    for obs in turn_observations:
        obs_type = obs.get("observation_type")
        data = obs.get("data", {})

        # ATUALIZAÇÃO DOS NOMES DE EVENTOS:
        if obs_type == OBS_LLM_REQUEST:
            analysis.llm_requests_count += 1
            # Se for uma requisição de LLM, a flag 'tool_names' não é mais necessária
            # se todas as chamadas de ferramenta vierem de OBS_TOOL_ATTEMPT.
            # Se for relevante, data.get("tool_names") pode ser mantido.
        elif obs_type == OBS_LLM_RESPONSE:
            analysis.llm_responses_count += 1
            # Não temos mais a função de "tool calls in response" diretamente,
            # mas podemos usar a chave 'finish_reason'
            analysis.last_llm_finish_reason = data.get("finish_reason")

        elif obs_type == OBS_TOOL_ATTEMPT:
            analysis.tool_calls_attempted += 1
        elif obs_type == OBS_TOOL_SUCCEEDED:
            analysis.tool_calls_succeeded += 1
        elif obs_type == OBS_TOOL_FAILED:
            # Novo: se houver falha, registrar explicitamente.
            analysis.tool_errors.append({
                "tool_name": data.get("tool_name"),
                "summary": data.get("error_message", "Unknown error")
            })

        # Adiciona a observação do VRE como um potencial "sinal de erro"
        elif obs_type == "vre_assessment_received":
            if data.get("alignment") != "aligned":
                analysis.add_note(f"VRE signaled misalignment: {data.get('alignment')}")

        # --- Basic Heuristics (Otimizados) ---
    if analysis.tool_calls_attempted > 0:
        # Erros são agora registrados via OBS_TOOL_FAILED
        failed_count = len(analysis.tool_errors)
        error_rate = failed_count / analysis.tool_calls_attempted
        if error_rate >= CONFIG_HIGH_TOOL_ERROR_RATE_THRESHOLD:
            analysis.flags["high_tool_error_rate"] = True
            analysis.add_note(
                f"High tool error rate: {error_rate:.2f} ({len(analysis.tool_errors)}/{analysis.tool_calls_attempted} failed).")

    if analysis.tool_calls_attempted >= CONFIG_MAX_FUNCTION_CALLS_PER_TURN_SOFT_LIMIT:
        analysis.flags["excessive_tool_use"] = True
        analysis.add_note(f"Potentially excessive tool use: {analysis.tool_calls_attempted} calls in one turn.")

    if analysis.last_llm_finish_reason == "MAX_TOKENS":
        analysis.flags["response_unusually_long"] = True
        analysis.add_note("LLM response may have been truncated (MAX_TOKENS).")


    # --- Quantitative EoC Scoring ---
    if ora_response_text:
        context_for_novelty = []
        if ncf_text_summary: context_for_novelty.append(ncf_text_summary)
        if user_query_text: context_for_novelty.append(user_query_text)

        # ### CORREÇÃO: Usa await para as novas funções assíncronas ###
        analysis.novelty_score, analysis.coherence_score, analysis.grounding_score = await asyncio.gather(
            calculate_novelty_score(
                ora_response_text,
                context_texts=context_for_novelty,
                previous_response_text=previous_ora_response_text
            ),
            calculate_coherence_score(
                ora_response_text,
                query_text=user_query_text,
                ncf_summary_text=ncf_text_summary
            ),
            calculate_grounding_score(
                ora_response_text,
                supporting_memory_snippets=retrieved_memory_snippets
            )
        )

        analysis.add_note(
            f"Novelty: {analysis.novelty_score:.2f}, Coherence: {analysis.coherence_score:.2f}, Grounding: {analysis.grounding_score:.2f}")

        # Update flags based on quantitative scores
        if analysis.novelty_score is not None and analysis.novelty_score < 0.3:
            analysis.flags["low_novelty_suspected"] = True
        if analysis.coherence_score is not None and analysis.coherence_score < 0.4:
            analysis.flags["low_coherence_suspected"] = True
        if analysis.grounding_score is not None and analysis.grounding_score < 0.4:
            analysis.flags["high_ungroundedness_suspected"] = True

        # Combine scores (simple weighted average for placeholder)
        weights = {"novelty": 0.3, "coherence": 0.4, "grounding": 0.3}
        score_sum = 0
        weight_sum = 0
        if analysis.novelty_score is not None:
            score_sum += analysis.novelty_score * weights["novelty"]
            weight_sum += weights["novelty"]
        if analysis.coherence_score is not None:
            score_sum += analysis.coherence_score * weights["coherence"]
            weight_sum += weights["coherence"]
        if analysis.grounding_score is not None:
            score_sum += analysis.grounding_score * weights["grounding"]
            weight_sum += weights["grounding"]

        if weight_sum > 0:
            analysis.eoc_quantitative_score = score_sum / weight_sum
            analysis.add_note(f"Combined Quantitative EoC Score: {analysis.eoc_quantitative_score:.2f}")


    # --- Refined Qualitative EoC Assessment (using flags and quantitative scores) ---
    num_negative_flags = sum(
        1 for flag_name, flag_val in analysis.flags.items() if flag_val and flag_name != "ncf_params_seem_misaligned")

    if analysis.eoc_quantitative_score is not None:
        if analysis.eoc_quantitative_score < 0.3 or num_negative_flags >= 2 or analysis.flags["high_tool_error_rate"]:
            analysis.eoc_assessment = "chaotic_leaning"
            analysis.eoc_confidence = 0.7
            analysis.add_note("Low quantitative EoC score or multiple flags suggest instability.")
        elif analysis.eoc_quantitative_score < 0.5 or num_negative_flags == 1:
            analysis.eoc_assessment = "suboptimal_critical"
            analysis.eoc_confidence = 0.6
            analysis.add_note("Moderate quantitative EoC score or some flags indicate suboptimal performance.")
        elif analysis.eoc_quantitative_score >= 0.75 and num_negative_flags == 0:
            analysis.eoc_assessment = "critical_optimal"
            analysis.eoc_confidence = 0.8
            analysis.add_note("Good quantitative EoC score and no negative flags suggest optimal operation.")
        else: # Covers cases like 0.5-0.75 quantitative or good score but some minor flags
            analysis.eoc_assessment = "critical_nominal"
            analysis.eoc_confidence = 0.65
            analysis.add_note("Nominal operation based on quantitative scores and flags.")
    else: # Fallback to original heuristic if no quantitative scores available
        if num_negative_flags >= 2 or analysis.flags["high_tool_error_rate"]:
            analysis.eoc_assessment = "chaotic_leaning"
            analysis.eoc_confidence = 0.6
        elif num_negative_flags == 1:
            analysis.eoc_assessment = "suboptimal_critical"
            analysis.eoc_confidence = 0.5
        elif analysis.llm_responses_count == 0 and analysis.tool_calls_attempted == 0 and not ora_response_text:
            analysis.eoc_assessment = "indeterminate_no_action"
            analysis.eoc_confidence = 0.4
        else:
            analysis.eoc_assessment = "critical_nominal"
            analysis.eoc_confidence = 0.5

    # Ensure a note is added if default assessment remains "critical_nominal" without specific reasoning from above
    if analysis.eoc_assessment == "critical_nominal" and not any("operation" in note for note in analysis.summary_notes):
         analysis.add_note("Defaulting to nominal operation as no strong positive or negative signals were detected by current heuristics.")


    logger.info(
        f"MCL Analyzer: Finished analysis for turn '{turn_id}'. EoC Assessment: {analysis.eoc_assessment} (Confidence: {analysis.eoc_confidence:.2f})")
    logger.debug(f"MCL Analysis details for turn '{turn_id}': {analysis.model_dump_json(indent=2)}")
    return analysis

