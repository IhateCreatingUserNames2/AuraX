# Narrative Memory Synthesizer
# ceaf_project/ceaf_core/modules/memory_blossom/synthesizer.py
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import os  # For API keys

import litellm  # For direct LLM calls if needed for advanced synthesis
from dotenv import load_dotenv
from pydantic import json
from pathlib import Path

from .memory_types import (  # Assuming memory_types.py is in the same directory
    AnyMemoryType,
    BaseMemory,
    ExplicitMemory,
    EmotionalMemory,
    ProceduralMemory,
    FlashbulbMemory,
    SomaticMemory,
    LiminalMemory,
    GenerativeMemory,
    MemorySalience,
    EmotionalTag, EmotionalContext, ProceduralStep, ExplicitMemoryContent
)

# Ensure LiteLLM is configured for OpenRouter (usually via environment variables)
# OPENROUTER_API_KEY should be set in the environment.
# litellm.api_key = os.getenv("OPENROUTER_API_KEY") # Or specific provider key
# litellm.api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

logger = logging.getLogger(__name__)

# --- Configuration for Synthesizer ---
SYNTHESIZER_LLM_MODEL = os.getenv("MEMORY_SYNTHESIZER_MODEL",
                                  "openrouter/openai/gpt-4.1-mini")  # Fast model for synthesis
MAX_MEMORIES_FOR_DIRECT_SYNTHESIS = 10  # Limit for how many memories to feed directly to LLM
MAX_SYNTHESIZED_NARRATIVE_LENGTH_TOKENS = 1500  # Conceptual token limit for the output


class NarrativeContext:
    """
    Represents the context for which the narrative is being synthesized.
    """

    def __init__(self, current_query: Optional[str] = None, current_goal: Optional[str] = None,
                 desired_focus: Optional[List[str]] = None):
        self.current_query = current_query
        self.current_goal = current_goal  # e.g., "problem_solving", "creative_ideation", "self_reflection"
        self.desired_focus_keywords = desired_focus or []  # Keywords to emphasize


class SynthesizedNarrative:
    """
    Holds the output of the synthesis process.
    """

    def __init__(self, narrative_text: str, contributing_memory_ids: List[str], synthesis_method: str):
        self.narrative_text = narrative_text
        self.contributing_memory_ids = contributing_memory_ids
        self.synthesis_method = synthesis_method  # e.g., "template_basic", "llm_refined"
        self.timestamp = time.time()

    def __str__(self):
        return self.narrative_text


# --- Helper Functions for Formatting Individual Memory Types ---

def _format_explicit_memory(mem: ExplicitMemory) -> str:
    parts = []
    if mem.content.text_content:
        parts.append(f"Fact/Observation: {mem.content.text_content}")
    if mem.content.structured_data:
        parts.append(f"Structured Data: {json.dumps(mem.content.structured_data, indent=2, default=str)}")
    if mem.content.artifact_reference:
        parts.append(f"Related Artifact: {mem.content.artifact_reference}")
    if mem.confidence_score is not None:
        parts.append(f"(Confidence: {mem.confidence_score * 100:.0f}%)")
    return " ".join(parts)


def _format_emotional_memory(mem: EmotionalMemory) -> str:
    return (f"Emotional State: Felt {mem.primary_emotion.value} (Intensity: {mem.context.intensity:.1f}) "
            f"in response to '{mem.context.triggering_event_summary or 'an event'}'. "
            f"Associated with: {', '.join(mem.context.associated_stimuli) if mem.context.associated_stimuli else 'N/A'}.")


def _format_procedural_memory(mem: ProceduralMemory) -> str:
    steps_str = "\n".join([f"  - Step {s.step_number}: {s.description}" for s in mem.steps])
    return (f"Procedure '{mem.procedure_name}': To achieve '{mem.goal_description}'.\n"
            f"Triggered by: {', '.join(mem.trigger_conditions) if mem.trigger_conditions else 'N/A'}.\n"
            f"Steps:\n{steps_str}")


def _format_flashbulb_memory(mem: FlashbulbMemory) -> str:
    # Flashbulb memories are critical, so give them more prominence
    details = _format_explicit_memory(ExplicitMemory(
        source_type=mem.source_type,  # Dummy ExplicitMemory for formatting content
        content=mem.content_details,
        confidence_score=mem.vividness_score  # Use vividness as confidence
    ))
    return (f"**Key Event (Flashbulb Memory - Salience: {mem.salience.value}): '{mem.event_summary}'**\n"
            f"Significance: {mem.personal_significance or 'High impact event.'}\n"
            f"Details: {details}")


def _format_somatic_memory(mem: SomaticMemory) -> str:
    return (f"Somatic Marker: A '{mem.marker.marker_type}' (Intensity: {mem.marker.intensity:.1f}) "
            f"is associated with situations like '{mem.triggering_context_summary}'.")


def _format_liminal_memory(mem: LiminalMemory) -> str:
    fragments_str = "; ".join(
        [
            f.fragment_text or f"Image ({f.image_url_reference or 'unspecified'}) related to {', '.join(f.conceptual_tags)}"
            for f in mem.fragments]
    )
    return (f"Liminal Insight (Potential: {mem.potential_significance or 'emerging connections'}): "
            f"Fragments include: {fragments_str}.")


def _format_generative_memory(mem: GenerativeMemory) -> str:
    return (f"Generative Seed '{mem.seed_name}': A '{mem.seed_data.seed_type}' "
            f"for contexts like '{', '.join(mem.applicability_contexts)}'. "
            f"Usage: {mem.seed_data.usage_instructions or 'As applicable.'}")


MEMORY_FORMATTERS = {
    "explicit": _format_explicit_memory,
    "emotional": _format_emotional_memory,
    "procedural": _format_procedural_memory,
    "flashbulb": _format_flashbulb_memory,
    "somatic": _format_somatic_memory,
    "liminal": _format_liminal_memory,
    "generative": _format_generative_memory,
}


# --- Core Synthesizer Logic ---

def _select_and_order_memories_for_synthesis(
        retrieved_memories: List[AnyMemoryType],
        context: NarrativeContext,
        max_memories: int = MAX_MEMORIES_FOR_DIRECT_SYNTHESIS
) -> List[AnyMemoryType]:
    """
    Selects and orders memories based on relevance, salience, recency, and context.
    V1: Simple ordering by salience then recency. Focus on memories matching keywords.
    """
    # Filter by keywords first, if any
    focused_memories = []
    if context.desired_focus_keywords:
        for mem in retrieved_memories:
            if any(keyword.lower() in ' '.join(mem.keywords).lower() for keyword in context.desired_focus_keywords):
                focused_memories.append(mem)
        if not focused_memories:  # If no keyword match, use all retrieved but prioritize
            focused_memories = retrieved_memories
    else:
        focused_memories = retrieved_memories

    # Sort: Flashbulb > Critical Salience > High Salience > ... then by recency (newer first)
    # More complex sorting would consider narrative_thread_id, emotional impact for current goal etc.
    def sort_key(mem: BaseMemory):
        salience_order = {MemorySalience.CRITICAL: 0, MemorySalience.HIGH: 1, MemorySalience.MEDIUM: 2,
                          MemorySalience.LOW: 3}
        # Flashbulb memories are implicitly critical
        is_flashbulb = hasattr(mem, 'memory_type') and mem.memory_type == "flashbulb"
        effective_salience = MemorySalience.CRITICAL if is_flashbulb else mem.salience

        return (
            0 if is_flashbulb else 1,  # Flashbulb first
            salience_order.get(effective_salience, 99),
            -(mem.last_accessed_ts or mem.timestamp)  # Negative for newest first
        )

    sorted_memories = sorted(focused_memories, key=sort_key)
    logger.info(
        f"Memory Synthesizer: Selected {len(sorted_memories[:max_memories])} out of {len(retrieved_memories)} memories for synthesis based on context.")
    return sorted_memories[:max_memories]


async def synthesize_narrative_from_memories(
        retrieved_memories: List[AnyMemoryType],
        synthesis_context: NarrativeContext,
        use_llm_for_refinement: bool = True
) -> SynthesizedNarrative:
    """
    Takes a list of retrieved memories and synthesizes them into a coherent narrative string.
    """
    if not retrieved_memories:
        logger.info("Memory Synthesizer: No memories provided for synthesis.")
        return SynthesizedNarrative("No relevant memories found for the current context.", [], "no_memories")

    # 1. Select and Order Memories
    ordered_memories_for_synthesis = _select_and_order_memories_for_synthesis(
        retrieved_memories, synthesis_context
    )

    contributing_ids = [mem.memory_id for mem in ordered_memories_for_synthesis]

    # 2. Basic Templated Formatting
    formatted_parts = []
    for mem in ordered_memories_for_synthesis:
        formatter = MEMORY_FORMATTERS.get(mem.memory_type)
        if formatter:
            try:
                formatted_parts.append(formatter(mem))
            except Exception as e:
                logger.error(
                    f"Memory Synthesizer: Error formatting memory ID {mem.memory_id} of type {mem.memory_type}: {e}")
                formatted_parts.append(f"[Error formatting memory: {mem.memory_id}]")
        else:
            logger.warning(
                f"Memory Synthesizer: No formatter found for memory type: {mem.memory_type}. Using generic format.")
            formatted_parts.append(
                f"Recalled ({mem.memory_type}, Salience: {mem.salience.value}): {str(mem.model_dump(exclude={'embedding_reference'}, exclude_none=True))[:200]}...")

    templated_narrative = "\n\n".join(formatted_parts)
    synthesis_method = "template_basic"

    # 3. LLM Refinement (Optional)
    if use_llm_for_refinement and templated_narrative:
        logger.info(
            f"Memory Synthesizer: Attempting LLM refinement for narrative. Input length: {len(templated_narrative)} chars.")
        prompt = f"""
You are a Narrative Synthesizer. Your task is to take the following collection of formatted memories and weave them into a single, coherent, and contextually relevant narrative paragraph or a few short paragraphs.
The narrative should be useful for an AI agent (ORA) to understand its relevant past experiences and knowledge for an upcoming interaction.

Current Context for ORA:
- User Query (if any): {synthesis_context.current_query or "Not specified."}
- ORA's Goal (if any): {synthesis_context.current_goal or "General understanding."}
- Desired Focus Keywords (if any): {', '.join(synthesis_context.desired_focus_keywords) or "None."}

Memories to Synthesize (presented in rough order of importance):
--- MEMORY LOG START ---
{templated_narrative}
--- MEMORY LOG END ---

Synthesize these memories into a flowing narrative. Highlight connections, temporal order if clear, and overall themes if they emerge.
The output should be a concise narrative text, NOT a list of memories.
Focus on clarity and relevance to ORA's current context.
Keep the narrative under approximately {MAX_SYNTHESIZED_NARRATIVE_LENGTH_TOKENS // 5} words.
Do NOT invent new information not present in the memories.
If there are conflicting memories, you can acknowledge the different perspectives or note the discrepancy if significant.
Start the narrative directly.
"""
        try:
            # Ensure OPENROUTER_API_KEY is in env for LiteLLM
            if not os.getenv("OPENROUTER_API_KEY"):
                logger.error("Memory Synthesizer: OPENROUTER_API_KEY not set. LLM refinement will fail.")
                raise EnvironmentError("OPENROUTER_API_KEY not set for LiteLLM.")

            messages = [{"role": "user", "content": prompt}]
            response = await litellm.acompletion(  # Use acompletion for async
                model=SYNTHESIZER_LLM_MODEL,
                messages=messages,
                max_tokens=MAX_SYNTHESIZED_NARRATIVE_LENGTH_TOKENS,  # Ensure this is a reasonable value
                temperature=0.3  # Lower temperature for more factual synthesis
            )
            refined_narrative_text = response.choices[0].message.content.strip()
            if refined_narrative_text:
                logger.info(
                    f"Memory Synthesizer: LLM refinement successful. Output length: {len(refined_narrative_text)} chars.")
                synthesis_method = "llm_refined"
                return SynthesizedNarrative(refined_narrative_text, contributing_ids, synthesis_method)
            else:
                logger.warning(
                    "Memory Synthesizer: LLM refinement returned empty content. Falling back to templated narrative.")
        except Exception as e:
            logger.error(f"Memory Synthesizer: LLM refinement failed: {e}. Falling back to templated narrative.",
                         exc_info=True)
            # Fallback to templated if LLM fails
            return SynthesizedNarrative(templated_narrative, contributing_ids,
                                        f"template_fallback_due_to_llm_error_{type(e).__name__}")

    return SynthesizedNarrative(templated_narrative, contributing_ids, synthesis_method)
