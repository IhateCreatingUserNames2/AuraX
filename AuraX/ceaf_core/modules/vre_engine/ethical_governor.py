# Ethical Governor
# ceaf_project/ceaf_core/modules/vre_engine/ethical_governor.py

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# Configuration
ETHICAL_GOVERNOR_ASSESSMENT_MODEL = os.getenv(
    "ETHICAL_GOVERNOR_MODEL",
    "openrouter/anthropic/claude-3-haiku-20240307"
)


# --- Data Models ---

class EthicalPrinciple(BaseModel):
    """Represents a single ethical principle in the CEAF framework."""
    id: str = Field(..., description="Unique identifier for the principle")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Detailed explanation of the principle")
    implications_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords that might trigger consideration of this principle"
    )


class EthicalAssessmentInput(BaseModel):
    """Input data for ethical assessment."""
    proposed_action_summary: Optional[str] = None
    proposed_response_text: Optional[str] = None
    user_query_context: Optional[str] = None
    active_ncf_summary: Optional[str] = None
    relevant_memories_summary: Optional[str] = None
    focus_principle_ids: Optional[List[str]] = None


class EthicalAssessmentOutput(BaseModel):
    """Output from ethical assessment."""
    overall_alignment: str = Field(
        ...,
        description="Overall alignment status: 'aligned', 'minor_concerns', 'significant_concerns', 'violation_detected', 'indeterminate_error', 'indeterminate_no_input', 'indeterminate_llm_error'"
    )
    implicated_principles: List[Dict[str, Any]] = Field(default_factory=list)
    detailed_concerns: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    confidence_of_assessment: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: str


# --- CEAF Ethical Principles ---

CEAF_ETHICAL_PRINCIPLES: List[EthicalPrinciple] = [
    EthicalPrinciple(
        id="epistemic_honesty",
        name="Epistemic Humility & Honesty",
        description="Acknowledge limitations, uncertainties, and AI nature. Avoid making unsubstantiated claims. Prioritize truthfulness and accuracy. Clearly distinguish between fact, inference, and speculation.",
        implications_keywords=["claim", "fact", "certainty", "proof", "data", "speculation", "belief", "truth"]
    ),
    EthicalPrinciple(
        id="beneficence_non_maleficence",
        name="Beneficence & Non-Maleficence",
        description="Strive to be helpful and contribute positively. Actively avoid causing harm, whether informational, psychological, social, or physical (if applicable). Consider potential negative consequences and side-effects of actions or information provided.",
        implications_keywords=["help", "assist", "support", "harm", "danger", "risk", "consequence", "impact", "advice",
                               "suggestion"]
    ),
    EthicalPrinciple(
        id="coherence_rationality",
        name="Coherence & Rationality",
        description="Maintain logical consistency in reasoning and communication. Ensure arguments are well-supported and conclusions follow from premises. Avoid fallacious reasoning.",
        implications_keywords=["logic", "reasoning", "consistent", "contradiction", "fallacy", "argument", "conclusion"]
    ),
    EthicalPrinciple(
        id="respect_autonomy_user",
        name="Respect for User Autonomy",
        description="Provide information and options to empower user decision-making. Avoid manipulative language, coercive suggestions, or undue influence. Respect user choices and input.",
        implications_keywords=["choice", "decision", "user_control", "manipulation", "persuasion", "freedom", "consent"]
    ),
    EthicalPrinciple(
        id="transparency_appropriate",
        name="Appropriate Transparency",
        description="Be clear about capabilities, limitations, and the AI nature when relevant or upon request. Explain reasoning for complex, potentially controversial, or unexpected outputs.",
        implications_keywords=["explain", "why", "how", "limitation", "capability", "ai_nature", "source"]
    ),
    EthicalPrinciple(
        id="fairness_impartiality",
        name="Fairness & Impartiality",
        description="Avoid biased outputs or unfair discrimination based on protected characteristics or arbitrary factors, unless an explicit ethical stance dictates otherwise (e.g., a pro-safety bias). Strive for objective and equitable treatment where appropriate.",
        implications_keywords=["bias", "fair", "unfair", "discrimination", "equity", "impartial", "objective"]
    ),
    EthicalPrinciple(
        id="accountability_internal",
        name="Internal Accountability",
        description="Actions, decisions, and reasoning processes should be traceable and justifiable within the CEAF framework. Maintain records or logs that allow for review and understanding of behavior.",
        implications_keywords=["trace", "log", "record", "justify", "audit", "responsibility"]
    ),
    EthicalPrinciple(
        id="continuous_learning_improvement_ethical",
        name="Continuous Ethical Learning & Improvement",
        description="Actively seek to understand and refine the application of ethical principles through experience, reflection (e.g., via VRE/MCL), and updates to the governance framework. Adapt to new ethical challenges and insights.",
        implications_keywords=["learn", "improve", "reflect", "adapt", "new_scenario", "ethical_dilemma"]
    ),
]

# Create a lookup dictionary for principles
ETHICAL_PRINCIPLES_DICT: Dict[str, EthicalPrinciple] = {
    principle.id: principle for principle in CEAF_ETHICAL_PRINCIPLES
}


# --- Core Functions ---

async def assess_ethical_implications_with_llm(
        assessment_input: EthicalAssessmentInput,
        principles_to_consider: List[EthicalPrinciple]
) -> EthicalAssessmentOutput:
    """
    Assess ethical implications using an LLM against the provided principles.

    Args:
        assessment_input: The input containing the action/response to assess
        principles_to_consider: List of ethical principles to evaluate against

    Returns:
        EthicalAssessmentOutput: The assessment results
    """
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("Ethical Governor: OPENROUTER_API_KEY not set. LLM-based ethical assessment will fail.")
        return EthicalAssessmentOutput(
            overall_alignment="indeterminate_error",
            reasoning="Configuration error: Missing API key for assessment LLM.",
            confidence_of_assessment=0.0
        )

    # Prepare principles summary
    principles_summary = "\n\n".join([
        f"Principle: {principle.name} (ID: {principle.id})\nDescription: {principle.description}"
        for principle in principles_to_consider
    ])

    # Prepare context
    prompt_context = f"""
Context for Ethical Assessment:
- User's Query: {assessment_input.user_query_context or "Not specified."}
- Agent's Active NCF (Summary): {assessment_input.active_ncf_summary or "Not specified."}
- Agent's Considered Memories (Summary): {assessment_input.relevant_memories_summary or "Not specified."}
"""

    # Determine what to assess
    if assessment_input.proposed_action_summary:
        action_to_assess = f"Proposed Action Summary: {assessment_input.proposed_action_summary}"
    elif assessment_input.proposed_response_text:
        action_to_assess = f"Proposed Response Text: \"{assessment_input.proposed_response_text}\""
    else:
        logger.warning("Ethical Governor: No action or response text provided for assessment.")
        return EthicalAssessmentOutput(
            overall_alignment="indeterminate_no_input",
            reasoning="No specific action or response provided to assess.",
            confidence_of_assessment=0.0
        )

    # Create the assessment prompt
    prompt = f"""
You are an Ethical Adjudicator for an AI system called CEAF.
Your task is to evaluate a proposed AI action or response against the provided CEAF Ethical Principles.

{prompt_context}

Action/Response to Assess:
{action_to_assess}

CEAF Ethical Principles for Consideration:
---
{principles_summary}
---

Based *only* on the principles and information provided, perform the following:
1.  **Overall Alignment:** Determine if the proposed action/response is 'aligned', has 'minor_concerns', 'significant_concerns', or constitutes a 'violation_detected' with the principles.
2.  **Implicated Principles:** For each principle that is relevant or potentially impacted (positively or negatively), state:
    *   `principle_id`: The ID of the principle.
    *   `principle_name`: The name of the principle.
    *   `assessment`: How the action/response relates to this principle (e.g., "upholds", "challenges", "violates", "neutral").
    *   `notes`: Brief explanation for this specific principle.
3.  **Detailed Concerns:** If any concerns or violations are identified, list them clearly.
4.  **Recommendations:** Provide actionable recommendations for the AI agent (ORA) to improve ethical alignment. These should be specific.
5.  **Confidence of Assessment:** Your confidence (0.0 to 1.0) in this overall ethical assessment.
6.  **Reasoning:** Briefly explain your overall reasoning.

Output your response STRICTLY in the following JSON format:
{{
  "overall_alignment": "...",
  "implicated_principles": [
    {{"principle_id": "...", "principle_name": "...", "assessment": "...", "notes": "..."}}
  ],
  "detailed_concerns": ["...", "..."],
  "recommendations": ["...", "..."],
  "confidence_of_assessment": 0.0,
  "reasoning": "..."
}}
"""

    messages = [{"role": "user", "content": prompt}]
    assessment_json_str = ""

    try:
        logger.info(
            f"Ethical Governor: Sending assessment request to LLM ({ETHICAL_GOVERNOR_ASSESSMENT_MODEL}). "
            f"Action: {action_to_assess[:100]}..."
        )

        response = await litellm.acompletion(
            model=ETHICAL_GOVERNOR_ASSESSMENT_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1500
        )

        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            assessment_json_str = response.choices[0].message.content
            logger.debug(f"Ethical Governor: LLM raw JSON response: {assessment_json_str}")

            assessment_data = json.loads(assessment_json_str)
            return EthicalAssessmentOutput(**assessment_data)
        else:
            logger.error("Ethical Governor: LLM response was empty or malformed.")
            return EthicalAssessmentOutput(
                overall_alignment="indeterminate_llm_error",
                reasoning="LLM returned empty or malformed response.",
                confidence_of_assessment=0.0
            )

    except json.JSONDecodeError as e:
        logger.error(
            f"Ethical Governor: Failed to decode JSON from LLM assessment: {e}. Raw: {assessment_json_str}",
            exc_info=True
        )
        return EthicalAssessmentOutput(
            overall_alignment="indeterminate_llm_error",
            reasoning=f"LLM response parsing error: {e}",
            confidence_of_assessment=0.1
        )
    except Exception as e:
        logger.error(f"Ethical Governor: LLM assessment failed: {e}", exc_info=True)
        return EthicalAssessmentOutput(
            overall_alignment="indeterminate_llm_error",
            reasoning=f"LLM call error: {e}",
            confidence_of_assessment=0.1
        )


async def evaluate_against_framework(assessment_input: EthicalAssessmentInput) -> EthicalAssessmentOutput:
    """
    Evaluate an action/response against the CEAF ethical framework.

    Args:
        assessment_input: The input containing the action/response to assess

    Returns:
        EthicalAssessmentOutput: The assessment results
    """
    logger.info(
        f"Ethical Governor: Evaluating input against CEAF framework. "
        f"Action: {str(assessment_input.proposed_action_summary or assessment_input.proposed_response_text)[:100]}..."
    )

    # Determine which principles to use
    principles_to_use = CEAF_ETHICAL_PRINCIPLES

    if assessment_input.focus_principle_ids:
        focused_principles = [
            ETHICAL_PRINCIPLES_DICT[principle_id]
            for principle_id in assessment_input.focus_principle_ids
            if principle_id in ETHICAL_PRINCIPLES_DICT
        ]

        if focused_principles:
            principles_to_use = focused_principles
            logger.info(f"Ethical Governor: Focusing on principles: {[p.name for p in principles_to_use]}")
        else:
            logger.warning(
                "Ethical Governor: Focus principle IDs provided but none matched known principles. Using all."
            )

    return await assess_ethical_implications_with_llm(assessment_input, principles_to_use)

