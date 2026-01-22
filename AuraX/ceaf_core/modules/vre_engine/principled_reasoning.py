# ceaf_core/modules/vre_engine/principled_reasoning.py

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ReasoningStrategy(Enum):
    """Different reasoning strategies available"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    DIALECTICAL = "dialectical"
    SYSTEMS_THINKING = "systems"


class ArgumentType(Enum):
    """Types of arguments in reasoning"""
    SUPPORTING = "supporting"
    OPPOSING = "opposing"
    QUALIFYING = "qualifying"
    ALTERNATIVE = "alternative"


@dataclass
class ReasoningPath:
    """Represents a reasoning path with its evaluation"""
    strategy: ReasoningStrategy
    premises: List[str]
    steps: List[str]
    conclusion: str
    confidence: float
    assumptions: List[str]
    challenges: List[str]
    metadata: Dict[str, Any]


@dataclass
class FallacyDetection:
    """Detected logical fallacy"""
    fallacy_type: str
    description: str
    location: str
    severity: float
    correction_suggestion: str


class PrincipledReasoningPathways:
    """
    Implements multiple reasoning strategies with internal red teaming
    """

    def __init__(self):
        self.strategies = self._initialize_strategies()
        self.fallacy_detectors = self._initialize_fallacy_detectors()
        self.red_team_prompts = self._load_red_team_prompts()

    def _initialize_strategies(self) -> Dict[ReasoningStrategy, callable]:
        """Initialize different reasoning strategies"""
        return {
            ReasoningStrategy.DEDUCTIVE: self._deductive_reasoning,
            ReasoningStrategy.INDUCTIVE: self._inductive_reasoning,
            ReasoningStrategy.ABDUCTIVE: self._abductive_reasoning,
            ReasoningStrategy.ANALOGICAL: self._analogical_reasoning,
            ReasoningStrategy.DIALECTICAL: self._dialectical_reasoning,
            ReasoningStrategy.SYSTEMS_THINKING: self._systems_thinking
        }

    def _initialize_fallacy_detectors(self) -> Dict[str, callable]:
        """Initialize fallacy detection patterns"""
        return {
            "ad_hominem": self._detect_ad_hominem,
            "straw_man": self._detect_straw_man,
            "false_dichotomy": self._detect_false_dichotomy,
            "slippery_slope": self._detect_slippery_slope,
            "circular_reasoning": self._detect_circular_reasoning,
            "hasty_generalization": self._detect_hasty_generalization,
            "appeal_to_authority": self._detect_appeal_to_authority,
            "post_hoc": self._detect_post_hoc,
            "bandwagon": self._detect_bandwagon,
            "false_equivalence": self._detect_false_equivalence
        }

    def _load_red_team_prompts(self) -> List[str]:
        """Load red teaming challenge prompts"""
        return [
            "What assumptions are being made here?",
            "What evidence contradicts this conclusion?",
            "What alternative explanations exist?",
            "What are the weakest points in this argument?",
            "How might this reasoning fail in edge cases?",
            "What biases might be influencing this conclusion?",
            "What context is being ignored?",
            "How would an adversary attack this reasoning?",
            "What are the second-order effects not considered?",
            "Where might this reasoning break down at scale?"
        ]

    def apply_reasoning(self, query: str, context: Dict[str, Any],
                        strategy: Optional[ReasoningStrategy] = None) -> Dict[str, Any]:
        """Apply principled reasoning with internal red teaming"""

        # Select strategy
        if strategy is None:
            strategy = self._select_best_strategy(query, context)

        # Generate initial reasoning path
        reasoning_path = self.strategies[strategy](query, context)

        # Detect fallacies
        fallacies = self._detect_fallacies(reasoning_path)

        # Red team the reasoning
        challenges = self._red_team_reasoning(reasoning_path)

        # Refine based on challenges
        refined_path = self._refine_reasoning(reasoning_path, fallacies, challenges)

        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(refined_path)

        return {
            "primary_path": refined_path,
            "fallacies_detected": fallacies,
            "red_team_challenges": challenges,
            "counterfactuals": counterfactuals,
            "confidence": self._calculate_confidence(refined_path, fallacies, challenges),
            "metadata": {
                "strategy_used": strategy.value,
                "timestamp": datetime.now().isoformat(),
                "iterations": len(challenges)
            }
        }

    def _select_best_strategy(self, query: str, context: Dict[str, Any]) -> ReasoningStrategy:
        """Select the most appropriate reasoning strategy"""
        # Analyze query characteristics
        if "prove" in query.lower() or "demonstrate" in query.lower():
            return ReasoningStrategy.DEDUCTIVE
        elif "pattern" in query.lower() or "trend" in query.lower():
            return ReasoningStrategy.INDUCTIVE
        elif "explain" in query.lower() or "why" in query.lower():
            return ReasoningStrategy.ABDUCTIVE
        elif "similar" in query.lower() or "like" in query.lower():
            return ReasoningStrategy.ANALOGICAL
        elif "debate" in query.lower() or "pros and cons" in query.lower():
            return ReasoningStrategy.DIALECTICAL
        elif "system" in query.lower() or "interconnect" in query.lower():
            return ReasoningStrategy.SYSTEMS_THINKING
        else:
            return ReasoningStrategy.DEDUCTIVE  # Default

    def _deductive_reasoning(self, query: str, context: Dict[str, Any]) -> ReasoningPath:
        """Apply deductive reasoning"""
        premises = self._extract_premises(query, context)
        steps = self._generate_deductive_steps(premises)
        conclusion = self._derive_conclusion(steps)

        return ReasoningPath(
            strategy=ReasoningStrategy.DEDUCTIVE,
            premises=premises,
            steps=steps,
            conclusion=conclusion,
            confidence=0.85,
            assumptions=self._identify_assumptions(premises),
            challenges=[],
            metadata={"method": "modus_ponens"}
        )

    def _inductive_reasoning(self, query: str, context: Dict[str, Any]) -> ReasoningPath:
        """Apply inductive reasoning"""
        observations = self._gather_observations(query, context)
        pattern = self._identify_pattern(observations)
        generalization = self._form_generalization(pattern)

        return ReasoningPath(
            strategy=ReasoningStrategy.INDUCTIVE,
            premises=observations,
            steps=[f"Pattern identified: {pattern}"],
            conclusion=generalization,
            confidence=0.75,
            assumptions=["Sample is representative", "Pattern will continue"],
            challenges=[],
            metadata={"sample_size": len(observations)}
        )

    def _abductive_reasoning(self, query: str, context: Dict[str, Any]) -> ReasoningPath:
        """Apply abductive reasoning"""
        observation = self._extract_observation(query)
        hypotheses = self._generate_hypotheses(observation, context)
        best_explanation = self._select_best_explanation(hypotheses)

        return ReasoningPath(
            strategy=ReasoningStrategy.ABDUCTIVE,
            premises=[observation],
            steps=[f"Hypothesis: {h}" for h in hypotheses],
            conclusion=best_explanation,
            confidence=0.70,
            assumptions=["All relevant hypotheses considered"],
            challenges=[],
            metadata={"hypotheses_count": len(hypotheses)}
        )

    def _analogical_reasoning(self, query: str, context: Dict[str, Any]) -> ReasoningPath:
        """Apply analogical reasoning"""
        source = self._identify_source_domain(query, context)
        target = self._identify_target_domain(query)
        mappings = self._create_mappings(source, target)
        inference = self._transfer_knowledge(mappings)

        return ReasoningPath(
            strategy=ReasoningStrategy.ANALOGICAL,
            premises=[f"Source: {source}", f"Target: {target}"],
            steps=[f"Mapping: {m}" for m in mappings],
            conclusion=inference,
            confidence=0.65,
            assumptions=["Analogy is valid", "Mappings are accurate"],
            challenges=[],
            metadata={"similarity_score": 0.8}
        )

    def _dialectical_reasoning(self, query: str, context: Dict[str, Any]) -> ReasoningPath:
        """Apply dialectical reasoning"""
        thesis = self._extract_thesis(query)
        antithesis = self._generate_antithesis(thesis)
        synthesis = self._create_synthesis(thesis, antithesis)

        return ReasoningPath(
            strategy=ReasoningStrategy.DIALECTICAL,
            premises=[thesis, antithesis],
            steps=["Thesis vs Antithesis", "Resolving contradictions"],
            conclusion=synthesis,
            confidence=0.80,
            assumptions=["Both perspectives have merit"],
            challenges=[],
            metadata={"dialectic_rounds": 1}
        )

    def _systems_thinking(self, query: str, context: Dict[str, Any]) -> ReasoningPath:
        """Apply systems thinking"""
        components = self._identify_system_components(query, context)
        interactions = self._map_interactions(components)
        emergent_properties = self._identify_emergent_properties(interactions)

        return ReasoningPath(
            strategy=ReasoningStrategy.SYSTEMS_THINKING,
            premises=[f"Component: {c}" for c in components],
            steps=[f"Interaction: {i}" for i in interactions],
            conclusion=f"System behavior: {emergent_properties}",
            confidence=0.75,
            assumptions=["System boundaries are correct", "All key interactions identified"],
            challenges=[],
            metadata={"complexity_level": len(interactions)}
        )

    def _detect_fallacies(self, path: ReasoningPath) -> List[FallacyDetection]:
        """Detect logical fallacies in reasoning path"""
        fallacies = []

        # Check each fallacy type
        full_reasoning = " ".join(path.premises + path.steps + [path.conclusion])

        for fallacy_name, detector in self.fallacy_detectors.items():
            detection = detector(full_reasoning, path)
            if detection:
                fallacies.append(detection)

        return fallacies

    def _detect_ad_hominem(self, text: str, path: ReasoningPath) -> Optional[FallacyDetection]:
        """Detect ad hominem attacks"""
        attack_patterns = ["stupid", "idiot", "moron", "ignorant", "biased"]
        for pattern in attack_patterns:
            if pattern in text.lower():
                return FallacyDetection(
                    fallacy_type="ad_hominem",
                    description="Attacking the person rather than the argument",
                    location=text,
                    severity=0.8,
                    correction_suggestion="Focus on the argument's merits, not the person"
                )
        return None

    def _detect_straw_man(self, text: str, path: ReasoningPath) -> Optional[FallacyDetection]:
        """Detect straw man fallacies"""
        indicators = ["you're saying", "so you think", "your position is"]
        misrepresentation_words = ["always", "never", "completely", "totally"]

        for indicator in indicators:
            if indicator in text.lower():
                for word in misrepresentation_words:
                    if word in text.lower():
                        return FallacyDetection(
                            fallacy_type="straw_man",
                            description="Misrepresenting the opposing position",
                            location=text,
                            severity=0.7,
                            correction_suggestion="Accurately represent the opposing view"
                        )
        return None

    def _detect_false_dichotomy(self, text: str, path: ReasoningPath) -> Optional[FallacyDetection]:
        """Detect false dichotomies"""
        patterns = ["either...or", "only two", "must choose between"]
        for pattern in patterns:
            if pattern in text.lower():
                return FallacyDetection(
                    fallacy_type="false_dichotomy",
                    description="Presenting only two options when more exist",
                    location=text,
                    severity=0.6,
                    correction_suggestion="Consider additional alternatives"
                )
        return None

    def _detect_slippery_slope(self, text: str, path: ReasoningPath) -> Optional[FallacyDetection]:
        """Detect slippery slope arguments"""
        if "will lead to" in text.lower() and "eventually" in text.lower():
            return FallacyDetection(
                fallacy_type="slippery_slope",
                description="Assuming one event will lead to extreme consequences",
                location=text,
                severity=0.65,
                correction_suggestion="Establish clear causal connections"
            )
        return None

    def _detect_circular_reasoning(self, text: str, path: ReasoningPath) -> Optional[FallacyDetection]:
        """Detect circular reasoning"""
        # Check if conclusion is too similar to premises
        for premise in path.premises:
            if self._calculate_similarity(premise, path.conclusion) > 0.9:
                return FallacyDetection(
                    fallacy_type="circular_reasoning",
                    description="Conclusion restates the premise",
                    location=path.conclusion,
                    severity=0.9,
                    correction_suggestion="Provide independent support for conclusion"
                )
        return None

    def _detect_hasty_generalization(self, text: str, path: ReasoningPath) -> Optional[FallacyDetection]:
        """Detect hasty generalizations"""
        if path.strategy == ReasoningStrategy.INDUCTIVE:
            sample_size = path.metadata.get("sample_size", float('inf'))
            if sample_size < 5:
                return FallacyDetection(
                    fallacy_type="hasty_generalization",
                    description="Drawing broad conclusions from limited examples",
                    location=path.conclusion,
                    severity=0.7,
                    correction_suggestion="Gather more examples before generalizing"
                )
        return None

    def _detect_appeal_to_authority(self, text: str, path: ReasoningPath) -> Optional[FallacyDetection]:
        """Detect appeals to authority"""
        authority_patterns = ["expert says", "according to", "studies show"]
        for pattern in authority_patterns:
            if pattern in text.lower() and "therefore" in text.lower():
                return FallacyDetection(
                    fallacy_type="appeal_to_authority",
                    description="Relying solely on authority without evidence",
                    location=text,
                    severity=0.5,
                    correction_suggestion="Provide direct evidence beyond authority"
                )
        return None

    def _detect_post_hoc(self, text: str, path: ReasoningPath) -> Optional[FallacyDetection]:
        """Detect post hoc ergo propter hoc"""
        if "after" in text.lower() and "caused" in text.lower():
            return FallacyDetection(
                fallacy_type="post_hoc",
                description="Assuming correlation implies causation",
                location=text,
                severity=0.7,
                correction_suggestion="Establish causal mechanism, not just sequence"
            )
        return None

    def _detect_bandwagon(self, text: str, path: ReasoningPath) -> Optional[FallacyDetection]:
        """Detect bandwagon fallacy"""
        patterns = ["everyone", "most people", "popular opinion"]
        for pattern in patterns:
            if pattern in text.lower():
                return FallacyDetection(
                    fallacy_type="bandwagon",
                    description="Appealing to popularity rather than truth",
                    location=text,
                    severity=0.6,
                    correction_suggestion="Evaluate merit independently of popularity"
                )
        return None

    def _detect_false_equivalence(self, text: str, path: ReasoningPath) -> Optional[FallacyDetection]:
        """Detect false equivalence"""
        if "same as" in text.lower() or "no different than" in text.lower():
            return FallacyDetection(
                fallacy_type="false_equivalence",
                description="Treating different things as equivalent",
                location=text,
                severity=0.65,
                correction_suggestion="Acknowledge relevant differences"
            )
        return None

    def _red_team_reasoning(self, path: ReasoningPath) -> List[Dict[str, Any]]:
        """Apply red team challenges to reasoning"""
        challenges = []

        for prompt in self.red_team_prompts:
            challenge = self._generate_challenge(prompt, path)
            if challenge:
                challenges.append({
                    "prompt": prompt,
                    "challenge": challenge,
                    "severity": self._assess_challenge_severity(challenge, path),
                    "suggested_revision": self._suggest_revision(challenge, path)
                })

        return challenges

    def _generate_challenge(self, prompt: str, path: ReasoningPath) -> str:
        """Generate specific challenge based on prompt"""
        if "assumptions" in prompt:
            return self._challenge_assumptions(path)
        elif "evidence" in prompt:
            return self._challenge_evidence(path)
        elif "alternative" in prompt:
            return self._challenge_alternatives(path)
        elif "weakest" in prompt:
            return self._identify_weakest_point(path)
        elif "edge cases" in prompt:
            return self._identify_edge_cases(path)
        elif "biases" in prompt:
            return self._identify_biases(path)
        elif "context" in prompt:
            return self._identify_missing_context(path)
        elif "adversary" in prompt:
            return self._adversarial_attack(path)
        elif "second-order" in prompt:
            return self._identify_second_order_effects(path)
        elif "scale" in prompt:
            return self._test_scalability(path)
        else:
            return ""

    def _refine_reasoning(self, path: ReasoningPath,
                          fallacies: List[FallacyDetection],
                          challenges: List[Dict[str, Any]]) -> ReasoningPath:
        """Refine reasoning based on detected issues"""
        refined_path = ReasoningPath(
            strategy=path.strategy,
            premises=path.premises.copy(),
            steps=path.steps.copy(),
            conclusion=path.conclusion,
            confidence=path.confidence,
            assumptions=path.assumptions.copy(),
            challenges=[c["challenge"] for c in challenges],
            metadata=path.metadata.copy()
        )

        # Address fallacies
        for fallacy in fallacies:
            if fallacy.severity > 0.7:
                refined_path = self._fix_fallacy(refined_path, fallacy)

        # Address challenges
        for challenge in challenges:
            if challenge["severity"] > 0.6:
                refined_path = self._address_challenge(refined_path, challenge)

        # Recalculate confidence
        refined_path.confidence = self._calculate_confidence(
            refined_path, fallacies, challenges
        )

        return refined_path

    def _generate_counterfactuals(self, path: ReasoningPath) -> List[Dict[str, Any]]:
        """Generate counterfactual scenarios"""
        counterfactuals = []

        # Vary premises
        for i, premise in enumerate(path.premises):
            negated = self._negate_premise(premise)
            alternative_path = self._trace_counterfactual(path, i, negated)
            counterfactuals.append({
                "type": "premise_negation",
                "original": premise,
                "counterfactual": negated,
                "result": alternative_path
            })

        # Vary assumptions
        for assumption in path.assumptions:
            relaxed = self._relax_assumption(assumption)
            impact = self._assess_assumption_impact(path, assumption, relaxed)
            counterfactuals.append({
                "type": "assumption_relaxation",
                "original": assumption,
                "counterfactual": relaxed,
                "impact": impact
            })

        return counterfactuals

    def _calculate_confidence(self, path: ReasoningPath,
                              fallacies: List[FallacyDetection],
                              challenges: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score"""
        base_confidence = path.confidence

        # Reduce for fallacies
        fallacy_penalty = sum(f.severity * 0.1 for f in fallacies)

        # Reduce for unaddressed challenges
        challenge_penalty = sum(c["severity"] * 0.05
                                for c in challenges
                                if c["severity"] > 0.7)

        # Boost for successfully addressed issues
        addressed_bonus = len([c for c in path.challenges]) * 0.02

        final_confidence = base_confidence - fallacy_penalty - challenge_penalty + addressed_bonus

        return max(0.1, min(1.0, final_confidence))

    # Helper methods (implementing stubs for completeness)
    def _extract_premises(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Extract premises from query and context"""
        return [f"Premise from query: {query[:50]}..."]

    def _generate_deductive_steps(self, premises: List[str]) -> List[str]:
        """Generate deductive reasoning steps"""
        return ["If P then Q", "P is true", "Therefore Q"]

    def _derive_conclusion(self, steps: List[str]) -> str:
        """Derive conclusion from reasoning steps"""
        return "Conclusion based on deductive steps"

    def _identify_assumptions(self, premises: List[str]) -> List[str]:
        """Identify underlying assumptions"""
        return ["Assumption 1", "Assumption 2"]

    def _gather_observations(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Gather relevant observations"""
        return ["Observation 1", "Observation 2", "Observation 3"]

    def _identify_pattern(self, observations: List[str]) -> str:
        """Identify pattern in observations"""
        return "Pattern identified in observations"

    def _form_generalization(self, pattern: str) -> str:
        """Form generalization from pattern"""
        return f"General rule based on {pattern}"

    def _extract_observation(self, query: str) -> str:
        """Extract key observation from query"""
        return f"Observation: {query[:50]}..."

    def _generate_hypotheses(self, observation: str, context: Dict[str, Any]) -> List[str]:
        """Generate possible explanations"""
        return ["Hypothesis 1", "Hypothesis 2", "Hypothesis 3"]

    def _select_best_explanation(self, hypotheses: List[str]) -> str:
        """Select most likely explanation"""
        return hypotheses[0] if hypotheses else "No explanation found"

    def _identify_source_domain(self, query: str, context: Dict[str, Any]) -> str:
        """Identify source domain for analogy"""
        return "Source domain"

    def _identify_target_domain(self, query: str) -> str:
        """Identify target domain for analogy"""
        return "Target domain"

    def _create_mappings(self, source: str, target: str) -> List[str]:
        """Create mappings between domains"""
        return [f"{source} maps to {target}"]

    def _transfer_knowledge(self, mappings: List[str]) -> str:
        """Transfer knowledge through analogy"""
        return "Knowledge transferred through analogy"

    def _extract_thesis(self, query: str) -> str:
        """Extract thesis from query"""
        return f"Thesis: {query[:50]}..."

    def _generate_antithesis(self, thesis: str) -> str:
        """Generate opposing view"""
        return f"Antithesis to {thesis}"

    def _create_synthesis(self, thesis: str, antithesis: str) -> str:
        """Create synthesis of opposing views"""
        return f"Synthesis of {thesis} and {antithesis}"

    def _identify_system_components(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Identify system components"""
        return ["Component A", "Component B", "Component C"]

    def _map_interactions(self, components: List[str]) -> List[str]:
        """Map component interactions"""
        return [f"{components[0]} affects {components[1]}"]

    def _identify_emergent_properties(self, interactions: List[str]) -> str:
        """Identify emergent system properties"""
        return "Emergent behavior from interactions"

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        # Simple placeholder - in practice use proper similarity metrics
        return 0.5

    def _challenge_assumptions(self, path: ReasoningPath) -> str:
        """Challenge identified assumptions"""
        if path.assumptions:
            return f"Assumption '{path.assumptions[0]}' may not hold because..."
        return "No explicit assumptions to challenge"

    def _challenge_evidence(self, path: ReasoningPath) -> str:
        """Challenge supporting evidence"""
        return "Evidence is limited to specific contexts and may not generalize"

    def _challenge_alternatives(self, path: ReasoningPath) -> str:
        """Suggest alternative explanations"""
        return f"Alternative to '{path.conclusion}': Consider..."

    def _identify_weakest_point(self, path: ReasoningPath) -> str:
        """Identify weakest point in reasoning"""
        return f"Weakest link: {path.steps[0] if path.steps else 'No clear steps'}"

    def _identify_edge_cases(self, path: ReasoningPath) -> str:
        """Identify potential edge cases"""
        return "Edge case: What if the initial conditions are reversed?"

    def _identify_biases(self, path: ReasoningPath) -> str:
        """Identify potential biases"""
        return "Potential confirmation bias in selecting supporting evidence"

    def _identify_missing_context(self, path: ReasoningPath) -> str:
        """Identify missing contextual information"""
        return "Missing context: Historical precedents and cultural factors"

    def _adversarial_attack(self, path: ReasoningPath) -> str:
        """Generate adversarial attack on reasoning"""
        return "An adversary could exploit the assumption that..."

    def _identify_second_order_effects(self, path: ReasoningPath) -> str:
        """Identify second-order effects"""
        return "Second-order effect: This conclusion might lead to..."

    def _test_scalability(self, path: ReasoningPath) -> str:
        """Test if reasoning scales"""
        return "At scale, this reasoning might break due to..."

    def _assess_challenge_severity(self, challenge: str, path: ReasoningPath) -> float:
        """Assess how severe a challenge is"""
        # Placeholder - would implement proper severity assessment
        return 0.7

    def _suggest_revision(self, challenge: str, path: ReasoningPath) -> str:
        """Suggest revision based on challenge"""
        return f"To address '{challenge}', consider..."

    def _fix_fallacy(self, path: ReasoningPath, fallacy: FallacyDetection) -> ReasoningPath:
        """Fix detected fallacy in reasoning path"""
        # Create a copy and fix the specific fallacy
        fixed_path = ReasoningPath(
            strategy=path.strategy,
            premises=path.premises.copy(),
            steps=path.steps.copy(),
            conclusion=path.conclusion,
            confidence=path.confidence * 0.9,  # Reduce confidence
            assumptions=path.assumptions.copy(),
            challenges=path.challenges.copy(),
            metadata=path.metadata.copy()
        )

        # Add note about fixed fallacy
        fixed_path.metadata[f"fixed_{fallacy.fallacy_type}"] = fallacy.correction_suggestion

        return fixed_path

    def _address_challenge(self, path: ReasoningPath, challenge: Dict[str, Any]) -> ReasoningPath:
        """Address a specific challenge"""
        # Create a copy and address the challenge
        addressed_path = ReasoningPath(
            strategy=path.strategy,
            premises=path.premises.copy(),
            steps=path.steps + [f"Addressing: {challenge['challenge']}"],
            conclusion=path.conclusion,
            confidence=path.confidence,
            assumptions=path.assumptions.copy(),
            challenges=path.challenges + [challenge['challenge']],
            metadata=path.metadata.copy()
        )

        addressed_path.metadata["addressed_challenges"] = \
            addressed_path.metadata.get("addressed_challenges", 0) + 1

        return addressed_path

    def _negate_premise(self, premise: str) -> str:
        """Negate a premise for counterfactual"""
        return f"NOT({premise})"

    def _trace_counterfactual(self, path: ReasoningPath, premise_index: int,
                              new_premise: str) -> str:
        """Trace reasoning with counterfactual premise"""
        return f"With '{new_premise}', conclusion would be different"

    def _relax_assumption(self, assumption: str) -> str:
        """Relax an assumption"""
        return f"Relaxed: {assumption} (may not always hold)"

    def _assess_assumption_impact(self, path: ReasoningPath,
                                  original: str, relaxed: str) -> str:
        """Assess impact of relaxing assumption"""
        return f"Impact: Conclusion reliability reduced by 20%"