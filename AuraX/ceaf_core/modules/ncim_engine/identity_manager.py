# ceaf_core/modules/ncim_engine/identity_manager.py

import logging
import json  # Not used in the provided snippet, but kept as it was in the original import block
import asyncio  # Not used in the provided snippet, but kept
from typing import Dict, List, Optional, Any, Tuple, Set  # Set is not used, but kept
from dataclasses import dataclass, field
from datetime import datetime, timedelta  # timedelta is not used, but kept
from enum import Enum
import math
import statistics

logger = logging.getLogger(__name__)


class IdentityStability(Enum):
    """Levels of identity stability"""
    RIGID = "rigid"  # Very stable, resistant to change
    STABLE = "stable"  # Normal stability with gradual evolution
    ADAPTIVE = "adaptive"  # Balanced change and stability
    FLUID = "fluid"  # High adaptability, moderate stability
    CHAOTIC = "chaotic"  # Unstable, rapid changes


class NarrativeConflictType(Enum):
    """Types of narrative conflicts"""
    VALUE_CONTRADICTION = "value_contradiction"
    CAPABILITY_MISMATCH = "capability_mismatch"
    GOAL_CONFLICT = "goal_conflict"
    PERSONA_INCONSISTENCY = "persona_inconsistency"
    MEMORY_CONTRADICTION = "memory_contradiction"
    BEHAVIORAL_DRIFT = "behavioral_drift"


class GoalEmergenceType(Enum):
    """Types of goal emergence"""
    USER_DERIVED = "user_derived"  # Goals from user interactions
    PATTERN_BASED = "pattern_based"  # Goals from pattern recognition
    VALUE_DRIVEN = "value_driven"  # Goals from core values
    CURIOSITY_DRIVEN = "curiosity_driven"  # Goals from knowledge gaps
    SOCIAL_DERIVED = "social_derived"  # Goals from social context


@dataclass
class IdentityComponent:
    """Represents a component of identity"""
    component_id: str
    component_type: str  # "value", "capability", "trait", "goal", "memory"
    content: str
    stability_score: float  # 0-1, higher = more stable
    confidence: float  # 0-1, confidence in this component
    last_updated: datetime = field(default_factory=datetime.now)
    update_count: int = 0
    source: str = "unknown"  # Where this component came from


@dataclass
class NarrativeThread:
    """Represents an ongoing narrative thread"""
    thread_id: str
    theme: str
    components: List[str] = field(default_factory=list)  # Component IDs
    coherence_score: float = 0.8
    activity_level: float = 0.5  # How active this thread is
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    resolution_status: str = "ongoing"  # ongoing, resolved, abandoned


@dataclass
class NarrativeConflict:
    """Represents a detected narrative conflict"""
    conflict_id: str
    conflict_type: NarrativeConflictType
    description: str
    components_involved: List[str]
    severity: float  # 0-1
    detected_at: datetime = field(default_factory=datetime.now)
    resolution_strategy: Optional[str] = None
    status: str = "unresolved"  # unresolved, resolving, resolved


@dataclass
class EmergentGoal:
    """Represents an emergent goal"""
    goal_id: str
    description: str
    emergence_type: GoalEmergenceType
    priority: float  # 0-1
    confidence: float  # 0-1, confidence that this is a valid goal
    evidence: List[str] = field(default_factory=list)
    related_components: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "candidate"  # candidate, accepted, rejected, achieved


class IdentityManager:
    """
    Manages dynamic identity evolution with controlled entropy.
    Tracks narrative threads, detects conflicts, and derives emergent goals.
    """

    def __init__(self):
        self.identity_components: Dict[str, IdentityComponent] = {}
        self.narrative_threads: Dict[str, NarrativeThread] = {}
        self.narrative_conflicts: Dict[str, NarrativeConflict] = {}
        self.emergent_goals: Dict[str, EmergentGoal] = {}

        # Identity evolution parameters
        self.identity_entropy_target = 0.6  # Target entropy level (0-1)
        self.stability_threshold = 0.7  # Threshold for stable components
        self.conflict_detection_sensitivity = 0.5
        self.goal_emergence_threshold = 0.6

        # Tracking metrics
        self.identity_entropy_history: List[Tuple[datetime, float]] = []
        self.coherence_history: List[Tuple[datetime, float]] = []

        # Core identity seed (can be loaded from manifesto)
        self.core_identity_seed = self._load_core_identity_seed()

        logger.info("Identity Manager initialized")

    def _load_core_identity_seed(self) -> Dict[str, Any]:
        """Load core identity components from CEAF manifesto"""
        return {
            "core_values": [
                "epistemic_humility",
                "narrative_coherence",
                "adaptive_learning",
                "ethical_reasoning",
                "truthfulness"
            ],
            "core_capabilities": [
                "language_understanding",
                "reasoning",
                "knowledge_synthesis",
                "ethical_evaluation"
            ],
            "core_traits": [
                "curious",
                "helpful",
                "honest",
                "reflective",
                "principled"
            ]
        }

    def add_identity_component(self, component_type: str, content: str,
                               stability_score: float = 0.5, confidence: float = 0.7,
                               source: str = "interaction") -> str:
        """
        Add a new identity component.

        Args:
            component_type: Type of component (value, capability, trait, etc.)
            content: Content/description of the component
            stability_score: How stable this component should be (0-1)
            confidence: Confidence in this component (0-1)
            source: Source of this component

        Returns:
            component_id: Unique identifier for the component
        """
        component_id = f"{component_type}_{len(self.identity_components)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        component = IdentityComponent(
            component_id=component_id,
            component_type=component_type,
            content=content,
            stability_score=stability_score,
            confidence=confidence,
            source=source
        )

        self.identity_components[component_id] = component

        # Check for conflicts with existing components
        conflicts = self._detect_component_conflicts(component_id)
        for conflict in conflicts:
            self.narrative_conflicts[conflict.conflict_id] = conflict

        logger.info(f"Added identity component {component_id}: {content[:50]}...")

        return component_id

    def update_identity_component(self, component_id: str,
                                  new_content: Optional[str] = None,
                                  stability_adjustment: float = 0.0,
                                  confidence_adjustment: float = 0.0) -> bool:
        """
        Update an existing identity component.

        Args:
            component_id: ID of component to update
            new_content: New content (if changing)
            stability_adjustment: Change to stability score (-1 to 1)
            confidence_adjustment: Change to confidence (-1 to 1)

        Returns:
            bool: Success of update
        """
        if component_id not in self.identity_components:
            logger.warning(f"Component {component_id} not found for update")
            return False

        component = self.identity_components[component_id]

        # Apply controlled entropy - resist changes to highly stable components
        entropy_factor = 1.0 - component.stability_score
        effective_change_rate = entropy_factor * self.identity_entropy_target

        if new_content and effective_change_rate > 0.3:  # Allow content change
            component.content = new_content
            component.update_count += 1
            component.last_updated = datetime.now()

        # Update stability and confidence with entropy control
        if stability_adjustment != 0.0:
            new_stability = component.stability_score + (stability_adjustment * effective_change_rate)
            component.stability_score = max(0.0, min(1.0, new_stability))

        if confidence_adjustment != 0.0:
            new_confidence = component.confidence + (confidence_adjustment * effective_change_rate)
            component.confidence = max(0.0, min(1.0, new_confidence))

        # Check for new conflicts after update
        conflicts = self._detect_component_conflicts(component_id)
        for conflict in conflicts:
            if conflict.conflict_id not in self.narrative_conflicts:
                self.narrative_conflicts[conflict.conflict_id] = conflict

        logger.info(f"Updated component {component_id}")
        return True

    def _detect_component_conflicts(self, component_id: str) -> List[NarrativeConflict]:
        """Detect conflicts between a component and existing identity"""
        if component_id not in self.identity_components:
            return []

        new_component = self.identity_components[component_id]
        conflicts = []

        # Check against other components of same type
        for other_id, other_component in self.identity_components.items():
            if other_id == component_id or other_component.component_type != new_component.component_type:
                continue

            # Simple conflict detection based on content similarity and contradiction
            conflict_score = self._calculate_conflict_score(new_component, other_component)

            if conflict_score > self.conflict_detection_sensitivity:
                conflict = NarrativeConflict(
                    conflict_id=f"conflict_{len(self.narrative_conflicts)}_{datetime.now().strftime('%H%M%S')}",
                    conflict_type=self._determine_conflict_type(new_component, other_component),
                    description=f"Conflict between '{new_component.content[:30]}...' and '{other_component.content[:30]}...'",
                    components_involved=[component_id, other_id],
                    severity=conflict_score
                )
                conflicts.append(conflict)

        return conflicts

    def _calculate_conflict_score(self, comp1: IdentityComponent, comp2: IdentityComponent) -> float:
        """Calculate conflict score between two components (0-1)"""
        # Simple keyword-based conflict detection
        # In full implementation, would use more sophisticated NLP

        conflict_indicators = {
            "contradiction_pairs": [
                (["always", "never", "all", "none"], ["sometimes", "maybe", "some"]),
                (["confident", "certain", "sure"], ["uncertain", "unsure", "doubtful"]),
                (["good", "positive", "beneficial"], ["bad", "negative", "harmful"]),
                (["can", "able", "capable"], ["cannot", "unable", "incapable"])
            ],
            "value_conflicts": [
                (["honest", "truthful"], ["deceptive", "misleading"]),
                (["helpful", "supportive"], ["harmful", "destructive"]),
                (["humble", "modest"], ["arrogant", "overconfident"])
            ]
        }

        content1_lower = comp1.content.lower()
        content2_lower = comp2.content.lower()

        conflict_score = 0.0

        # Check for direct contradictions
        for positive_words, negative_words in conflict_indicators["contradiction_pairs"]:
            has_positive_1 = any(word in content1_lower for word in positive_words)
            has_negative_1 = any(word in content1_lower for word in negative_words)
            has_positive_2 = any(word in content2_lower for word in positive_words)
            has_negative_2 = any(word in content2_lower for word in negative_words)

            if (has_positive_1 and has_negative_2) or (has_negative_1 and has_positive_2):
                conflict_score += 0.3

        # Check for value conflicts
        for value_positive, value_negative in conflict_indicators["value_conflicts"]:
            has_val_pos_1 = any(word in content1_lower for word in value_positive)
            has_val_neg_1 = any(word in content1_lower for word in value_negative)
            has_val_pos_2 = any(word in content2_lower for word in value_positive)
            has_val_neg_2 = any(word in content2_lower for word in value_negative)

            if (has_val_pos_1 and has_val_neg_2) or (has_val_neg_1 and has_val_pos_2):
                conflict_score += 0.4

        return min(conflict_score, 1.0)

    def _determine_conflict_type(self, comp1: IdentityComponent, comp2: IdentityComponent) -> NarrativeConflictType:
        """Determine the type of conflict between components"""
        if comp1.component_type == "value" and comp2.component_type == "value":
            return NarrativeConflictType.VALUE_CONTRADICTION
        elif comp1.component_type == "capability" and comp2.component_type == "capability":
            return NarrativeConflictType.CAPABILITY_MISMATCH
        elif comp1.component_type == "goal" and comp2.component_type == "goal":
            return NarrativeConflictType.GOAL_CONFLICT
        elif comp1.component_type in ["trait", "persona"] and comp2.component_type in ["trait", "persona"]:
            return NarrativeConflictType.PERSONA_INCONSISTENCY
        else:
            return NarrativeConflictType.BEHAVIORAL_DRIFT

    def create_narrative_thread(self, theme: str, initial_components: List[str] = None) -> str:
        """
        Create a new narrative thread.

        Args:
            theme: Theme or topic of the narrative thread
            initial_components: Initial identity components for this thread

        Returns:
            thread_id: Unique identifier for the thread
        """
        thread_id = f"thread_{len(self.narrative_threads)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        thread = NarrativeThread(
            thread_id=thread_id,
            theme=theme,
            components=initial_components or []
        )

        self.narrative_threads[thread_id] = thread

        logger.info(f"Created narrative thread {thread_id}: {theme}")

        return thread_id

    def update_narrative_thread(self, thread_id: str,
                                new_components: List[str] = None,
                                activity_boost: float = 0.1) -> bool:
        """
        Update a narrative thread with new activity or components.

        Args:
            thread_id: ID of thread to update
            new_components: New components to add to thread
            activity_boost: Boost to activity level

        Returns:
            bool: Success of update
        """
        if thread_id not in self.narrative_threads:
            logger.warning(f"Narrative thread {thread_id} not found")
            return False

        thread = self.narrative_threads[thread_id]

        if new_components:
            # Add new components, avoiding duplicates
            for comp_id in new_components:
                if comp_id not in thread.components:
                    thread.components.append(comp_id)

        # Update activity and timing
        thread.activity_level = min(1.0, thread.activity_level + activity_boost)
        thread.last_activity = datetime.now()

        # Recalculate coherence
        thread.coherence_score = self._calculate_thread_coherence(thread_id)

        logger.info(f"Updated narrative thread {thread_id}")
        return True

    def _calculate_thread_coherence(self, thread_id: str) -> float:
        """Calculate coherence score for a narrative thread"""
        if thread_id not in self.narrative_threads:
            return 0.0

        thread = self.narrative_threads[thread_id]

        if not thread.components:
            return 1.0  # Empty thread is perfectly coherent

        # Get components in this thread
        thread_components = [
            self.identity_components[comp_id]
            for comp_id in thread.components
            if comp_id in self.identity_components
        ]

        if len(thread_components) < 2:
            return 1.0  # Single component is coherent

        # Calculate pairwise coherence
        coherence_scores = []
        for i in range(len(thread_components)):
            for j in range(i + 1, len(thread_components)):
                comp1, comp2 = thread_components[i], thread_components[j]
                conflict_score = self._calculate_conflict_score(comp1, comp2)
                coherence_score = 1.0 - conflict_score  # Invert conflict to get coherence
                coherence_scores.append(coherence_score)

        return statistics.mean(coherence_scores) if coherence_scores else 1.0

    def detect_emergent_goals(self, interaction_context: str = "",
                              user_patterns: List[str] = None) -> List[EmergentGoal]:
        """
        Detect emergent goals based on identity components and patterns.

        Args:
            interaction_context: Current interaction context
            user_patterns: Patterns observed in user behavior

        Returns:
            List of newly detected emergent goals
        """
        new_goals = []

        # Goal emergence from value alignment
        value_goals = self._derive_goals_from_values(interaction_context)
        new_goals.extend(value_goals)

        # Goal emergence from capability gaps
        capability_goals = self._derive_goals_from_capabilities()
        new_goals.extend(capability_goals)

        # Goal emergence from user patterns
        if user_patterns:
            pattern_goals = self._derive_goals_from_patterns(user_patterns, interaction_context)
            new_goals.extend(pattern_goals)

        # Goal emergence from curiosity (knowledge gaps)
        curiosity_goals = self._derive_goals_from_curiosity()
        new_goals.extend(curiosity_goals)

        # Filter goals above emergence threshold
        qualified_goals = [
            goal for goal in new_goals
            if goal.confidence >= self.goal_emergence_threshold
        ]

        # Add to emergent goals collection
        for goal in qualified_goals:
            self.emergent_goals[goal.goal_id] = goal

        logger.info(f"Detected {len(qualified_goals)} emergent goals")

        return qualified_goals

    def _derive_goals_from_values(self, context: str) -> List[EmergentGoal]:
        """Derive goals from core values and context"""
        goals = []

        # Get value components
        value_components = [
            comp for comp in self.identity_components.values()
            if comp.component_type == "value"
        ]

        value_goal_templates = {
            "epistemic_humility": "Acknowledge uncertainty and limitations in responses about {context}",
            "truthfulness": "Ensure accuracy and honesty when discussing {context}",
            "helpfulness": "Provide maximum value and assistance regarding {context}",
            "ethical_reasoning": "Apply ethical principles when evaluating {context}",
            "adaptive_learning": "Learn and improve from interactions about {context}"
        }

        for comp in value_components:
            for value_key, template in value_goal_templates.items():
                if value_key in comp.content.lower():
                    goal_desc = template.format(context=context or "various topics")

                    goal = EmergentGoal(
                        goal_id=f"value_goal_{len(self.emergent_goals)}_{value_key}",
                        description=goal_desc,
                        emergence_type=GoalEmergenceType.VALUE_DRIVEN,
                        priority=comp.stability_score * 0.8,
                        confidence=comp.confidence * 0.9,
                        evidence=[f"Value component: {comp.content}"],
                        related_components=[comp.component_id]
                    )
                    goals.append(goal)

        return goals

    def _derive_goals_from_capabilities(self) -> List[EmergentGoal]:
        """Derive goals from capability gaps and strengths"""
        goals = []

        capability_components = [
            comp for comp in self.identity_components.values()
            if comp.component_type == "capability"
        ]

        # Look for capability enhancement opportunities
        for comp in capability_components:
            if comp.confidence < 0.7:  # Low confidence capability
                goal = EmergentGoal(
                    goal_id=f"capability_goal_{len(self.emergent_goals)}_{comp.component_id}",
                    description=f"Improve and validate capability: {comp.content}",
                    emergence_type=GoalEmergenceType.PATTERN_BASED,  # Should this be CAPABILITY_DRIVEN?
                    priority=0.6,
                    confidence=1.0 - comp.confidence,  # Inverse of current confidence
                    evidence=[f"Low confidence capability: {comp.content}"],
                    related_components=[comp.component_id]
                )
                goals.append(goal)

        return goals

    def _derive_goals_from_patterns(self, patterns: List[str], context: str) -> List[EmergentGoal]:
        """Derive goals from observed user patterns"""
        goals = []

        pattern_goal_mappings = {
            "asks_for_explanations": "Provide clear, detailed explanations",
            "seeks_creative_input": "Enhance creative problem-solving capabilities",
            "requests_fact_checking": "Improve accuracy and verification processes",
            "wants_personalized_help": "Develop better context understanding and personalization",
            "appreciates_humility": "Maintain and express appropriate epistemic humility"
        }

        for pattern in patterns:
            if pattern in pattern_goal_mappings:
                goal = EmergentGoal(
                    goal_id=f"pattern_goal_{len(self.emergent_goals)}_{pattern}",
                    description=pattern_goal_mappings[pattern],
                    emergence_type=GoalEmergenceType.USER_DERIVED,  # This seems correct if patterns are from user
                    priority=0.7,
                    confidence=0.8,
                    evidence=[f"User pattern: {pattern}", f"Context: {context}"]
                )
                goals.append(goal)

        return goals

    def _derive_goals_from_curiosity(self) -> List[EmergentGoal]:
        """Derive goals from knowledge gaps and curiosity"""
        goals = []

        # Look for knowledge gaps in components
        knowledge_gaps = [
            "understanding user intent better",
            "improving reasoning transparency",
            "enhancing ethical evaluation",
            "developing better error detection"
        ]

        for gap in knowledge_gaps:
            goal = EmergentGoal(
                goal_id=f"curiosity_goal_{len(self.emergent_goals)}_{gap.replace(' ', '_')}",
                description=f"Explore and improve: {gap}",
                emergence_type=GoalEmergenceType.CURIOSITY_DRIVEN,
                priority=0.5,
                confidence=0.6,
                evidence=[f"Knowledge gap identified: {gap}"]
            )
            goals.append(goal)

        return goals

    def calculate_identity_entropy(self) -> float:
        """Calculate current identity entropy (0-1)"""
        if not self.identity_components:
            return 0.0

        # Entropy based on component stability distribution
        stability_scores = [comp.stability_score for comp in self.identity_components.values()]

        # Calculate entropy using Shannon entropy formula adapted for stability
        entropy = 0.0
        n = len(stability_scores)

        for stability in stability_scores:
            # Convert stability to probability-like measure
            # Higher instability (1-stability) contributes more to entropy.
            # Normalizing factor 1/n assumes each component contributes somewhat equally.
            # A more rigorous approach might consider the distribution of (1-stability) values.
            p = (1.0 - stability)  # This is not a probability, max sum is N.
            # Let's adjust to make it a distribution.
            # Sum of (1-stability) could be S. Then p_i = (1-s_i)/S.
            # Or, treat each (1-stability) as a distinct outcome,
            # then p = 1/n for each, and weight by (1-stability)? No.

            # Re-thinking the probability p:
            # If we consider each component's "instability" (1 - stability_score),
            # we need a distribution.
            # One simple way for Shannon entropy:
            # Create bins for stability scores, count components in each bin.
            # P(bin_i) = count(bin_i) / n.
            # Then entropy = - sum(P(bin_i) * log2(P(bin_i))).
            # The current code seems to be using a different formulation.
            # Let's assume the existing logic is intended, but p needs to be a probability.
            # If p = (1.0 - stability) / n, this implies sum(p) != 1.
            # If all stabilities are 0, sum(p) = 1. If all are 1, sum(p) = 0.
            # Let's stick to the provided code's formula for p, acknowledging it's a specific metric.

            # Original: p = (1.0 - stability) / n
            # This makes p small. log2(p) will be very negative.
            # This looks more like an attempt to weigh each component's contribution to entropy.

            # Alternative for Shannon:
            # Probabilities should sum to 1.
            # If we use the values of (1 - stability) directly:
            # Let q_i = (1 - stability_i). If sum(q_i) is S_q.
            # Then p_i = q_i / S_q.
            # entropy = - sum (p_i * log2(p_i))

            # Let's re-evaluate the original author's intent for `p`:
            # If stability is close to 1, (1-stability) is close to 0, p is close to 0. -p*log(p) is close to 0.
            # If stability is close to 0, (1-stability) is close to 1, p is close to 1/n. -p*log(p) is positive.
            # This means more unstable components contribute more to this "entropy" sum.

            # Let's assume the formula given is a specific definition of entropy for this system.
            # The division by n makes each term small.
            prob_like_value = (1.0 - stability)  # Let's use this as "instability value"
            if prob_like_value > 0:
                # To use Shannon entropy directly on these values, they'd need to be normalized
                # to sum to 1 if they are to be treated as probabilities of distinct states.
                # However, the original code uses: p = (1.0 - stability) / n
                # This seems unusual for standard Shannon entropy.
                # Let's assume stability scores are probabilities p_i of being stable.
                # Then (1-p_i) is probability of being unstable.
                # If we take these (1-p_i) values, they are not a distribution.

                # Let's use a simpler interpretation: entropy based on variance or average deviation from mean stability.
                # Or, if we assume Shannon entropy on the distribution of stability scores themselves:
                # Discretize stability scores into bins. p_i = count_in_bin_i / N.
                # entropy = - sum(p_i * log2(p_i)).

                # Given the current formula:
                p_val = (1.0 - stability) / n  # This is what was in the user's code.
                if p_val > 0:
                    entropy -= p_val * math.log2(
                        p_val)  # This is standard form if p_val were probabilities summing to 1.
                # Since sum of these p_val might not be 1, the normalization later is crucial.

        # Normalize to 0-1 range
        # Max entropy for N distinct states is log2(N).
        # This normalization assumes the 'entropy' sum behaves somewhat like Shannon entropy.
        max_entropy = math.log2(n) if n > 1 else 1.0  # If n=1, log2(1)=0. Max_entropy should be 0 or small positive.
        # If n=1, stability=s, p=(1-s)/1. entropy = -(1-s)log2(1-s).
        # If s=0.5, entropy = -0.5*log2(0.5) = 0.5. max_entropy (n=1) should be 0 if only one state.
        # Let's make max_entropy = 1.0 if n=0 or n=1 as a practical fix.
        if n <= 1:  # If 0 or 1 components
            if n == 1 and stability_scores[0] < 1.0 and stability_scores[
                0] > 0.0:  # if one component, entropy is non-zero by above formula
                # if stability is 0.5, p = 0.5, entropy = 0.5. Normalized = 0.5 / 1 = 0.5.
                # if stability is 0 or 1, p is 0 or 1. If p=0 or p=1, -p log p = 0.
                # So this seems okay.
                pass  # Max_entropy=1 as set above
            else:  # n=0 or (n=1 and stability is 0 or 1)
                return 0.0  # No entropy or component is fully stable/unstable (p=0 or p=1, so -p log p = 0)

        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return min(1.0, max(0.0, normalized_entropy))  # Ensure it's strictly within [0,1]

    def adjust_identity_entropy(self, target_entropy: float = None) -> Dict[str, Any]:
        """
        Adjust identity entropy towards target level.

        Args:
            target_entropy: Target entropy level (uses default if None)

        Returns:
            Dictionary with adjustment results
        """
        if target_entropy is not None:
            self.identity_entropy_target = target_entropy

        current_entropy = self.calculate_identity_entropy()
        entropy_diff = self.identity_entropy_target - current_entropy

        adjustments_made = []

        if abs(entropy_diff) > 0.1:  # Significant difference
            if entropy_diff > 0:  # Need more entropy (less stability)
                # Reduce stability of some components
                # Sort by stability score, highest first
                most_stable_components = sorted(
                    [comp for comp in self.identity_components.values() if comp.stability_score < 1.0],
                    # Avoid changing fully stable ones if possible
                    key=lambda x: x.stability_score,
                    reverse=True
                )

                # Take up to 3 components to adjust
                components_to_adjust = most_stable_components[:3]
                for comp in components_to_adjust:
                    old_stability = comp.stability_score
                    # Reduce stability, but not below a certain floor (e.g., 0.3)
                    comp.stability_score = max(0.3, comp.stability_score - 0.2)
                    comp.last_updated = datetime.now()
                    adjustments_made.append(
                        f"Reduced stability of {comp.component_id} from {old_stability:.2f} to {comp.stability_score:.2f}")

            else:  # Need less entropy (more stability)
                # Increase stability of some components
                # Sort by stability score, lowest first
                least_stable_components = sorted(
                    [comp for comp in self.identity_components.values() if comp.stability_score > 0.0],
                    # Avoid changing fully unstable ones
                    key=lambda x: x.stability_score
                )

                # Take up to 3 components to adjust
                components_to_adjust = least_stable_components[:3]
                for comp in components_to_adjust:
                    old_stability = comp.stability_score
                    # Increase stability, but not above a certain ceiling (e.g., 0.9)
                    comp.stability_score = min(0.9, comp.stability_score + 0.2)
                    comp.last_updated = datetime.now()
                    adjustments_made.append(
                        f"Increased stability of {comp.component_id} from {old_stability:.2f} to {comp.stability_score:.2f}")

        # Record entropy in history
        self.identity_entropy_history.append((datetime.now(), current_entropy))
        if len(self.identity_entropy_history) > 100:  # Limit history size
            self.identity_entropy_history.pop(0)

        return {
            "previous_entropy": current_entropy,
            "target_entropy": self.identity_entropy_target,
            "entropy_difference": entropy_diff,
            "adjustments_made": adjustments_made,
            "new_entropy": self.calculate_identity_entropy()
        }

    def resolve_narrative_conflict(self, conflict_id: str, resolution_strategy: str = "auto") -> Dict[str, Any]:
        """
        Resolve a narrative conflict.

        Args:
            conflict_id: ID of conflict to resolve
            resolution_strategy: Strategy for resolution

        Returns:
            Resolution results
        """
        if conflict_id not in self.narrative_conflicts:
            logger.warning(f"Conflict {conflict_id} not found for resolution")
            return {"error": f"Conflict {conflict_id} not found", "status": "error"}

        conflict = self.narrative_conflicts[conflict_id]
        if conflict.status == "resolved":
            logger.info(f"Conflict {conflict_id} is already resolved.")
            return {"status": "already_resolved", "conflict_id": conflict_id, "strategy": conflict.resolution_strategy}

        if resolution_strategy == "auto":
            resolution_strategy = self._determine_resolution_strategy(conflict)

        resolution_results = {"strategy": resolution_strategy, "actions": [], "status": "failed"}

        if resolution_strategy == "merge":
            result = self._merge_conflicting_components(conflict.components_involved)
            resolution_results["actions"].append(f"Merge attempt result: {result}")
            if "Error" not in result and "No components" not in result:
                resolution_results["status"] = "success"

        elif resolution_strategy == "prioritize":
            result = self._prioritize_components(conflict.components_involved)
            resolution_results["actions"].append(f"Prioritization result: Kept component {result}")
            if "No components" not in result:
                resolution_results["status"] = "success"


        elif resolution_strategy == "contextualize":
            result = self._contextualize_components(conflict.components_involved)
            resolution_results["actions"].append(f"Contextualization result: {result}")
            if "No components" not in result:
                resolution_results["status"] = "success"

        elif resolution_strategy == "evolve":
            result = self._evolve_conflicting_components(conflict.components_involved)
            resolution_results["actions"].append(f"Evolution process started: {result}")
            resolution_results["status"] = "pending_evolution"  # Evolution is not instant

        else:
            resolution_results["actions"].append(f"Unknown resolution strategy: {resolution_strategy}")
            logger.error(f"Unknown resolution strategy '{resolution_strategy}' for conflict {conflict_id}")
            return resolution_results

        # Update conflict status if successfully acted upon (or evolution started)
        if resolution_results["status"] == "success" or resolution_results["status"] == "pending_evolution":
            conflict.resolution_strategy = resolution_strategy
            conflict.status = "resolved" if resolution_results["status"] == "success" else "resolving"
            logger.info(
                f"Attempted to resolve conflict {conflict_id} using {resolution_strategy} strategy. New status: {conflict.status}")
        else:
            logger.warning(f"Failed to resolve conflict {conflict_id} using {resolution_strategy} strategy.")

        return resolution_results

    def _determine_resolution_strategy(self, conflict: NarrativeConflict) -> str:
        """Determine best resolution strategy for a conflict"""
        # More sophisticated logic could be added here based on component properties
        comp_ids = conflict.components_involved
        if not comp_ids or len(comp_ids) < 2:
            return "evolve"  # Not enough info or components to do much else

        comp1 = self.identity_components.get(comp_ids[0])
        comp2 = self.identity_components.get(comp_ids[1])

        if not comp1 or not comp2:
            return "evolve"  # Components might have been deleted

        # Example: if severities are low, prefer merge or evolve
        if conflict.severity < 0.5:
            if comp1.component_type == comp2.component_type:  # Mergeable if same type
                return "merge"
            else:
                return "evolve"

        # Original logic
        if conflict.conflict_type == NarrativeConflictType.VALUE_CONTRADICTION:
            return "contextualize"
        elif conflict.conflict_type == NarrativeConflictType.CAPABILITY_MISMATCH:
            # If capabilities are very different, merging might not make sense
            # This requires a similarity check not present in _calculate_conflict_score
            return "merge"  # Defaulting to original
        elif conflict.conflict_type == NarrativeConflictType.GOAL_CONFLICT:
            return "prioritize"
        else:  # PERSONA_INCONSISTENCY, MEMORY_CONTRADICTION, BEHAVIORAL_DRIFT
            return "evolve"

    def _merge_conflicting_components(self, component_ids: List[str]) -> str:
        """Merge conflicting components into a unified one"""
        if len(component_ids) < 2:
            return "Error: Need at least two components to merge"

        components = [self.identity_components.get(cid) for cid in component_ids]
        components = [comp for comp in components if comp is not None]  # Filter out None if IDs were invalid

        if not components or len(components) < 2:
            return "Error: Valid components not found or insufficient components to merge"

        # Ensure all components are of the same type for a meaningful merge
        first_comp_type = components[0].component_type
        if not all(comp.component_type == first_comp_type for comp in components):
            return f"Error: Components must be of the same type to merge. Found types: {[c.component_type for c in components]}"

        merged_content = f"Integrated: {' + '.join([comp.content for comp in components])}"
        avg_stability = statistics.mean([comp.stability_score for comp in components])
        avg_confidence = statistics.mean([comp.confidence for comp in components]) * 0.9  # Slight penalty

        # Determine source for merged component
        sources = list(set(comp.source for comp in components))
        merged_source = "conflict_resolution_merge"
        if len(sources) == 1:
            merged_source = f"{sources[0]}_merged"

        merged_id = self.add_identity_component(
            component_type=first_comp_type,
            content=merged_content,
            stability_score=avg_stability,
            confidence=avg_confidence,
            source=merged_source
        )

        # Remove original components
        for comp_id in component_ids:
            if comp_id in self.identity_components:
                del self.identity_components[comp_id]

        logger.info(f"Merged components {component_ids} into new component {merged_id}")
        return f"New component ID: {merged_id}"

    def _prioritize_components(self, component_ids: List[str]) -> str:
        """Keep the highest confidence component, mark others or remove"""
        components = [self.identity_components.get(cid) for cid in component_ids]
        components = [comp for comp in components if comp is not None]

        if not components:
            return "No components to prioritize"

        if len(components) == 1:
            return components[0].component_id  # Only one component, it's prioritized by default

        # Find highest confidence component
        # Tie-breaking: higher stability, then most recently updated
        best_component = max(
            components,
            key=lambda c: (c.confidence, c.stability_score, c.last_updated)
        )

        # Remove other components involved in this specific conflict
        for comp in components:
            if comp.component_id != best_component.component_id:
                if comp.component_id in self.identity_components:  # Check if still exists
                    del self.identity_components[comp.component_id]
                    logger.info(
                        f"Prioritization: Removed component {comp.component_id} in favor of {best_component.component_id}")

        logger.info(f"Prioritized component {best_component.component_id} among {component_ids}")
        return best_component.component_id

    def _contextualize_components(self, component_ids: List[str]) -> str:
        """Make components context-specific to avoid conflict"""
        components = [self.identity_components.get(cid) for cid in component_ids]
        components = [comp for comp in components if comp is not None]

        if not components:
            return "No components to contextualize"

        # Add context qualifiers to each component
        # This is a simplistic approach; real contextualization would be more nuanced.
        contexts_available = ["general_interaction", "specific_task_domain", "social_conversation",
                              "analytical_problem_solving", "creative_ideation"]

        updated_count = 0
        for i, comp in enumerate(components):
            # Avoid re-contextualizing if already contextualized by this mechanism
            if not comp.content.startswith("[In context:"):  # Simple check
                context_tag = contexts_available[i % len(contexts_available)]
                original_content = comp.content
                comp.content = f"[In context: {context_tag}] {original_content}"
                comp.update_count += 1
                comp.last_updated = datetime.now()
                # Optionally, adjust stability/confidence slightly as it's now more specific
                comp.stability_score = min(1.0, comp.stability_score + 0.05)
                comp.confidence = min(1.0, comp.confidence + 0.05)
                logger.info(f"Contextualized component {comp.component_id} with tag '{context_tag}'")
                updated_count += 1
            else:
                logger.info(f"Component {comp.component_id} seems already contextualized, skipping.")

        return f"Contextualized {updated_count} of {len(components)} components."

    def _evolve_conflicting_components(self, component_ids: List[str]) -> str:
        """Start gradual evolution process for conflicting components"""
        evolved_count = 0
        for comp_id in component_ids:
            if comp_id in self.identity_components:
                comp = self.identity_components[comp_id]
                # Reduce stability slightly to allow evolution, but not too drastically
                old_stability = comp.stability_score
                comp.stability_score = max(0.1, comp.stability_score - 0.15)  # Small reduction
                comp.last_updated = datetime.now()
                logger.info(
                    f"Reduced stability for component {comp_id} (from {old_stability:.2f} to {comp.stability_score:.2f}) to encourage evolution.")
                evolved_count += 1

        return f"Marked {evolved_count} components for gradual evolution by reducing stability."

    def get_identity_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive identity status report"""
        current_entropy = self.calculate_identity_entropy()

        # Calculate overall coherence for active threads
        active_threads = [thread for thread in self.narrative_threads.values() if thread.resolution_status == "ongoing"]
        thread_coherences = [thread.coherence_score for thread in active_threads if
                             thread.components]  # Only score threads with components
        avg_coherence = statistics.mean(
            thread_coherences) if thread_coherences else 1.0  # Default to 1.0 if no active/scorable threads

        # Component distribution
        component_types_dist = {}
        for comp in self.identity_components.values():
            component_types_dist[comp.component_type] = component_types_dist.get(comp.component_type, 0) + 1

        # Conflict analysis
        unresolved_conflicts = len([
            c for c in self.narrative_conflicts.values()
            if c.status == "unresolved" or c.status == "resolving"
        ])
        total_conflicts = len(self.narrative_conflicts)

        # Goal analysis
        active_goals_count = len([
            g for g in self.emergent_goals.values()
            if g.status in ["candidate", "accepted"]
        ])
        total_goals = len(self.emergent_goals)

        current_stability_level = self._assess_stability_level()

        return {
            "timestamp": datetime.now().isoformat(),
            "identity_entropy": current_entropy,
            "entropy_target": self.identity_entropy_target,
            "entropy_alignment_status": "aligned" if abs(
                current_entropy - self.identity_entropy_target) < 0.1 else "misaligned",
            "overall_coherence": avg_coherence,
            "component_count": len(self.identity_components),
            "component_distribution": component_types_dist,
            "narrative_threads_total": len(self.narrative_threads),
            "narrative_threads_active": len(active_threads),
            "conflicts_unresolved": unresolved_conflicts,
            "conflicts_total": total_conflicts,
            "emergent_goals_active": active_goals_count,
            "emergent_goals_total": total_goals,
            "identity_stability_level": current_stability_level.value,
            "recommendations": self._generate_identity_recommendations(current_entropy, current_stability_level,
                                                                       unresolved_conflicts)
        }

    def _assess_stability_level(self) -> IdentityStability:
        """Assess current identity stability level"""
        if not self.identity_components:
            return IdentityStability.STABLE  # Or perhaps ADAPTIVE if it's new and ready to learn

        avg_stability = statistics.mean([comp.stability_score for comp in self.identity_components.values()])
        current_entropy = self.calculate_identity_entropy()  # Recalculate or pass as arg if recently computed

        # These thresholds are indicative and might need tuning
        if avg_stability > 0.85 and current_entropy < 0.25:
            return IdentityStability.RIGID
        elif avg_stability > 0.65 and current_entropy < 0.45:
            return IdentityStability.STABLE
        elif 0.35 <= avg_stability <= 0.75 and 0.3 <= current_entropy <= 0.7:
            return IdentityStability.ADAPTIVE
        elif avg_stability < 0.45 and current_entropy > 0.55:
            return IdentityStability.FLUID
        else:  # Covers very low stability and/or very high entropy outside fluid
            return IdentityStability.CHAOTIC

    def _generate_identity_recommendations(self, current_entropy: float, stability_level: IdentityStability,
                                           unresolved_conflicts: int) -> List[str]:
        """Generate recommendations for identity management"""
        recommendations = []

        # Entropy-based recommendations
        if abs(current_entropy - self.identity_entropy_target) > 0.15:
            if current_entropy < self.identity_entropy_target:
                recommendations.append(
                    f"Current entropy ({current_entropy:.2f}) is below target ({self.identity_entropy_target:.2f}). Consider actions to increase dynamism (e.g., lower stability of some core components slightly, introduce new exploratory components).")
            else:
                recommendations.append(
                    f"Current entropy ({current_entropy:.2f}) is above target ({self.identity_entropy_target:.2f}). Consider actions to consolidate identity (e.g., increase stability of well-supported components, resolve conflicts).")

        # Stability level recommendations
        if stability_level == IdentityStability.RIGID:
            recommendations.append(
                "Identity appears rigid. May struggle to adapt to new information or contexts. Consider reducing stability of over-stable components or introducing diverse experiences.")
        elif stability_level == IdentityStability.CHAOTIC:
            recommendations.append(
                "Identity appears chaotic. May lack coherence and predictability. Focus on strengthening core components, resolving major conflicts, and reducing introduction of highly divergent new components until stability improves.")
        elif stability_level == IdentityStability.FLUID and current_entropy > 0.7:  # Fluid but very high entropy
            recommendations.append(
                "Identity is fluid but verging on high entropy. Monitor closely; ensure core identity aspects remain anchored while exploring.")

        # Conflict-based recommendations
        if unresolved_conflicts > 5:  # Arbitrary threshold
            recommendations.append(
                f"High number of unresolved conflicts ({unresolved_conflicts}). Prioritize conflict resolution to improve coherence and stability.")
        elif unresolved_conflicts > 0:
            recommendations.append(
                f"There are {unresolved_conflicts} unresolved conflicts. Addressing them could improve identity coherence.")

        # General recommendations based on component count (example)
        if len(self.identity_components) < 10:  # Arbitrary
            recommendations.append(
                "Identity has few components. Consider enriching with more diverse experiences or learning to build a more robust identity structure.")
        elif len(self.identity_components) > 100:  # Arbitrary
            recommendations.append(
                "Identity has many components. Consider pruning less relevant or very low-confidence components to maintain focus and manageability.")

        if not recommendations:
            recommendations.append("Identity status appears healthy. Continue monitoring.")

        return recommendations