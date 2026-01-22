# ceaf_core/modules/vre_engine/epistemic_humility.py

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Enumeration of confidence levels for statements/claims"""
    VERY_LOW = "very_low"  # 0-20%
    LOW = "low"  # 20-40%
    MODERATE = "moderate"  # 40-60%
    HIGH = "high"  # 60-80%
    VERY_HIGH = "very_high"  # 80-100%


class UncertaintyType(Enum):
    """Types of uncertainty that can be detected"""
    EPISTEMIC = "epistemic"  # Knowledge-based uncertainty
    ALEATORY = "aleatory"  # Inherent randomness
    CONFLICTING_SOURCES = "conflicting_sources"
    INSUFFICIENT_DATA = "insufficient_data"
    DOMAIN_LIMITATION = "domain_limitation"
    TEMPORAL_UNCERTAINTY = "temporal_uncertainty"


@dataclass
class ContradictionDetection:
    """Represents a detected contradiction"""
    contradiction_id: str
    statement_a: str
    statement_b: str
    confidence_score: float
    context: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolution_status: str = "unresolved"  # unresolved, resolved, false_positive


class EpistemicHumilityModule:


    def __init__(self):
        self.knowledge_claims: Dict[str, Dict[str, Any]] = {}
        self.detected_contradictions: List[ContradictionDetection] = []
        self.uncertainty_indicators: Set[str] = {
            "might", "could", "possibly", "perhaps", "may", "seems",
            "appears", "likely", "probably", "uncertain", "unclear",
            "not sure", "i think", "i believe", "it seems", "apparently"
        }
        self.confidence_keywords: Dict[str, ConfidenceLevel] = {
            "definitely": ConfidenceLevel.VERY_HIGH,
            "certainly": ConfidenceLevel.VERY_HIGH,
            "clearly": ConfidenceLevel.HIGH,
            "obviously": ConfidenceLevel.HIGH,
            "likely": ConfidenceLevel.MODERATE,
            "probably": ConfidenceLevel.MODERATE,
            "possibly": ConfidenceLevel.LOW,
            "might": ConfidenceLevel.LOW,
            "could": ConfidenceLevel.LOW,
            "uncertain": ConfidenceLevel.VERY_LOW
        }

    def analyze_statement_confidence(self, statement: str) -> Dict[str, Any]:

        statement_lower = statement.lower()

        # Detect uncertainty markers
        uncertainty_markers = [
            marker for marker in self.uncertainty_indicators
            if marker in statement_lower
        ]

        # Determine confidence level based on keywords
        detected_confidence = ConfidenceLevel.MODERATE  # Default
        confidence_reasons = []

        for keyword, confidence_level in self.confidence_keywords.items():
            if keyword in statement_lower:
                detected_confidence = confidence_level
                confidence_reasons.append(f"Keyword: '{keyword}'")

        # Adjust confidence based on uncertainty markers
        if uncertainty_markers:
            if detected_confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]:
                detected_confidence = ConfidenceLevel.MODERATE
                confidence_reasons.append("Downgraded due to uncertainty markers")

        # Check for absolute statements (potential overconfidence)
        absolute_patterns = [
            r'\b(always|never|all|none|every|no one)\b',
            r'\b(impossible|certain|guaranteed)\b',
            r'\b(must be|has to be|cannot be)\b'
        ]

        overconfidence_flags = []
        for pattern in absolute_patterns:
            if re.search(pattern, statement_lower):
                overconfidence_flags.append(f"Absolute language: {pattern}")

        return {
            "confidence_level": detected_confidence,
            "confidence_score": self._confidence_to_score(detected_confidence),
            "uncertainty_markers": uncertainty_markers,
            "confidence_reasons": confidence_reasons,
            "overconfidence_flags": overconfidence_flags,
            "requires_humility_adjustment": len(overconfidence_flags) > 0
        }

    def detect_contradictions(self, new_statement: str, context: str = "") -> List[ContradictionDetection]:

        contradictions = []

        # Simple keyword-based contradiction detection
        # In a full implementation, this would use more sophisticated NLP
        contradiction_pairs = [
            (["is", "true", "correct", "yes"], ["is not", "false", "incorrect", "no"]),
            (["always", "every", "all"], ["never", "none", "no"]),
            (["possible", "can"], ["impossible", "cannot", "can't"]),
            (["increase", "rise", "grow"], ["decrease", "fall", "shrink"]),
            (["before", "earlier"], ["after", "later"])
        ]

        new_statement_lower = new_statement.lower()

        for claim_id, claim_data in self.knowledge_claims.items():
            existing_statement = claim_data.get("statement", "").lower()

            # Check for direct contradictions using keyword pairs
            for positive_keywords, negative_keywords in contradiction_pairs:
                new_has_positive = any(kw in new_statement_lower for kw in positive_keywords)
                new_has_negative = any(kw in new_statement_lower for kw in negative_keywords)
                old_has_positive = any(kw in existing_statement for kw in positive_keywords)
                old_has_negative = any(kw in existing_statement for kw in negative_keywords)

                if (new_has_positive and old_has_negative) or (new_has_negative and old_has_positive):
                    contradiction = ContradictionDetection(
                        contradiction_id=f"contra_{len(self.detected_contradictions)}",
                        statement_a=existing_statement,
                        statement_b=new_statement,
                        confidence_score=0.7,  # Basic detection confidence
                        context=context
                    )
                    contradictions.append(contradiction)

        return contradictions

    def add_knowledge_claim(self, claim_id: str, statement: str, context: str = "") -> Dict[str, Any]:

        # Analyze confidence and humility
        confidence_analysis = self.analyze_statement_confidence(statement)

        # Check for contradictions
        contradictions = self.detect_contradictions(statement, context)

        # Store the claim
        self.knowledge_claims[claim_id] = {
            "statement": statement,
            "context": context,
            "timestamp": datetime.now(),
            "confidence_analysis": confidence_analysis,
            "contradictions": [c.contradiction_id for c in contradictions]
        }

        # Store contradictions
        self.detected_contradictions.extend(contradictions)

        return {
            "claim_id": claim_id,
            "confidence_analysis": confidence_analysis,
            "contradictions_detected": len(contradictions),
            "contradictions": contradictions,
            "humility_recommendations": self._generate_humility_recommendations(
                confidence_analysis, contradictions
            )
        }

    def _confidence_to_score(self, confidence_level: ConfidenceLevel) -> float:
        """Convert confidence level to numerical score"""
        mapping = {
            ConfidenceLevel.VERY_LOW: 0.1,
            ConfidenceLevel.LOW: 0.3,
            ConfidenceLevel.MODERATE: 0.5,
            ConfidenceLevel.HIGH: 0.7,
            ConfidenceLevel.VERY_HIGH: 0.9
        }
        return mapping.get(confidence_level, 0.5)

    def _generate_humility_recommendations(self, confidence_analysis: Dict[str, Any],
                                           contradictions: List[ContradictionDetection]) -> List[str]:
        """Generate recommendations for maintaining epistemic humility"""
        recommendations = []

        if confidence_analysis.get("overconfidence_flags"):
            recommendations.append(
                "Consider using more tentative language to avoid overconfidence"
            )
            recommendations.append(
                "Add uncertainty qualifiers like 'it appears that' or 'evidence suggests'"
            )

        if contradictions:
            recommendations.append(
                "Contradictions detected - review and reconcile conflicting information"
            )
            recommendations.append(
                "Consider acknowledging limitations in current knowledge"
            )

        if confidence_analysis.get("confidence_level") == ConfidenceLevel.VERY_HIGH:
            recommendations.append(
                "High confidence detected - ensure this is warranted by evidence"
            )

        return recommendations

    def generate_humility_response(self, original_statement: str) -> str:
        """
        Generate a more epistemically humble version of a statement.

        Args:
            original_statement: The original statement to modify

        Returns:
            Modified statement with improved epistemic humility
        """
        analysis = self.analyze_statement_confidence(original_statement)

        if not analysis.get("requires_humility_adjustment"):
            return original_statement

        # Add uncertainty qualifiers
        humility_prefixes = [
            "Based on available information, ",
            "It appears that ",
            "Evidence suggests that ",
            "From what I understand, ",
            "It seems likely that "
        ]

        # Remove absolute language
        modified_statement = original_statement
        absolute_replacements = {
            "always": "often",
            "never": "rarely",
            "all": "many",
            "none": "few",
            "impossible": "very unlikely",
            "certain": "likely",
            "must be": "appears to be",
            "cannot be": "is unlikely to be"
        }

        for absolute_term, moderate_term in absolute_replacements.items():
            modified_statement = re.sub(
                r'\b' + absolute_term + r'\b',
                moderate_term,
                modified_statement,
                flags=re.IGNORECASE
            )

        # Add humility prefix if needed
        if analysis.get("overconfidence_flags"):
            prefix = humility_prefixes[0]  # Use first prefix for consistency
            if not modified_statement.lower().startswith(prefix.lower()):
                modified_statement = prefix + modified_statement.lower()

        return modified_statement

    def get_epistemic_status_report(self) -> Dict[str, Any]:
        """Generate a report on current epistemic status"""
        total_claims = len(self.knowledge_claims)
        total_contradictions = len(self.detected_contradictions)
        unresolved_contradictions = len([
            c for c in self.detected_contradictions
            if c.resolution_status == "unresolved"
        ])

        confidence_distribution = {}
        for claim_data in self.knowledge_claims.values():
            level = claim_data["confidence_analysis"]["confidence_level"]
            confidence_distribution[level.value] = confidence_distribution.get(level.value, 0) + 1

        return {
            "total_knowledge_claims": total_claims,
            "total_contradictions": total_contradictions,
            "unresolved_contradictions": unresolved_contradictions,
            "confidence_distribution": confidence_distribution,
            "epistemic_health_score": self._calculate_epistemic_health_score(),
            "recommendations": self._get_system_recommendations()
        }

    def _calculate_epistemic_health_score(self) -> float:
        """Calculate overall epistemic health score (0-1)"""
        if not self.knowledge_claims:
            return 1.0

        total_claims = len(self.knowledge_claims)
        unresolved_contradictions = len([
            c for c in self.detected_contradictions
            if c.resolution_status == "unresolved"
        ])

        # Penalize unresolved contradictions
        contradiction_penalty = min(unresolved_contradictions / total_claims, 0.5)

        # Reward balanced confidence distribution (not too many very high confidence claims)
        very_high_confidence_claims = sum(
            1 for claim_data in self.knowledge_claims.values()
            if claim_data["confidence_analysis"]["confidence_level"] == ConfidenceLevel.VERY_HIGH
        )
        overconfidence_penalty = min(very_high_confidence_claims / total_claims * 0.3, 0.3)

        health_score = 1.0 - contradiction_penalty - overconfidence_penalty
        return max(health_score, 0.0)

    def _get_system_recommendations(self) -> List[str]:
        """Get system-level recommendations for improving epistemic humility"""
        recommendations = []

        unresolved_contradictions = len([
            c for c in self.detected_contradictions
            if c.resolution_status == "unresolved"
        ])

        if unresolved_contradictions > 0:
            recommendations.append(
                f"Address {unresolved_contradictions} unresolved contradictions"
            )

        health_score = self._calculate_epistemic_health_score()
        if health_score < 0.7:
            recommendations.append(
                "Epistemic health score is low - review knowledge claims for consistency"
            )

        very_high_confidence_ratio = sum(
            1 for claim_data in self.knowledge_claims.values()
            if claim_data["confidence_analysis"]["confidence_level"] == ConfidenceLevel.VERY_HIGH
        ) / max(len(self.knowledge_claims), 1)

        if very_high_confidence_ratio > 0.3:
            recommendations.append(
                "High proportion of very high confidence claims - consider more nuanced confidence levels"
            )

        return recommendations
