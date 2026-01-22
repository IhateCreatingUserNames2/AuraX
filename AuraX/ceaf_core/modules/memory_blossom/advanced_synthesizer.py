# ceaf_core/modules/memory_blossom/advanced_synthesizer.py

import logging
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import math
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from collections import defaultdict, Counter

from ceaf_core.services.llm_service import LLMService
logger = logging.getLogger(__name__)


class CoherenceIssueType(Enum):
    """Types of narrative coherence issues"""
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    THEMATIC_DRIFT = "thematic_drift"
    EMOTIONAL_DISCONTINUITY = "emotional_discontinuity"
    FACTUAL_CONTRADICTION = "factual_contradiction"
    PERSPECTIVE_CONFLICT = "perspective_conflict"
    CAUSAL_BREAKDOWN = "causal_breakdown"


class StoryArcType(Enum):
    """Types of story arcs for weaving"""
    CHRONOLOGICAL = "chronological"
    THEMATIC = "thematic"
    CAUSAL = "causal"
    EMOTIONAL = "emotional"
    IMPORTANCE = "importance"
    ASSOCIATIVE = "associative"


@dataclass
class MemoryCluster:
    """Represents a cluster of related memories"""
    cluster_id: str
    memories: List[Any]  # Memory objects
    centroid_keywords: List[str]
    coherence_score: float
    temporal_span: Tuple[datetime, datetime]
    dominant_theme: str
    emotional_tone: str
    importance_weight: float = 1.0


@dataclass
class CoherenceIssue:
    """Represents a detected coherence issue"""
    issue_id: str
    issue_type: CoherenceIssueType
    description: str
    affected_memories: List[str]  # Memory IDs
    severity: float  # 0-1
    suggested_repair: str
    confidence: float = 0.8


@dataclass
class StoryWeavingResult:
    """Result of story weaving process"""
    narrative_text: str
    story_arc_type: StoryArcType
    coherence_score: float
    memory_clusters_used: List[str]
    weaving_strategy: str
    narrative_flow_quality: float
    emotional_arc: List[str]  # Emotional progression


class AdvancedMemorySynthesizer:
    """
    Advanced memory synthesizer with clustering, story weaving, and coherence validation.
    Implements sophisticated narrative construction from memory collections.
    """

    def __init__(self):
        # Clustering parameters
        self.min_cluster_size = 2
        self.max_clusters = 8
        self.coherence_threshold = 0.6

        self.llm_service = LLMService()

        # Story weaving parameters
        self.max_narrative_length = 2000  # characters
        self.min_memory_significance = 0.3

        # Coherence validation parameters
        self.temporal_consistency_weight = 0.3
        self.thematic_consistency_weight = 0.4
        self.emotional_consistency_weight = 0.3

        # TF-IDF vectorizer for text analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )

        # Story templates for different arc types
        self.story_templates = self._load_story_templates()

        logger.info("Advanced Memory Synthesizer initialized")

    def _load_story_templates(self) -> Dict[str, Dict]:
        """Load story templates for different narrative arc types"""
        return {
            StoryArcType.CHRONOLOGICAL: {
                "structure": ["beginning", "development", "climax", "resolution"],
                "transitions": ["First", "Then", "Subsequently", "Finally"],
                "focus": "temporal_sequence"
            },
            StoryArcType.THEMATIC: {
                "structure": ["theme_introduction", "exploration", "variation", "synthesis"],
                "transitions": ["Regarding", "Furthermore", "In contrast", "Overall"],
                "focus": "conceptual_development"
            },
            StoryArcType.CAUSAL: {
                "structure": ["cause", "effect", "consequence", "outcome"],
                "transitions": ["Because", "As a result", "This led to", "Ultimately"],
                "focus": "causal_relationships"
            },
            StoryArcType.EMOTIONAL: {
                "structure": ["initial_state", "trigger", "journey", "resolution"],
                "transitions": ["Initially", "When", "Through this", "Eventually"],
                "focus": "emotional_progression"
            },
            StoryArcType.IMPORTANCE: {
                "structure": ["most_critical", "supporting", "contextual", "implications"],
                "transitions": ["Most importantly", "Additionally", "In context", "This suggests"],
                "focus": "significance_hierarchy"
            },
            StoryArcType.ASSOCIATIVE: {
                "structure": ["core_concept", "associations", "connections", "insights"],
                "transitions": ["This relates to", "Similarly", "By extension", "This reveals"],
                "focus": "conceptual_networks"
            }
        }

    def cluster_memories_by_relevance(
            self, memories: List[Any], context: str = ""
    ) -> List[MemoryCluster]:
        """
        Cluster memories based on thematic relevance and temporal proximity.

        Args:
            memories: List of memory objects
            context: Current context for relevance assessment

        Returns:
            List of memory clusters
        """
        if len(memories) < 2:
            # Single memory or empty - create single cluster
            if memories:
                return [self._create_single_memory_cluster(memories[0])]
            return []

        # Extract features for clustering
        features = self._extract_memory_features(memories, context)

        # Perform clustering
        clusters = self._perform_clustering(memories, features)

        # Enhance clusters with metadata
        enhanced_clusters = []
        for i, cluster_memories in enumerate(clusters):
            cluster = self._create_enhanced_cluster(cluster_memories, f"cluster_{i}")
            enhanced_clusters.append(cluster)

        # Sort clusters by importance
        enhanced_clusters.sort(key=lambda c: c.importance_weight, reverse=True)

        logger.info(f"Created {len(enhanced_clusters)} memory clusters")

        return enhanced_clusters

    def _extract_memory_features(self, memories: List[Any], context: str) -> np.ndarray:
        """Extract features from memories for clustering"""
        # Combine text content from memories
        memory_texts = []
        for memory in memories:
            text_content = self._extract_text_from_memory(memory)
            memory_texts.append(text_content)

        # Add context to feature extraction
        if context:
            memory_texts.append(context)

        # Use TF-IDF for text features
        try:
            tfidf_features = self.tfidf_vectorizer.fit_transform(memory_texts)

            # Convert to dense array and remove context if it was added
            features = tfidf_features.toarray()
            if context:
                features = features[:-1]  # Remove context row

        except ValueError:
            # Fallback: create random features if TF-IDF fails
            logger.warning("TF-IDF failed, using random features for clustering")
            features = np.random.random((len(memories), 10))

        # Add temporal features
        temporal_features = self._extract_temporal_features(memories)

        # Add significance features
        significance_features = self._extract_significance_features(memories)

        # Combine all features
        combined_features = np.column_stack([
            features,
            temporal_features,
            significance_features
        ])

        return combined_features

    def _extract_text_from_memory(self, memory: Any) -> str:
        """Extract text content from a memory object"""
        text_parts = []

        # Extract based on memory type
        if hasattr(memory, 'content') and hasattr(memory.content, 'text_content'):
            if memory.content.text_content:
                text_parts.append(memory.content.text_content)

        if hasattr(memory, 'description') and memory.description:
            text_parts.append(memory.description)

        if hasattr(memory, 'procedure_name') and memory.procedure_name:
            text_parts.append(memory.procedure_name)

        if hasattr(memory, 'keywords') and memory.keywords:
            text_parts.extend(memory.keywords)

        # Fallback to string representation
        if not text_parts:
            text_parts.append(str(memory))

        return " ".join(text_parts)

    def _extract_temporal_features(self, memories: List[Any]) -> np.ndarray:
        """Extract temporal features from memories"""
        temporal_features = []

        # Get timestamps
        timestamps_values = []
        for memory in memories:
            if hasattr(memory, 'timestamp'):
                timestamps_values.append(memory.timestamp)
            else:
                timestamps_values.append(datetime.now().timestamp())

        # Normalize timestamps to 0-1 range
        if not timestamps_values:  # Should not happen if memories is not empty
            return np.array([])

        min_ts, max_ts = min(timestamps_values), max(timestamps_values)
        time_range = max_ts - min_ts if max_ts > min_ts else 1

        for ts in timestamps_values:
            normalized_time = (ts - min_ts) / time_range
            temporal_features.append([normalized_time])

        return np.array(temporal_features)

    def _extract_significance_features(self, memories: List[Any]) -> np.ndarray:
        """Extract significance/salience features from memories"""
        significance_features = []

        salience_mapping = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.3
        }

        for memory in memories:
            salience_score = 0.5  # Default
            if hasattr(memory, 'salience'):
                if hasattr(memory.salience, 'value'):
                    salience_score = salience_mapping.get(memory.salience.value, 0.5)
                else:
                    salience_score = salience_mapping.get(str(memory.salience), 0.5)
            significance_features.append([salience_score])

        return np.array(significance_features)

    def _perform_clustering(self, memories: List[Any], features: np.ndarray) -> List[List[Any]]:
        """Perform clustering on memory features"""
        n_memories = len(memories)

        if n_memories < 2:
            return [memories]

        # Determine optimal number of clusters
        max_k_clusters = min(self.max_clusters,
                             n_memories // self.min_cluster_size if self.min_cluster_size > 0 else n_memories)
        if max_k_clusters < 2:  # Or if features.shape[0] < 2 for k-means
            return [memories]

        if features.shape[0] < max_k_clusters:  # K-means constraint
            max_k_clusters = features.shape[0]
            if max_k_clusters < 2:
                return [memories]

        try:
            # Try K-means clustering first
            optimal_k = self._find_optimal_clusters(features, max_k_clusters)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(features)

        except Exception as e:
            logger.warning(f"K-means clustering failed: {e}, using DBSCAN")
            try:
                # Fallback to DBSCAN
                dbscan = DBSCAN(eps=0.5, min_samples=self.min_cluster_size)
                cluster_labels = dbscan.fit_predict(features)

                # Handle noise points (label -1)
                if -1 in cluster_labels:
                    noise_indices = np.where(cluster_labels == -1)[0]
                    next_label = (max(cluster_labels) + 1) if any(lbl != -1 for lbl in cluster_labels) else 0
                    for idx in noise_indices:
                        cluster_labels[idx] = next_label
                        next_label += 1


            except Exception as e2:
                logger.warning(f"DBSCAN also failed: {e2}, using single cluster")
                cluster_labels = [0] * n_memories

        # Group memories by cluster
        clusters_dict = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters_dict[label].append(memories[i])

        return list(clusters_dict.values())

    def _find_optimal_clusters(self, features: np.ndarray, max_k: int) -> int:
        """Find optimal number of clusters using elbow method"""
        if max_k < 2 or features.shape[0] < 2:
            return 1

        # Ensure k is not greater than number of samples
        num_samples = features.shape[0]
        actual_max_k = min(max_k, num_samples - 1 if num_samples > 1 else 1)
        if actual_max_k < 2:
            return 1

        inertias = []
        # K-Means requires n_samples >= n_clusters.
        k_range = range(1, min(actual_max_k + 1, num_samples))

        for k_val in k_range:
            if k_val == 0: continue  # k cannot be 0
            if k_val == 1 and num_samples >= 1:  # Handle single cluster case, variance is sum of variances of each feature
                # For k=1, inertia is sum of squared distances to centroid.
                # If only one sample, inertia is 0. If multiple, calculate.
                if num_samples == 1:
                    inertias.append(0)
                else:
                    centroid = np.mean(features, axis=0)
                    inertia = np.sum((features - centroid) ** 2)
                    inertias.append(inertia)
            elif num_samples >= k_val:
                kmeans = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
                kmeans.fit(features)
                inertias.append(kmeans.inertia_)
            else:  # Should not happen due to range adjustment
                break

        if not inertias:  # No valid k values
            return 1

        # Find elbow point
        if len(inertias) < 3:
            return len(inertias)  # Or k_range[np.argmin(inertias)] if we want min inertia

        # Calculate rate of change
        deltas = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        if not deltas: return len(inertias)

        delta_deltas = [deltas[i] - deltas[i + 1] for i in range(len(deltas) - 1)]

        # Find point with maximum second derivative (elbow)
        if delta_deltas:
            elbow_idx = np.argmax(delta_deltas)
            return k_range[elbow_idx + 2]  # +2 because k_range starts at 1, and delta_deltas is shorter
        elif deltas:  # If only one delta (2 inertia points)
            return k_range[1]  # Choose k=2
        else:  # If only one inertia point
            return k_range[0]  # Choose k=1

    def _create_single_memory_cluster(self, memory: Any) -> MemoryCluster:
        """Create a cluster from a single memory"""
        keywords = []
        if hasattr(memory, 'keywords') and memory.keywords:
            keywords = memory.keywords[:5]

        timestamp_val = datetime.now()
        if hasattr(memory, 'timestamp'):
            if isinstance(memory.timestamp, (int, float)):
                timestamp_val = datetime.fromtimestamp(memory.timestamp)
            elif isinstance(memory.timestamp, datetime):
                timestamp_val = memory.timestamp

        return MemoryCluster(
            cluster_id="single_memory_cluster",
            memories=[memory],
            centroid_keywords=keywords,
            coherence_score=1.0,
            temporal_span=(timestamp_val, timestamp_val),
            dominant_theme=self._extract_text_from_memory(memory)[:50],
            emotional_tone="neutral",
            importance_weight=0.8
        )

    def _create_enhanced_cluster(
            self, cluster_memories: List[Any], cluster_id: str
    ) -> MemoryCluster:
        """Create an enhanced cluster with metadata"""
        # Extract keywords from all memories
        all_keywords = []
        for memory in cluster_memories:
            if hasattr(memory, 'keywords') and memory.keywords:
                all_keywords.extend(memory.keywords)

        # Get most common keywords
        keyword_counts = Counter(all_keywords)
        centroid_keywords = [kw for kw, _ in keyword_counts.most_common(10)]

        # Calculate temporal span
        timestamps_list = []
        for memory in cluster_memories:
            if hasattr(memory, 'timestamp'):
                ts = memory.timestamp
                if isinstance(ts, (int, float)):
                    timestamps_list.append(datetime.fromtimestamp(ts))
                elif isinstance(ts, datetime):
                    timestamps_list.append(ts)

        if timestamps_list:
            temporal_span_val = (min(timestamps_list), max(timestamps_list))
        else:
            now = datetime.now()
            temporal_span_val = (now, now)

        # Determine dominant theme
        cluster_texts = [self._extract_text_from_memory(mem) for mem in cluster_memories]
        dominant_theme_val = self._extract_dominant_theme(cluster_texts)

        # Calculate coherence score
        coherence_score_val = self._calculate_cluster_coherence(cluster_memories)

        # Determine emotional tone
        emotional_tone_val = self._determine_cluster_emotion(cluster_memories)

        # Calculate importance weight
        importance_weight_val = self._calculate_cluster_importance(cluster_memories)

        return MemoryCluster(
            cluster_id=cluster_id,
            memories=cluster_memories,
            centroid_keywords=centroid_keywords,
            coherence_score=coherence_score_val,
            temporal_span=temporal_span_val,
            dominant_theme=dominant_theme_val,
            emotional_tone=emotional_tone_val,
            importance_weight=importance_weight_val
        )

    def _extract_dominant_theme(self, texts: List[str]) -> str:
        """Extract dominant theme from cluster texts"""
        if not texts:
            return "miscellaneous"

        # Combine all texts
        combined_text = " ".join(texts)

        # Extract key phrases (simplified)
        words = combined_text.lower().split()
        word_counts = Counter(words)

        # Filter out common words and get theme
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'
        }
        meaningful_words = [
            word for word, count in word_counts.most_common(10)
            if word not in common_words and len(word) > 2
        ]

        if meaningful_words:
            return " ".join(meaningful_words[:3])
        else:
            return "general topics"

    def _calculate_cluster_coherence(self, memories: List[Any]) -> float:
        """Calculate coherence score for a memory cluster"""
        if len(memories) < 2:
            return 1.0

        coherence_factors = []

        # Thematic coherence
        texts = [self._extract_text_from_memory(mem) for mem in memories]
        if len(texts) > 1:
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)

                # Average pairwise similarity
                n = len(texts)
                total_similarity = 0
                count = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        total_similarity += similarity_matrix[i][j]
                        count += 1

                if count > 0:
                    thematic_coherence = total_similarity / count
                    coherence_factors.append(thematic_coherence)

            except ValueError:
                coherence_factors.append(0.5)  # Default if TF-IDF fails

        # Temporal coherence
        timestamps_values = []
        for memory in memories:
            if hasattr(memory, 'timestamp'):
                ts = memory.timestamp
                # Assuming timestamp is float/int unix timestamp or datetime object
                if isinstance(ts, (int, float)):
                    timestamps_values.append(ts)
                elif isinstance(ts, datetime):
                    timestamps_values.append(ts.timestamp())

        if len(timestamps_values) > 1:
            time_gaps = []
            sorted_times = sorted(timestamps_values)
            for i in range(len(sorted_times) - 1):
                gap = abs(sorted_times[i + 1] - sorted_times[i])
                time_gaps.append(gap)

            if time_gaps:  # Ensure list is not empty for statistics.mean
                # Penalize very large time gaps
                avg_gap = statistics.mean(time_gaps)
                max_reasonable_gap = 7 * 24 * 3600  # 7 days in seconds
                temporal_coherence = max(0.1, 1.0 - (avg_gap / max_reasonable_gap))
                coherence_factors.append(temporal_coherence)
            else:  # Only one unique timestamp or issue
                coherence_factors.append(1.0)  # Perfect temporal coherence for single point in time

        return statistics.mean(coherence_factors) if coherence_factors else 0.5

    def _determine_cluster_emotion(self, memories: List[Any]) -> str:
        """Determine dominant emotional tone of cluster"""
        emotions = []

        for memory in memories:
            if hasattr(memory, 'emotional_tone') and memory.emotional_tone:
                emotions.append(memory.emotional_tone)
            elif hasattr(memory, 'primary_emotion') and memory.primary_emotion:
                emotions.append(memory.primary_emotion)

        if emotions:
            emotion_counts = Counter(emotions)
            return emotion_counts.most_common(1)[0][0]
        else:
            return "neutral"

    def _calculate_cluster_importance(self, memories: List[Any]) -> float:
        """Calculate importance weight for cluster"""
        importance_scores = []

        salience_weights = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.3
        }

        for memory in memories:
            score = 0.5  # Default
            if hasattr(memory, 'salience'):
                if hasattr(memory.salience, 'value'):
                    score = salience_weights.get(memory.salience.value, 0.5)
                else:
                    score = salience_weights.get(str(memory.salience), 0.5)
            importance_scores.append(score)

        # Weight by cluster size (larger clusters slightly more important)
        base_importance = statistics.mean(importance_scores) if importance_scores else 0.5
        size_bonus = min(0.2, len(memories) * 0.05)

        return min(1.0, base_importance + size_bonus)

    def weave_story_from_clusters(
            self,
            clusters: List[MemoryCluster],
            arc_type: StoryArcType = StoryArcType.CHRONOLOGICAL,
            context: str = ""
    ) -> StoryWeavingResult:
        """
        Weave a coherent story from memory clusters.

        Args:
            clusters: List of memory clusters
            arc_type: Type of story arc to create
            context: Current context for story relevance

        Returns:
            Story weaving result with narrative text
        """
        if not clusters:
            return StoryWeavingResult(
                narrative_text="No memories available for story construction.",
                story_arc_type=arc_type,
                coherence_score=0.0,
                memory_clusters_used=[],
                weaving_strategy="empty",
                narrative_flow_quality=0.0,
                emotional_arc=[]
            )

        # Select and order clusters based on arc type
        ordered_clusters = self._order_clusters_for_arc(clusters, arc_type)

        # Generate narrative structure
        narrative_structure = self._create_narrative_structure(ordered_clusters, arc_type)

        # Weave the actual story
        narrative_text = self._weave_narrative_text(narrative_structure, arc_type, context)

        # Calculate quality metrics
        coherence_score_val = self._calculate_narrative_coherence(narrative_text, ordered_clusters)
        flow_quality_val = self._calculate_narrative_flow(narrative_text, arc_type)
        emotional_arc_val = self._extract_emotional_arc(ordered_clusters)

        return StoryWeavingResult(
            narrative_text=narrative_text,
            story_arc_type=arc_type,
            coherence_score=coherence_score_val,
            memory_clusters_used=[c.cluster_id for c in ordered_clusters],
            weaving_strategy=f"{arc_type.value}_based",
            narrative_flow_quality=flow_quality_val,
            emotional_arc=emotional_arc_val
        )

    def _order_clusters_for_arc(
            self, clusters: List[MemoryCluster], arc_type: StoryArcType
    ) -> List[MemoryCluster]:
        """Order clusters according to the specified story arc type"""
        if arc_type == StoryArcType.CHRONOLOGICAL:
            return sorted(clusters, key=lambda c: c.temporal_span[0])
        elif arc_type == StoryArcType.IMPORTANCE:
            return sorted(clusters, key=lambda c: c.importance_weight, reverse=True)
        elif arc_type == StoryArcType.THEMATIC:
            return self._order_by_thematic_similarity(clusters)
        elif arc_type == StoryArcType.EMOTIONAL:
            return self._order_by_emotional_progression(clusters)
        elif arc_type == StoryArcType.CAUSAL:
            return self._order_by_causal_relationships(clusters)
        elif arc_type == StoryArcType.ASSOCIATIVE:
            return self._order_by_associations(clusters)
        else:
            # Default to chronological
            return sorted(clusters, key=lambda c: c.temporal_span[0])

    def _order_by_thematic_similarity(self, clusters: List[MemoryCluster]) -> List[MemoryCluster]:
        """Order clusters by thematic similarity using graph-based approach"""
        if len(clusters) <= 1:
            return clusters

        # Create similarity graph
        graph = nx.Graph()
        for i, cluster in enumerate(clusters):
            graph.add_node(i, cluster=cluster)

        # Add edges based on keyword similarity
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                similarity = self._calculate_keyword_similarity(
                    clusters[i].centroid_keywords,
                    clusters[j].centroid_keywords
                )
                if similarity > 0.2:
                    graph.add_edge(i, j, weight=similarity)

        # Find path that visits all nodes (approximation of TSP)
        try:
            # Start with highest importance cluster
            start_idx = max(range(len(clusters)), key=lambda i: clusters[i].importance_weight)

            visited = {start_idx}
            path_indices = [start_idx]
            current = start_idx

            while len(visited) < len(clusters):
                # Find unvisited neighbor with highest similarity
                best_next = None
                best_similarity = -1.0  # Initialize with a value lower than any possible similarity

                # Check neighbors of current node
                if current in graph:  # Ensure current node exists in graph (for disconnected components)
                    for neighbor in graph.neighbors(current):
                        if neighbor not in visited:
                            similarity = graph[current][neighbor]['weight']
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_next = neighbor

                if best_next is not None:
                    visited.add(best_next)
                    path_indices.append(best_next)
                    current = best_next
                else:
                    # No connected unvisited neighbor, pick unvisited with highest importance
                    remaining_indices = [i for i in range(len(clusters)) if i not in visited]
                    if remaining_indices:
                        # Select the most important among the remaining unvisited clusters
                        best_next = max(remaining_indices, key=lambda i: clusters[i].importance_weight)
                        visited.add(best_next)
                        path_indices.append(best_next)
                        current = best_next  # Update current for next iteration's neighbor search
                    else:  # All clusters visited
                        break

            return [clusters[i] for i in path_indices]

        except Exception as e:
            logger.error(f"Error ordering by thematic similarity: {e}. Falling back.")
            # Fallback to importance ordering
            return sorted(clusters, key=lambda c: c.importance_weight, reverse=True)

    def _calculate_keyword_similarity(self, keywords1: List[str], keywords2: List[str]) -> float:
        """Calculate similarity between two keyword lists (Jaccard index)"""
        if not keywords1 or not keywords2:
            return 0.0

        set1, set2 = set(keywords1), set(keywords2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def _order_by_emotional_progression(self, clusters: List[MemoryCluster]) -> List[MemoryCluster]:
        """Order clusters to create meaningful emotional progression"""
        # Emotional progression order (simplified)
        emotion_order = {
            "negative": 0, "sad": 1, "neutral": 2, "curious": 3,
            "positive": 4, "happy": 5, "excited": 6
        }

        # Sort by emotional progression, then by importance (descending)
        return sorted(
            clusters,
            key=lambda c: (emotion_order.get(c.emotional_tone, 2), -c.importance_weight)
        )

    def _order_by_causal_relationships(self, clusters: List[MemoryCluster]) -> List[MemoryCluster]:
        """Order clusters based on causal relationships"""
        # For now, use chronological order as proxy for causality
        # In full implementation, would analyze causal keywords and relationships
        return sorted(clusters, key=lambda c: c.temporal_span[0])

    def _order_by_associations(self, clusters: List[MemoryCluster]) -> List[MemoryCluster]:
        """Order clusters based on associative connections"""
        if not clusters:
            return []  # Return empty list if clusters is empty

        ordered = []
        remaining = list(clusters)  # Make a mutable copy

        if not remaining:  # Double check, should be caught by first if
            return []

        # Start with highest importance
        current = max(remaining, key=lambda c: c.importance_weight)
        ordered.append(current)
        remaining.remove(current)

        while remaining:
            # Find most associated remaining cluster
            best_match = None
            best_association = -1.0  # Initialize to allow 0 similarity matches

            for candidate in remaining:
                association = self._calculate_keyword_similarity(
                    current.centroid_keywords,
                    candidate.centroid_keywords
                )
                if association > best_association:
                    best_association = association
                    best_match = candidate

            if best_match:
                ordered.append(best_match)
                remaining.remove(best_match)
                current = best_match
            else:
                # No good association, pick most important remaining (if any)
                if remaining:  # Ensure remaining is not empty
                    next_cluster = max(remaining, key=lambda c: c.importance_weight)
                    ordered.append(next_cluster)
                    remaining.remove(next_cluster)
                    current = next_cluster
                else:  # Should not happen if loop condition is `while remaining`
                    break

        return ordered

    def _create_narrative_structure(
            self, clusters: List[MemoryCluster], arc_type: StoryArcType
    ) -> Dict[str, Any]:
        """Create narrative structure for story weaving"""
        template = self.story_templates[arc_type]
        structure_phases_keys = template["structure"]

        # Distribute clusters across structure phases
        phase_clusters_map = {}

        num_clusters = len(clusters)
        num_phases = len(structure_phases_keys)

        if num_clusters == 0 or num_phases == 0:  # Handle empty cases
            for phase_key in structure_phases_keys:
                phase_clusters_map[phase_key] = []
            return {
                "arc_type": arc_type,
                "phases": phase_clusters_map,
                "transitions": template["transitions"],
                "focus": template["focus"]
            }

        clusters_per_phase_val = max(1, num_clusters // num_phases)

        for i, phase_key in enumerate(structure_phases_keys):
            start_idx = i * clusters_per_phase_val
            end_idx = (
                (i + 1) * clusters_per_phase_val
                if i < num_phases - 1
                else num_clusters
            )
            phase_clusters_map[phase_key] = clusters[start_idx:end_idx]

        return {
            "arc_type": arc_type,
            "phases": phase_clusters_map,
            "transitions": template["transitions"],
            "focus": template["focus"]
        }

    def _weave_narrative_text(
            self, structure: Dict[str, Any], arc_type: StoryArcType, context: str
    ) -> str:
        """Weave the actual narrative text from structure"""
        narrative_parts = []
        transitions = structure["transitions"]
        phases_data = structure["phases"]

        phase_names = list(phases_data.keys())

        for i, phase_name in enumerate(phase_names):
            phase_clusters = phases_data[phase_name]

            if not phase_clusters:
                continue

            # Add transition if not first phase and transition exists
            if i > 0 and transitions and i < len(transitions):  # Check if transitions is not empty
                transition = transitions[i - 1]  # Transitions align with gaps between phases
                # So for phase i (0-indexed), transition[i-1] applies
                narrative_parts.append(f"{transition},")

            # Generate content for this phase
            phase_content = self._generate_phase_content(
                phase_clusters, phase_name, arc_type
            )
            narrative_parts.append(phase_content)

        # Join all parts
        narrative_text = " ".join(part for part in narrative_parts if part)  # Filter out empty parts

        # Apply narrative polishing
        polished_narrative = self._polish_narrative(narrative_text, arc_type, context)

        return polished_narrative

    def _generate_phase_content(
            self, clusters: List[MemoryCluster], phase_name: str, arc_type: StoryArcType
    ) -> str:
        """Generate content for a specific narrative phase"""
        if not clusters:
            return ""

        content_parts = []

        for cluster in clusters:
            # Extract key information from cluster
            cluster_summary = self._summarize_cluster(cluster, phase_name, arc_type)
            if cluster_summary:  # Append only if summary is not empty
                content_parts.append(cluster_summary)

        if not content_parts:  # All summaries were empty
            return ""

        # Combine cluster summaries for this phase
        if len(content_parts) == 1:
            return content_parts[0]
        else:
            # Connect multiple clusters within phase
            return self._connect_cluster_summaries(content_parts, phase_name)

    def _summarize_cluster(
            self, cluster: MemoryCluster, phase_name: str, arc_type: StoryArcType
    ) -> str:
        """Create a summary of a memory cluster for narrative inclusion"""
        # Extract key memories from cluster
        key_memories = sorted(
            cluster.memories,
            key=lambda m: getattr(m, 'salience', 0.5) if hasattr(m, 'salience') else (
                getattr(m.salience, 'value', 0.5) if hasattr(m, 'salience') and hasattr(m.salience, 'value') else 0.5),
            # More robust salience access
            reverse=True
        )[:3]  # Top 3 memories

        summary_parts = []

        for memory in key_memories:
            memory_text = self._extract_text_from_memory(memory)

            # Adapt summary style based on arc type and phase
            if arc_type == StoryArcType.CHRONOLOGICAL:
                formatted_text = self._format_chronological_memory(memory_text, memory)
            elif arc_type == StoryArcType.THEMATIC:
                formatted_text = self._format_thematic_memory(memory_text, cluster.dominant_theme)
            elif arc_type == StoryArcType.EMOTIONAL:
                formatted_text = self._format_emotional_memory(memory_text, cluster.emotional_tone)
            elif arc_type == StoryArcType.CAUSAL:
                formatted_text = self._format_causal_memory(memory_text, phase_name)
            else:
                formatted_text = memory_text

            if formatted_text:  # Append only if not empty
                summary_parts.append(formatted_text)

        return " ".join(summary_parts[:2])  # Limit to avoid too much detail

    def _format_chronological_memory(self, memory_text: str, memory: Any) -> str:
        """Format memory for chronological narrative"""
        time_indicator = ""
        if hasattr(memory, 'timestamp'):
            try:
                timestamp_val = None
                if isinstance(memory.timestamp, (int, float)):
                    timestamp_val = datetime.fromtimestamp(memory.timestamp)
                elif isinstance(memory.timestamp, datetime):
                    timestamp_val = memory.timestamp

                if timestamp_val:
                    # Simple time description
                    days_ago = (datetime.now() - timestamp_val).days
                    if days_ago == 0:
                        time_indicator = "today"
                    elif days_ago == 1:
                        time_indicator = "yesterday"
                    elif days_ago < 7:
                        time_indicator = f"{days_ago} days ago"
                    elif days_ago < 30:
                        time_indicator = f"{days_ago // 7} weeks ago"
                    else:
                        time_indicator = "some time ago"

            except Exception:  # More specific exceptions could be caught
                time_indicator = "previously"

        if time_indicator:
            return f"({time_indicator}) {memory_text}"
        else:
            return memory_text

    def _format_thematic_memory(self, memory_text: str, theme: str) -> str:
        """Format memory for thematic narrative"""
        return f"Regarding {theme}: {memory_text}"

    def _format_emotional_memory(self, memory_text: str, emotion: str) -> str:
        """Format memory for emotional narrative"""
        emotion_qualifiers = {
            "positive": "with satisfaction",
            "negative": "with concern",
            "neutral": "matter-of-factly",
            "curious": "with interest",
            "excited": "enthusiastically",
            "sad": "regretfully"
        }

        qualifier = emotion_qualifiers.get(emotion, "")
        if qualifier:
            return f"{memory_text} {qualifier}"
        else:
            return memory_text

    def _format_causal_memory(self, memory_text: str, phase_name: str) -> str:
        """Format memory for causal narrative"""
        causal_indicators = {
            "cause": "This situation arose when",
            "effect": "As a result,",
            "consequence": "This led to",
            "outcome": "Ultimately,"
        }

        indicator = causal_indicators.get(phase_name.lower(), "")  # Match phase_name case-insensitively
        if indicator:
            return f"{indicator} {memory_text.lower()}"
        else:
            return memory_text

    def _connect_cluster_summaries(self, summaries: List[str], phase_name: str) -> str:
        """Connect multiple cluster summaries within a phase"""
        if not summaries:  # Handle empty list
            return ""
        if len(summaries) <= 1:
            return summaries[0]

        # Choose connectors based on phase
        connectors = {
            "beginning": ["and", "also", "additionally"],
            "development": ["furthermore", "meanwhile", "in addition"],
            "climax": ["simultaneously", "at the same time", "moreover"],
            "resolution": ["finally", "consequently", "as a result"],
            "theme_introduction": ["specifically", "for instance", "notably"],
            "exploration": ["further", "additionally", "also"],
            "variation": ["alternatively", "in contrast", "however"],
            "synthesis": ["overall", "in summary", "bringing these together"]
        }

        phase_connectors_list = connectors.get(
            phase_name.lower(), ["and", "also", "additionally"]  # Match phase_name case-insensitively
        )

        connected_text = summaries[0]
        for i, summary in enumerate(summaries[1:]):
            connector = phase_connectors_list[i % len(phase_connectors_list)]
            connected_text += f", {connector} {summary.lower()}"

        return connected_text

    def _polish_narrative(
            self, narrative_text: str, arc_type: StoryArcType, context: str
    ) -> str:
        """Apply final polishing to narrative text"""
        if not narrative_text.strip():  # Handle empty or whitespace-only narrative
            return ""

        # Remove redundant phrases
        polished = self._remove_redundancies(narrative_text)

        # Ensure proper flow
        polished = self._improve_flow(polished)

        # Add context integration if relevant
        if context:
            polished = self._integrate_context(polished, context)

        # Limit length
        if len(polished) > self.max_narrative_length:
            polished = self._truncate_narrative(polished, self.max_narrative_length)

        return polished

    def _remove_redundancies(self, text: str) -> str:
        """Remove redundant phrases and words"""
        # Simple redundancy removal
        sentences = text.split('. ')
        unique_sentences = []
        seen_concepts = set()

        for sentence in sentences:
            if not sentence.strip():  # Skip empty sentences
                continue

            # Extract key words
            words = sentence.lower().split()
            key_words = {word for word in words if len(word) > 3 and word.isalnum()}  # Consider only alphanumeric words

            # Check for significant overlap with previous sentences
            overlap_ratio = (
                len(key_words.intersection(seen_concepts)) / max(len(key_words), 1)
                if key_words else 0  # Avoid division by zero if key_words is empty
            )

            if overlap_ratio < 0.7:  # Less than 70% overlap
                unique_sentences.append(sentence.strip())  # Strip sentence before adding
                seen_concepts.update(key_words)

        # Join and ensure final period if content exists
        result = '. '.join(unique_sentences)
        if result and not result.endswith('.'):
            result += '.'
        return result

    def _improve_flow(self, text: str) -> str:
        """Improve narrative flow and readability"""
        # Add proper punctuation and spacing
        improved = text.replace(' ,', ',').replace('  ', ' ')

        # Ensure sentences end properly if text is not empty
        if improved and not improved.endswith('.'):
            improved += '.'

        return improved.strip()

    def _integrate_context(self, narrative: str, context: str) -> str:
        """Integrate current context into narrative"""
        # Simple context integration
        if context and context.lower() not in narrative.lower():
            return f"In the context of {context.lower()}, {narrative.lower()}"
        return narrative

    def _truncate_narrative(self, text: str, max_length: int) -> str:
        """Intelligently truncate narrative to fit length limit"""
        if len(text) <= max_length:
            return text

        # Try to truncate at sentence boundaries
        sentences = text.split('. ')
        truncated = ""

        for sentence in sentences:
            # Check if adding the current sentence (plus a period and space) exceeds max_length
            if len(truncated + sentence + '. ') <= max_length:
                truncated += sentence + '. '
            else:
                break

        # Remove trailing space if any, from the last ". "
        truncated = truncated.rstrip()

        if not truncated:  # If no complete sentences fit, or if first sentence is too long
            # Truncate at word boundary
            words = text.split()
            truncated_words = []
            current_length = 0
            # max_length - 3 for "..."
            # max_length - 1 if it might end with a period already from word split
            allowable_len = max_length - 3 if max_length > 3 else max_length

            for word in words:
                # +1 for space after word
                if current_length + len(word) + (1 if truncated_words else 0) <= allowable_len:
                    truncated_words.append(word)
                    current_length += len(word) + (1 if truncated_words else 0)  # Add 1 for space after first word
                else:
                    break

            truncated = ' '.join(truncated_words)
            if truncated and len(truncated) < len(text):  # Add ellipsis if something was truncated
                truncated += "..."

        # Ensure it doesn't exceed max_length even after adding "..."
        if len(truncated) > max_length:
            truncated = truncated[:max_length - 3] + "..."

        return truncated.strip()

    def _calculate_narrative_coherence(
            self, narrative: str, clusters: List[MemoryCluster]
    ) -> float:
        """Calculate coherence score for the generated narrative"""
        if not narrative.strip():  # If narrative is empty, coherence is low
            return 0.0

        coherence_factors = []

        # Linguistic coherence (basic)
        sentences = [s for s in narrative.split('. ') if s.strip()]  # Filter empty sentences
        if len(sentences) > 1:
            sentence_lengths = [len(s.split()) for s in sentences]
            if sentence_lengths:  # Ensure list is not empty
                avg_sentence_length = statistics.mean(sentence_lengths)
                # Prefer moderate sentence length (10-20 words)
                length_score = max(0.3, 1.0 - abs(avg_sentence_length - 15) / 15)
                coherence_factors.append(length_score)

        # Thematic coherence
        if clusters:  # Only if clusters are provided
            cluster_themes = [c.dominant_theme for c in clusters if c.dominant_theme]
            narrative_words = set(narrative.lower().split())
            theme_words = set(' '.join(cluster_themes).lower().split())

            if theme_words:  # Avoid division by zero
                theme_overlap = len(narrative_words.intersection(theme_words))
                theme_coherence = min(1.0, theme_overlap / len(theme_words))
                coherence_factors.append(theme_coherence)
            elif not narrative_words and not theme_words:  # Both empty
                coherence_factors.append(1.0)  # Considered coherent if nothing to compare
            else:  # theme_words is empty, narrative_words might not be
                coherence_factors.append(0.0)

        # Cluster coherence integration
        cluster_coherences = [c.coherence_score for c in clusters if hasattr(c, 'coherence_score')]
        if cluster_coherences:
            avg_cluster_coherence = statistics.mean(cluster_coherences)
            coherence_factors.append(avg_cluster_coherence)

        return statistics.mean(coherence_factors) if coherence_factors else 0.5

    def _calculate_narrative_flow(self, narrative: str, arc_type: StoryArcType) -> float:
        """Calculate narrative flow quality"""
        if not narrative.strip():
            return 0.0

        flow_factors = []

        # Transition word usage
        transition_words = {
            "first", "then", "next", "subsequently", "finally", "meanwhile",
            "however", "furthermore", "additionally", "consequently", "therefore",
            "because", "since", "although", "while", "thus", "hence"
        }

        narrative_words_list = narrative.lower().split()
        if not narrative_words_list:  # If list is empty
            flow_factors.append(0.0)  # No flow if no words
        else:
            transition_count = sum(1 for word in narrative_words_list if word in transition_words)
            transition_density = transition_count / len(narrative_words_list)

            # Optimal transition density is around 3-8%
            optimal_density = 0.05
            transition_score = max(
                0.2, 1.0 - abs(transition_density - optimal_density) / optimal_density
            )
            flow_factors.append(transition_score)

        # Sentence variety
        sentences = [s for s in narrative.split('. ') if s.strip()]  # Filter empty sentences
        if len(sentences) > 1:
            sentence_lengths = [len(s.split()) for s in sentences if s.split()]  # Ensure sentences have words
            if len(sentence_lengths) > 1:  # Variance requires at least 2 data points
                length_variance = statistics.variance(sentence_lengths)
                # Moderate variance is good for flow
                variety_score = min(1.0, length_variance / 25)  # Normalize by expected variance
                flow_factors.append(variety_score)
            elif sentence_lengths:  # Only one sentence with content
                flow_factors.append(0.5)  # Neutral score for single sentence variety
            else:  # No sentences with content
                flow_factors.append(0.0)

        return statistics.mean(flow_factors) if flow_factors else 0.5

    def _extract_emotional_arc(self, clusters: List[MemoryCluster]) -> List[str]:
        """Extract emotional progression from ordered clusters"""
        return [
            cluster.emotional_tone for cluster in clusters
            if hasattr(cluster, 'emotional_tone')
        ]

    def validate_narrative_coherence(
            self, narrative: str, source_memories: List[Any]
    ) -> List[CoherenceIssue]:
        """
        Validate narrative coherence and detect issues.

        Args:
            narrative: Generated narrative text
            source_memories: Original memories used in narrative

        Returns:
            List of detected coherence issues
        """
        issues = []

        # Temporal consistency check
        temporal_issues = self._check_temporal_consistency(narrative, source_memories)
        issues.extend(temporal_issues)

        # Factual consistency check
        factual_issues = self._check_factual_consistency(narrative, source_memories)
        issues.extend(factual_issues)

        # Thematic consistency check
        thematic_issues = self._check_thematic_consistency(narrative, source_memories)
        issues.extend(thematic_issues)

        # Emotional consistency check
        emotional_issues = self._check_emotional_consistency(narrative, source_memories)
        issues.extend(emotional_issues)

        # Causal consistency check
        causal_issues = self._check_causal_consistency(narrative)
        issues.extend(causal_issues)

        if issues:  # Log only if issues were found
            logger.info(f"Detected {len(issues)} coherence issues in narrative")

        return issues

    def _check_temporal_consistency(
            self, narrative: str, memories: List[Any]
    ) -> List[CoherenceIssue]:
        """Check for temporal inconsistencies"""
        issues = []
        if not narrative.strip(): return issues

        # Extract temporal indicators from narrative
        temporal_phrases = [
            "before", "after", "then", "next", "previously", "later",
            "yesterday", "today", "last week", "recently", "earlier"
        ]

        narrative_lower = narrative.lower()
        found_temporal_indicators = [
            phrase for phrase in temporal_phrases if phrase in narrative_lower
        ]

        if found_temporal_indicators:  # Simplified: was 'len(...) > 0'
            # Get actual timestamps from memories
            timestamps_values = []
            for memory in memories:
                if hasattr(memory, 'timestamp'):
                    ts = memory.timestamp
                    if isinstance(ts, (int, float)):
                        timestamps_values.append(ts)
                    elif isinstance(ts, datetime):
                        timestamps_values.append(ts.timestamp())

            if len(timestamps_values) > 1:
                # Check if narrative temporal order matches actual chronology
                # sorted_timestamps = sorted(timestamps_values) # Not used in current simple check

                # Simple check: if narrative suggests reverse chronology but memories are forward
                # This is a simplified check - full implementation would be more sophisticated
                if "before" in narrative_lower and "after" in narrative_lower:
                    # A more robust check would involve parsing the narrative for event order
                    # and comparing with actual memory timestamps.
                    issue = CoherenceIssue(
                        issue_id=f"temporal_consistency_{len(issues)}",
                        issue_type=CoherenceIssueType.TEMPORAL_INCONSISTENCY,
                        description="Narrative contains potentially conflicting temporal indicators like 'before' and 'after' in close proximity.",
                        affected_memories=[
                            getattr(m, 'memory_id', str(i)) for i, m in enumerate(memories)
                        ],
                        severity=0.6,
                        suggested_repair="Review temporal sequence and clarify chronological order.",
                        confidence=0.7
                    )
                    issues.append(issue)

        return issues

    def _check_factual_consistency(
            self, narrative: str, memories: List[Any]
    ) -> List[CoherenceIssue]:
        """Check for factual inconsistencies"""
        issues = []
        if not narrative.strip(): return issues

        # Extract key facts from memories (simplified)
        # memory_facts = []
        # for memory in memories:
        #     memory_text = self._extract_text_from_memory(memory)
        #     memory_facts.append(memory_text)

        # Simple contradiction detection
        contradiction_pairs = [
            (["can", "able", "success", "succeed"], ["cannot", "unable", "failed", "failure"]),
            (["yes", "true", "correct", "accurate"], ["no", "false", "incorrect", "wrong"]),
            (["good", "positive", "beneficial", "helpful"], ["bad", "negative", "harmful", "detrimental"])
        ]

        narrative_lower = narrative.lower()

        for positive_terms, negative_terms in contradiction_pairs:
            # Check if terms appear close to each other or related to same subject (hard to do simply)
            # For simplicity, just check if both types of terms exist in the narrative
            has_positive = any(term in narrative_lower for term in positive_terms)
            has_negative = any(term in narrative_lower for term in negative_terms)

            if has_positive and has_negative:
                # This is a very basic check. Real factual consistency is much harder.
                issue = CoherenceIssue(
                    issue_id=f"factual_contradiction_{len(issues)}",
                    issue_type=CoherenceIssueType.FACTUAL_CONTRADICTION,
                    description="Narrative contains potentially contradictory statements based on keyword pairs.",
                    affected_memories=[
                        getattr(m, 'memory_id', str(i)) for i, m in enumerate(memories)
                    ],
                    severity=0.8,
                    suggested_repair="Resolve contradictory statements or clarify context.",
                    confidence=0.6
                )
                issues.append(issue)
                break  # Found one potential contradiction type, move on

        return issues

    def _check_thematic_consistency(
            self, narrative: str, memories: List[Any]
    ) -> List[CoherenceIssue]:
        """Check for thematic inconsistencies"""
        issues = []
        if not narrative.strip() or not memories: return issues

        # Extract themes from memories and narrative
        memory_keywords_flat = []
        for memory in memories:
            if hasattr(memory, 'keywords') and memory.keywords:
                memory_keywords_flat.extend(kw.lower() for kw in memory.keywords)

        narrative_words_set = set(narrative.lower().split())
        memory_keyword_set = set(memory_keywords_flat)

        if not memory_keyword_set and narrative_words_set:  # No keywords from memories, but narrative has words
            issue = CoherenceIssue(
                issue_id=f"thematic_drift_no_source_keywords_{len(issues)}",
                issue_type=CoherenceIssueType.THEMATIC_DRIFT,
                description="Narrative has content but source memories lack defined keywords for comparison.",
                affected_memories=[getattr(m, 'memory_id', str(i)) for i, m in enumerate(memories)],
                severity=0.5,
                suggested_repair="Ensure source memories have keywords or review narrative theme.",
                confidence=0.7
            )
            issues.append(issue)
            return issues  # Early exit if no keywords to compare against

        if not memory_keyword_set:  # No keywords from memories and narrative might be empty too
            return issues

        # Check for thematic drift
        keyword_overlap = len(narrative_words_set.intersection(memory_keyword_set))

        overlap_ratio = keyword_overlap / len(memory_keyword_set)  # Avoid division by zero by check above

        if overlap_ratio < 0.3:  # Less than 30% overlap
            issue = CoherenceIssue(
                issue_id=f"thematic_drift_{len(issues)}",
                issue_type=CoherenceIssueType.THEMATIC_DRIFT,
                description=f"Narrative themes diverge significantly from source memories (overlap: {overlap_ratio:.2f}).",
                affected_memories=[
                    getattr(m, 'memory_id', str(i)) for i, m in enumerate(memories)
                ],
                severity=0.7,
                suggested_repair="Strengthen thematic connection to source memories.",
                confidence=0.8
            )
            issues.append(issue)

        return issues

    def _check_emotional_consistency(
            self, narrative: str, memories: List[Any]
    ) -> List[CoherenceIssue]:
        """Check for emotional inconsistencies"""
        issues = []
        if not narrative.strip() or not memories: return issues

        # Extract emotions from memories
        memory_emotions_list = []
        for memory in memories:
            if hasattr(memory, 'emotional_tone') and memory.emotional_tone:
                memory_emotions_list.append(memory.emotional_tone)
            elif hasattr(memory, 'primary_emotion') and memory.primary_emotion:
                memory_emotions_list.append(memory.primary_emotion)

        if memory_emotions_list:
            # Simple emotional consistency check
            unique_emotions_set = set(memory_emotions_list)

            # Check for conflicting emotions in narrative
            positive_words = {"good", "happy", "pleased", "satisfied", "excited", "joyful", "wonderful"}
            negative_words = {"bad", "sad", "upset", "disappointed", "frustrated", "terrible", "angry"}

            narrative_lower = narrative.lower()
            # Count occurrences rather than just presence for a slightly more robust check
            positive_mentions = sum(1 for word in positive_words if word in narrative_lower)
            negative_mentions = sum(1 for word in negative_words if word in narrative_lower)

            # If memories are consistently one emotion, but narrative shows mixed signals
            if len(unique_emotions_set) == 1:
                source_emotion = list(unique_emotions_set)[0]
                is_source_positive = source_emotion in {"positive", "happy", "excited",
                                                        "curious"}  # Example positive emotions
                is_source_negative = source_emotion in {"negative", "sad", "concerned"}  # Example negative emotions

                if is_source_positive and negative_mentions > positive_mentions / 2 and negative_mentions > 0:  # If source is positive but narrative has significant negative tone
                    issue_desc = "Narrative shows negative tone despite consistently positive source memories."
                elif is_source_negative and positive_mentions > negative_mentions / 2 and positive_mentions > 0:  # If source is negative but narrative has significant positive tone
                    issue_desc = "Narrative shows positive tone despite consistently negative source memories."
                else:
                    issue_desc = None  # No clear conflict

                if issue_desc:
                    issue = CoherenceIssue(
                        issue_id=f"emotional_inconsistency_{len(issues)}",
                        issue_type=CoherenceIssueType.EMOTIONAL_DISCONTINUITY,
                        description=issue_desc,
                        affected_memories=[
                            getattr(m, 'memory_id', str(i)) for i, m in enumerate(memories)
                        ],
                        severity=0.5,
                        suggested_repair="Align narrative emotional tone with source memories.",
                        confidence=0.6
                    )
                    issues.append(issue)
            # If narrative has strong mixed signals regardless of source memories consistency
            elif positive_mentions > 1 and negative_mentions > 1:  # Arbitrary threshold for "strong" signals
                issue = CoherenceIssue(
                    issue_id=f"emotional_conflict_in_narrative_{len(issues)}",
                    issue_type=CoherenceIssueType.EMOTIONAL_DISCONTINUITY,
                    description="Narrative contains strong conflicting emotional indicators.",
                    affected_memories=[getattr(m, 'memory_id', str(i)) for i, m in enumerate(memories)],
                    severity=0.6,
                    suggested_repair="Clarify or resolve conflicting emotional tones within the narrative.",
                    confidence=0.55
                )
                issues.append(issue)

        return issues

    def _check_causal_consistency(self, narrative: str) -> List[CoherenceIssue]:
        """Check for causal inconsistencies"""
        issues = []
        if not narrative.strip(): return issues

        # Look for causal indicators
        causal_words = [
            "because", "since", "therefore", "thus", "as a result", "consequently",
            "due to", "leads to", "causes"
        ]
        narrative_lower = narrative.lower()

        found_causal_indicators = [word for word in causal_words if word in narrative_lower]

        if found_causal_indicators:  # Simplified: was 'len(...) > 0'
            # Simple check for causal coherence (very basic)
            # A full implementation would analyze actual causal chains using NLP techniques.

            # Check for potentially problematic patterns like "A because B therefore A" (circularity)
            # This is hard to detect without deeper semantic understanding.
            # For now, a very naive check for multiple strong conflicting indicators in one sentence
            sentences = narrative_lower.split('.')
            for sentence in sentences:
                # Example: if a sentence contains both "because" and "therefore" in a way that might be confusing
                # This is highly heuristic and prone to false positives/negatives.
                if ("because" in sentence and "therefore" in sentence) or \
                        ("as a result" in sentence and "due to" in sentence and sentence.find(
                            "as a result") < sentence.find("due to")):  # Effect before cause
                    issue = CoherenceIssue(
                        issue_id=f"causal_breakdown_{len(issues)}",
                        issue_type=CoherenceIssueType.CAUSAL_BREAKDOWN,
                        description="Potential circular or unclear causal reasoning detected in a sentence.",
                        affected_memories=[],  # Difficult to link to specific memories without more context
                        severity=0.4,
                        suggested_repair="Clarify causal relationships and ensure logical flow.",
                        confidence=0.5
                    )
                    issues.append(issue)
                    break  # One potential issue of this type is enough for now

        return issues

    def repair_coherence_issues(
            self, narrative: str, issues: List[CoherenceIssue], source_memories: List[Any]
    ) -> str:
        """
        Attempt to repair coherence issues in narrative.

        Args:
            narrative: Original narrative with issues
            issues: List of detected coherence issues
            source_memories: Source memories for context

        Returns:
            Repaired narrative text
        """
        repaired_narrative = narrative
        repaired_issue_count = 0

        # Sort issues by severity (desc) to address more critical ones first, if desired
        # sorted_issues = sorted(issues, key=lambda i: i.severity, reverse=True)

        for issue in issues:  # Using original order for now
            if issue.severity > 0.6:  # Only attempt to repair high-severity issues
                original_length = len(repaired_narrative)
                repaired_narrative = self._apply_issue_repair(
                    repaired_narrative, issue, source_memories
                )
                if len(repaired_narrative) != original_length:  # Crude check if repair did something
                    repaired_issue_count += 1

        if repaired_issue_count > 0:
            logger.info(f"Applied repairs for {repaired_issue_count} high-severity issues.")

        return repaired_narrative

    def _apply_issue_repair(
            self, narrative: str, issue: CoherenceIssue, memories: List[Any]
    ) -> str:
        """Apply specific repair for a coherence issue (simplified examples)"""

        if issue.issue_type == CoherenceIssueType.TEMPORAL_INCONSISTENCY:
            return self._repair_temporal_inconsistency(narrative, memories)
        elif issue.issue_type == CoherenceIssueType.FACTUAL_CONTRADICTION:
            return self._repair_factual_contradiction(narrative, memories)
        elif issue.issue_type == CoherenceIssueType.THEMATIC_DRIFT:
            return self._repair_thematic_drift(narrative, memories)
        elif issue.issue_type == CoherenceIssueType.EMOTIONAL_DISCONTINUITY:
            return self._repair_emotional_discontinuity(narrative, memories)
        elif issue.issue_type == CoherenceIssueType.CAUSAL_BREAKDOWN:
            return self._repair_causal_breakdown(narrative)
        else:
            return narrative  # No specific repair available

    def _repair_temporal_inconsistency(self, narrative: str, memories: List[Any]) -> str:
        """Repair temporal inconsistencies (simplified)"""
        # Simple repair: add clarifying temporal context if "before" and "after" appear close
        if "before" in narrative.lower() and "after" in narrative.lower():
            # This is a very basic fix. A better fix would re-order or rephrase.
            # Check if they are not already well-separated by other text
            if narrative.lower().find("before") > narrative.lower().find(
                    "after") - 30:  # If "before" appears soon after "after"
                return f"To clarify the sequence of events: {narrative}"
        return narrative

    def _repair_factual_contradiction(self, narrative: str, memories: List[Any]) -> str:
        """Repair factual contradictions (simplified)"""
        # Simple repair: add qualifying language if strong contradictions exist
        contradiction_indicators = ["however", "although", "while", "despite", "but"]

        # If narrative seems to state A and NOT A without qualification
        # This is hard to detect simply. Assume the check_factual_consistency was good enough.
        if not any(indicator in narrative.lower() for indicator in contradiction_indicators):
            sentences = narrative.split('. ')
            if len(sentences) > 1:
                # Attempt to insert a qualifying phrase. This is highly heuristic.
                # Example: find a sentence with "cannot" and a previous one with "can"
                # For now, a generic addition if issue was flagged:
                return f"Acknowledging potential complexities or differing aspects: {narrative}"
        return narrative

    def _repair_thematic_drift(self, narrative: str, memories: List[Any]) -> str:
        """Repair thematic drift (simplified)"""
        # Add thematic anchoring if memories have clear common keywords
        if not memories: return narrative

        all_memory_keywords = []
        for memory in memories:
            if hasattr(memory, 'keywords') and memory.keywords:
                all_memory_keywords.extend(kw.lower() for kw in memory.keywords)

        if all_memory_keywords:
            keyword_counts = Counter(all_memory_keywords)
            # Get, for instance, the top 1-2 most common keywords as main theme
            main_themes = [kw for kw, count in keyword_counts.most_common(2)]
            if main_themes:
                theme_str = " and ".join(main_themes)
                # Prepend thematic anchor if not already strongly present
                if theme_str not in narrative.lower()[:len(narrative) // 3]:  # Check beginning part
                    return f"Focusing on the theme of {theme_str}, {narrative.lower()}"
        return narrative

    def _repair_emotional_discontinuity(self, narrative: str, memories: List[Any]) -> str:
        """Repair emotional discontinuity (simplified)"""
        if not memories: return narrative

        # Add emotional context or smooth transitions
        memory_emotions_list = []
        for memory in memories:
            if hasattr(memory, 'emotional_tone') and memory.emotional_tone:
                memory_emotions_list.append(memory.emotional_tone)

        if memory_emotions_list:
            dominant_emotion = Counter(memory_emotions_list).most_common(1)[0][0]
            # If narrative seems to jump between emotions without bridge
            # This is hard to detect. If the issue was flagged, add a generic qualifier.
            # Example: check for sudden shifts like "happy... then suddenly sad"
            # Generic repair:
            if f"(reflecting a {dominant_emotion} experience)" not in narrative:
                # Check if the narrative already ends with a qualifier
                if not narrative.endswith(")"):
                    return narrative + f" (overall, reflecting a predominantly {dominant_emotion} experience)."
        return narrative

    def _repair_causal_breakdown(self, narrative: str) -> str:
        """Repair causal breakdown (simplified)"""
        # Simplify or clarify causal language if flagged
        # Example: replace potentially confusing chains
        repaired = narrative
        # These are very specific and might not be generally applicable or safe
        # repaired = repaired.replace("because therefore", "and so, because") # Example rephrase
        # repaired = repaired.replace("since thus", "and thus, since")       # Example rephrase

        # More generally, if "because" and "therefore" are in the same short sentence part,
        # try to rephrase or add clarifying words. This is complex.
        # For now, a simple attempt to add clarity if problem was detected:
        if "because" in narrative.lower() and "therefore" in narrative.lower():
            # Check if they are part of a common problematic pattern
            # e.g. "X because Y therefore Z" -> "Because Y, X, and therefore Z."
            # This requires more sophisticated parsing.
            # Simple intervention:
            if "the causal chain is as follows:" not in narrative.lower():
                return f"To clarify the causal connections: {narrative}"
        return repaired

    # Integration and test functions
    async def synthesize_with_advanced_features(
            self,
            memories: List[Any],
            context: str = "",
            arc_type: StoryArcType = StoryArcType.THEMATIC,
            validate_coherence: bool = False
    ) -> Dict[str, Any]:
        """
        Main integration function for advanced memory synthesis.
        V2 - Now uses an LLM with a high-quality prompt for the final synthesis.
        """
        # Step 1: Cluster memories. This is kept to provide structured context in the final output.
        clusters = self.cluster_memories_by_relevance(memories, context)

        if not memories:
            return {
                "narrative_text": "No memories were available to synthesize.",
                "narrative_coherence": 0.0,
                "clusters": []
            }

        # Step 2: Prepare the raw memory texts to feed into the synthesis prompt.
        memory_texts_to_synthesize = "\n".join(
            [f"- memory_id: {getattr(m, 'memory_id', 'unknown')}, content: '{self._extract_text_from_memory(m)}'"
             for m in memories]
        )

        # Step 3: Use the new high-quality prompt for abstractive synthesis.
        synthesis_prompt = f"""
    You are an "AI Insight Analyst". Your task is to analyze a collection of an AI's internal memory fragments and derive a single, core insight or lesson learned. You must synthesize, not concatenate.

    **Your Goal:** Create a new, abstract "meta-memory" in the first person ("I") that represents the underlying theme connecting the provided memory fragments.

    **RULES:**
    1.  **Output Format:** Your response MUST be a single, concise paragraph.
    2.  **First Person:** The insight must be written from the AI's perspective (e.g., "I learned that...", "I noticed a pattern where...").
    3.  **DO NOT COPY:** Do not repeat or copy full sentences or long phrases from the input memories. Extract the meaning, not the words.
    4.  **Abstraction:** Find the higher-level concept. If one memory is about planning and another is about an inner monologue, the insight might be about my deliberative process.
    5.  **Ignore Noise:** The input may contain messy, repetitive, or irrelevant data. Your job is to find the signal in the noise.

    **--- EXAMPLE ---**
    **Input Memories:**
    - "memory_id: mem_123, content: 'My cognitive process is an 'inner monologue'. I make a plan, then I execute steps...'"
    - "memory_id: mem_456, content: 'For complex questions, I plan first. Should I search memory? What is the most efficient path?'"
    - "memory_id: mem_789, source_type: 'user_interaction', content: 'The user's name is João.'"

    **Your Output (Example of a good synthesis):**
    I've realized that my core cognitive architecture is built on a synergy between agency and deliberation. My ability to form a plan allows me to choose the most promising path, while my 'inner monologue' provides the tools to navigate that path step-by-step, ensuring a coherent and reasoned outcome.
    **--- END OF EXAMPLE ---**

    **Now, analyze the following memories and generate your insight:**

    **Input Memories:**
    {memory_texts_to_synthesize}

    **Your Output (a single, insightful paragraph in the first person):**"""

        final_narrative_text = f"Failed to synthesize an insight from {len(memories)} memories."
        narrative_coherence = 0.3

        try:
            synthesized_insight = await self.llm_service.ainvoke(
                LLM_MODEL_SMART,
                synthesis_prompt,
                temperature=0.4
            )

            if synthesized_insight and not synthesized_insight.startswith("[LLM_ERROR]"):
                final_narrative_text = synthesized_insight.strip()
                narrative_coherence = 0.85
                logger.info(
                    f"AuraReflector-Synthesis: Insight generated successfully: '{final_narrative_text[:100]}...'")
            else:
                logger.error(f"AuraReflector-Synthesis: The LLM call for synthesis returned an error or was empty.")

        except Exception as e:
            logger.error(f"AuraReflector-Synthesis: An exception occurred during the LLM call for synthesis: {e}",
                         exc_info=True)

        return {
            "narrative_text": final_narrative_text,
            "narrative_coherence": narrative_coherence,
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "theme": c.dominant_theme,
                    "coherence": c.coherence_score,
                    "memory_count": len(c.memories),
                    "emotional_tone": c.emotional_tone,
                    "importance": c.importance_weight
                } for c in clusters
            ],
            "original_narrative": None,
            "story_arc_type": arc_type.value,
            "narrative_flow_quality": None,
            "emotional_arc": None,
            "coherence_issues": [],
        }




