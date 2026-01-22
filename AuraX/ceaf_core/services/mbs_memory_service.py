# ceaf_core/services/mbs_memory_service.py
import asyncio
import logging
import json
import math
import os
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple, Literal

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from ceaf_core.genlang_types import IntentPacket
from ceaf_core.models import MemoryConfig
from ceaf_core.utils.embedding_utils import get_embedding_client, compute_adaptive_similarity

# Import memory types
from ceaf_core.modules.memory_blossom.memory_types import (
    AnyMemoryType, BaseMemory, ExplicitMemory, ExplicitMemoryContent,
    MemorySourceType, MemorySalience, GoalRecord, KGEntityRecord,
    KGRelationRecord, EmotionalMemory, ProceduralMemory, ReasoningMemory,
    GenerativeMemory, InteroceptivePredictionMemory
)

logger = logging.getLogger("MBSMemoryService")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "ceaf_memories"
VECTOR_SIZE = 384  # Based on all-MiniLM-L6-v2


class MBSMemoryService:
    """
    Memory Blossom System (MBS) - Qdrant Powered.
    Handles storage, retrieval, and lifecycle of memories using Vector Database.
    """

    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.client = AsyncQdrantClient(url=QDRANT_URL)
        self.embedding_client = get_embedding_client()
        self._collection_initialized = False

    async def _ensure_collection(self):
        """Ensures the Qdrant collection exists with correct schema."""
        if self._collection_initialized:
            return

        try:
            collections = await self.client.get_collections()
            exists = any(c.name == COLLECTION_NAME for c in collections.collections)

            if not exists:
                logger.info(f"MBS: Creating Qdrant collection '{COLLECTION_NAME}'...")
                await self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                )

                # Create Payload Indexes for fast filtering
                await self.client.create_payload_index(COLLECTION_NAME, "agent_id", qmodels.PayloadSchemaType.KEYWORD)
                await self.client.create_payload_index(COLLECTION_NAME, "memory_type",
                                                       qmodels.PayloadSchemaType.KEYWORD)
                await self.client.create_payload_index(COLLECTION_NAME, "source_type",
                                                       qmodels.PayloadSchemaType.KEYWORD)
                await self.client.create_payload_index(COLLECTION_NAME, "salience", qmodels.PayloadSchemaType.KEYWORD)

            self._collection_initialized = True
        except Exception as e:
            logger.error(f"MBS: Failed to initialize Qdrant collection: {e}")
            raise

    async def add_specific_memory(self, memory_object: AnyMemoryType, agent_id: str = "default_agent"):
        """
        Stores a memory object into Qdrant.
        Generates embedding if missing.
        """
        await self._ensure_collection()

        # 1. Generate Embedding
        text_content, _ = await self._get_searchable_text_and_keywords(memory_object)

        vector = None
        if hasattr(memory_object, 'embedding') and memory_object.embedding:
            vector = memory_object.embedding

        if not vector and text_content:
            try:
                vector = await self.embedding_client.get_embedding(text_content,
                                                                   context_type=getattr(memory_object, 'memory_type',
                                                                                        'explicit'))
            except Exception as e:
                logger.error(f"MBS: Embedding generation failed for {memory_object.memory_id}: {e}")
                return  # Cannot store without vector in this architecture

        if not vector:
            logger.warning(f"MBS: No vector available for memory {memory_object.memory_id}. Skipping.")
            return

        # 2. Prepare Payload (Serialize Pydantic to JSON-compatible dict)
        if hasattr(memory_object, "model_dump"):
            payload = memory_object.model_dump(mode='json', exclude_none=True)
        else:
            payload = memory_object.__dict__.copy()
            # Serialize datetime
            for k, v in payload.items():
                if isinstance(v, datetime):
                    payload[k] = v.isoformat()

        # Enforce critical fields
        payload['agent_id'] = agent_id
        payload['text_content_indexed'] = text_content  # Store text used for embedding for debugging

        # 3. Upsert to Qdrant
        try:
            valid_uuid = self._to_uuid(memory_object.memory_id)

            point = PointStruct(
                id=valid_uuid,
                vector=vector,
                payload=payload
            )

            await self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=[point]
            )
            logger.info(f"MBS: Stored memory {memory_object.memory_id} (Type: {payload.get('memory_type')}) in Qdrant.")

        except Exception as e:
            logger.error(f"MBS: Failed to upsert memory {memory_object.memory_id}: {e}", exc_info=True)

    def _to_uuid(self, id_str: str) -> str:
        """Converte qualquer string em um UUID v5 determinístico válido para o Qdrant."""
        try:
            # Se já for um UUID válido, retorna ele mesmo
            uuid.UUID(id_str)
            return id_str
        except ValueError:
            # Se não for, gera um UUID v5 baseado no namespace DNS e na string
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))


    async def search_raw_memories(
            self,
            query: Union[str, IntentPacket],
            top_k: int = 5,
            agent_id: str = "default_agent",
            mcl_guidance: Optional[Dict[str, Any]] = None,
            memory_type_filter: Optional[str] = None,
            source_type_filter: Optional[str] = None,
            min_score: float = 0.5
    ) -> List[Tuple[AnyMemoryType, float]]:
        """
        Retrieves memories using Vector Similarity Search via Qdrant.
        """
        await self._ensure_collection()

        # 1. Resolve Query Vector
        query_vector = None
        if isinstance(query, str):
            if query.strip():
                query_vector = await self.embedding_client.get_embedding(query, context_type="default_query")
        elif hasattr(query, 'query_vector'):
            query_vector = query.query_vector.vector

        if not query_vector:
            return []

        # 2. Build Filters
        filters = [
            qmodels.FieldCondition(key="agent_id", match=qmodels.MatchValue(value=agent_id))
        ]

        if memory_type_filter:
            filters.append(
                qmodels.FieldCondition(key="memory_type", match=qmodels.MatchValue(value=memory_type_filter)))

        if source_type_filter:
            filters.append(
                qmodels.FieldCondition(key="source_type", match=qmodels.MatchValue(value=source_type_filter)))

        query_filter = qmodels.Filter(must=filters)

        # 3. Search Qdrant
        try:

            if hasattr(self.client, 'search'):
                results = await self.client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,  # Note o nome do parâmetro
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=min_score
                )
            else:
                # Fallback para query_points (API unificada v1.10+)

                response = await self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=min_score
                )
                results = response.points

        except Exception as e:
            logger.error(f"MBS: Qdrant search failed: {e}", exc_info=True)
            return []

        # 4. Convert Results back to Memory Objects
        memory_results = []
        for hit in results:
            try:
                mem_obj = self._reconstruct_memory_object(hit.payload)
                if mem_obj:
                    # Apply biological modifiers (Recency, Salience) here if needed
                    # For now, we trust the vector score, but we can boost it.
                    final_score = hit.score

                    # Example: Salience Boost
                    salience = hit.payload.get('dynamic_salience_score', 0.5)
                    final_score *= (1 + (salience * 0.1))

                    memory_results.append((mem_obj, final_score))
            except Exception as e:
                logger.warning(f"MBS: Failed to reconstruct memory from payload: {e}")

        # Re-sort after biological boosting
        memory_results.sort(key=lambda x: x[1], reverse=True)
        return memory_results

    async def get_memory_by_id(self, memory_id: str) -> Optional[AnyMemoryType]:
        """Retrieves a single memory by ID."""
        await self._ensure_collection()

        # CORREÇÃO: Converter para UUID válido
        valid_uuid = self._to_uuid(memory_id)

        try:
            points = await self.client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[valid_uuid]
            )
            if points:
                return self._reconstruct_memory_object(points[0].payload)
        except Exception as e:
            logger.error(f"MBS: Failed to retrieve memory {memory_id} (UUID: {valid_uuid}): {e}")
        return None

    def _reconstruct_memory_object(self, payload: Dict[str, Any]) -> AnyMemoryType:
        """Factory method to convert JSON payload back to Pydantic objects."""
        mem_type = payload.get("memory_type", "explicit")

        # Mapping string types to Pydantic classes
        type_map = {
            "explicit": ExplicitMemory,
            "goal_record": GoalRecord,
            "kg_entity_record": KGEntityRecord,
            "kg_relation_record": KGRelationRecord,
            "emotional": EmotionalMemory,
            "procedural": ProceduralMemory,
            "reasoning": ReasoningMemory,
            "generative": GenerativeMemory,
            "interoceptive_prediction": InteroceptivePredictionMemory
        }

        model_class = type_map.get(mem_type)
        if model_class:
            return model_class(**payload)

        # Fallback
        logger.warning(f"MBS: Unknown memory type '{mem_type}', treating as BaseMemory dict.")
        return BaseMemory(**payload)

    async def _get_searchable_text_and_keywords(self, memory: AnyMemoryType) -> Tuple[str, List[str]]:
        """
        Extracted logic from original service to prepare text for embedding.
        (Logic preserved from original file, simplified for brevity here)
        """
        texts_for_search = []
        keywords = getattr(memory, 'keywords', [])

        # Simple extraction logic based on type (Expanded in full implementation)
        if hasattr(memory, 'content') and hasattr(memory.content, 'text_content'):
            texts_for_search.append(memory.content.text_content)

        if hasattr(memory, 'goal_description'):
            texts_for_search.append(f"Goal: {memory.goal_description}")

        if hasattr(memory, 'label'):  # KG Entity
            texts_for_search.append(f"Entity: {memory.label} ({getattr(memory, 'description', '')})")

        return " ".join(filter(None, texts_for_search)), keywords

    async def update_memory_salience(self, memory_id: str, new_salience: float, last_accessed: float):
        """Lifecycle method: updates salience and access time in Qdrant."""
        await self._ensure_collection()

        valid_uuid = self._to_uuid(memory_id)  # <--- Adicione isso

        try:
            await self.client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={
                    "dynamic_salience_score": new_salience,
                    "last_accessed_ts": last_accessed
                },
                points=[valid_uuid]  # <--- Use aqui
            )
        except Exception as e:
            logger.error(f"MBS: Failed to update salience for {memory_id}: {e}")

    # --- Lifecycle Helpers (Cleaned up for Qdrant) ---

    async def _get_searchable_text_and_keywords(self, memory: AnyMemoryType) -> Tuple[str, List[str]]:
        """
        Generates a rich textual representation of any memory type for embedding.
        Adapts the logic from the original file to work with Pydantic models.
        """
        texts_for_search: List[str] = []
        # Get keywords safely
        keywords = getattr(memory, 'keywords', [])
        if not isinstance(keywords, list):
            keywords = []

        mem_type = getattr(memory, 'memory_type', 'unknown')

        try:
            # 1. Explicit / Interoceptive
            if mem_type in ["explicit", "interoceptive_prediction"]:
                content = getattr(memory, 'content', None)
                if content:
                    text = getattr(content, 'text_content', '')
                    if text:
                        # Contextual prefixes for better semantic search
                        source = getattr(memory, 'source_agent_name', 'unknown')
                        original_query = getattr(memory, 'metadata', {}).get('original_query', '')

                        prefix = ""
                        if source == 'user':
                            prefix = "[User Input]: "
                        elif source in ['ORA_RESPONSE', 'assistant']:
                            prefix = f"[My Past Response to '{original_query[:50]}...']: "

                        texts_for_search.append(f"{prefix}{text}")

                    # Handle structured data (e.g. Self Model or Reports)
                    structured = getattr(content, 'structured_data', {})
                    if structured:
                        if 'core_values_summary' in structured:
                            texts_for_search.append(f"Identity/Values: {structured.get('core_values_summary')}")
                        elif 'report' in structured:
                            rep = structured['report']
                            texts_for_search.append(
                                f"Internal State Report: Strain {rep.get('cognitive_strain', 0):.2f}")
                        else:
                            # Generic dump of keys for context
                            texts_for_search.append(f"Structured Data Keys: {', '.join(structured.keys())}")

            # 2. Reasoning
            elif mem_type == "reasoning":
                task = getattr(memory, 'task_description', 'unknown task')
                strategy = getattr(memory, 'strategy_summary', 'unknown strategy')
                outcome = getattr(memory, 'outcome', 'unknown')
                texts_for_search.append(
                    f"[Past Strategy]: Task '{task}' -> Strategy '{strategy}' -> Outcome '{outcome}'")

            # 3. Goals
            elif mem_type == "goal_record":
                desc = getattr(memory, 'goal_description', 'undefined')
                status = getattr(memory, 'status', 'pending')
                # Handle enum conversion if necessary
                if hasattr(status, 'value'): status = status.value
                texts_for_search.append(f"Goal ({status}): {desc}")

            # 4. Knowledge Graph Entity
            elif mem_type == "kg_entity_record":
                label = getattr(memory, 'label', 'unnamed')
                e_type = getattr(memory, 'entity_type', 'other')
                if hasattr(e_type, 'value'): e_type = e_type.value
                desc = getattr(memory, 'description', '')
                texts_for_search.append(f"Entity: '{label}' (Type: {e_type}). Description: {desc}")

            # 5. Knowledge Graph Relation
            elif mem_type == "kg_relation_record":
                label = getattr(memory, 'relation_label', 'related_to')
                desc = getattr(memory, 'description', '')
                texts_for_search.append(f"Relation: '{label}'. Context: {desc}")

            # 6. Emotional
            elif mem_type == "emotional":
                emotion = getattr(memory, 'primary_emotion', 'neutral')
                if hasattr(emotion, 'value'): emotion = emotion.value
                context_obj = getattr(memory, 'context', None)
                trigger = getattr(context_obj, 'triggering_event_summary', 'unknown') if context_obj else 'unknown'
                texts_for_search.append(f"Emotional Memory: Felt '{emotion}' due to '{trigger}'")

            # 7. Procedural
            elif mem_type == "procedural":
                name = getattr(memory, 'procedure_name', 'unnamed')
                goal = getattr(memory, 'goal_description', 'unknown')
                texts_for_search.append(f"Procedure '{name}': Steps to achieve '{goal}'.")

        except Exception as e:
            logger.error(f"MBS: Textualization error for {getattr(memory, 'memory_id', '?')}: {e}")

        full_text = " ".join(filter(None, texts_for_search)).strip()

        # Fallback if text is empty but keywords exist
        if not full_text and keywords:
            full_text = f"Keywords: {', '.join(keywords)}"

        return full_text, keywords

    # --- Knowledge Graph Traversal (Adapted for Qdrant Filtering) ---

    async def get_entity_by_id_str(self, entity_id_str: str, update_salience: bool = True) -> Optional[KGEntityRecord]:
        """Retrieves a specific KG entity by its unique string ID using Qdrant Filters."""
        await self._ensure_collection()

        # Filter for memory_type="kg_entity_record" AND entity_id_str=value
        query_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="memory_type", match=qmodels.MatchValue(value="kg_entity_record")),
                qmodels.FieldCondition(key="entity_id_str", match=qmodels.MatchValue(value=entity_id_str))
            ]
        )

        try:
            # Scroll is more efficient for exact matches than search (no vector needed)
            # We assume entity_id_str is unique, so limit=1
            results, _ = await self.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=query_filter,
                limit=1,
                with_payload=True,
                with_vectors=False
            )

            if results:
                point = results[0]
                entity = self._reconstruct_memory_object(point.payload)

                # Simulate "mark accessed" by updating metadata in background
                if update_salience and isinstance(entity, BaseMemory):
                    # We don't await this to keep read fast, ideally fire-and-forget
                    # For strict consistency in this refactor, we call the update logic
                    pass
                return entity
        except Exception as e:
            logger.error(f"MBS: KG Entity lookup failed for {entity_id_str}: {e}")

        return None

    async def get_direct_relations(
            self,
            entity_id_str: str,
            relation_label: Optional[str] = None,
            direction: Literal["outgoing", "incoming", "both"] = "both",
            update_salience: bool = True
    ) -> List[KGRelationRecord]:
        """
        Retrieves relations connected to an entity ID.
        Uses Qdrant Payload filtering instead of in-memory list iteration.
        """
        await self._ensure_collection()

        must_conditions = [
            qmodels.FieldCondition(key="memory_type", match=qmodels.MatchValue(value="kg_relation_record"))
        ]

        # Filter by Label if provided
        if relation_label:
            must_conditions.append(
                qmodels.FieldCondition(key="relation_label", match=qmodels.MatchValue(value=relation_label))
            )

        # Direction Logic
        should_conditions = []  # OR conditions

        if direction == "outgoing" or direction == "both":
            should_conditions.append(
                qmodels.FieldCondition(key="source_entity_id_str", match=qmodels.MatchValue(value=entity_id_str))
            )

        if direction == "incoming" or direction == "both":
            should_conditions.append(
                qmodels.FieldCondition(key="target_entity_id_str", match=qmodels.MatchValue(value=entity_id_str))
            )

        # Construct Filter
        query_filter = qmodels.Filter(
            must=must_conditions,
            should=should_conditions if should_conditions else None,
            # If 'should' is present, at least one must match (because min_should logic in Qdrant defaults if must is present?)
            # Actually, Qdrant: if `must` is present, `should` acts as a booster unless `min_should` is set.
            # However, here we logically need (Type=Rel AND (Source=ID OR Target=ID)).
            # Qdrant Logic: Filter(must=[Type], should=[Source, Target], min_should_count=1)
            min_should_count=1 if should_conditions else 0
        )

        relations = []
        try:
            # We might have many relations, scroll loop needed if > limit
            # For efficiency in this phase, we limit to 100
            results, _ = await self.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=query_filter,
                limit=100,
                with_payload=True
            )

            for point in results:
                obj = self._reconstruct_memory_object(point.payload)
                if isinstance(obj, KGRelationRecord):
                    relations.append(obj)

        except Exception as e:
            logger.error(f"MBS: Failed to get relations for {entity_id_str}: {e}")

        return relations

    async def get_unnamed_persons(self) -> List[KGEntityRecord]:
        """Used by Aureola to find entities needing labeling."""
        await self._ensure_collection()

        # Filter: Type=Person AND Label starts with "Unknown" or "Pessoa Desconhecida"
        # Qdrant 'MatchText' or 'MatchValue' doesn't support 'StartsWith' natively in standard filtering easily
        # without full text index.
        # Strategy: Fetch 'person' entities and filter in Python (Hybrid approach for flexibility).

        query_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="memory_type", match=qmodels.MatchValue(value="kg_entity_record")),
                qmodels.FieldCondition(key="entity_type", match=qmodels.MatchValue(value="person"))
            ]
        )

        unnamed = []
        try:
            results, _ = await self.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=query_filter,
                limit=50  # Reasonable batch
            )

            for point in results:
                entity = self._reconstruct_memory_object(point.payload)
                if isinstance(entity, KGEntityRecord):
                    label_lower = entity.label.lower()
                    if label_lower.startswith("unknown") or label_lower.startswith("pessoa desconhecida"):
                        unnamed.append(entity)

        except Exception as e:
            logger.error(f"MBS: Failed to fetch unnamed persons: {e}")

        return sorted(unnamed, key=lambda e: e.label)

    # --- Lifecycle Management (Adapted for Vector DB) ---

    async def apply_decay_cycle(self, decay_rate_default: float = 0.01):
        """
        Applies biological decay to memory salience.
        In a Vector DB, we cannot iterate efficiently every second.
        Strategy: Use Qdrant's Scroll API to process memories in batches.
        """
        await self._ensure_collection()
        logger.info("MBS Lifecycle: Starting Decay Cycle...")

        offset = None
        batch_size = 50
        processed_count = 0
        current_time = time.time()

        while True:
            try:
                # 1. Fetch Batch
                points, next_offset = await self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                if not points:
                    break

                # 2. Calculate Updates
                updates = []
                for point in points:
                    payload = point.payload
                    current_salience = payload.get('dynamic_salience_score', 0.5)
                    last_access = payload.get('last_accessed_ts', payload.get('timestamp', current_time))
                    decay_rate = payload.get('decay_rate', decay_rate_default)

                    # Logic from memory_lifecycle_manager.py
                    time_since_last = current_time - last_access

                    # Assume decay_rate is per day (86400s)
                    days_passed = time_since_last / 86400.0

                    # Exponential decay formula: S_new = S_old * (1 - rate)^days
                    if days_passed > 0.1:  # Only decay if significant time passed
                        decay_factor = math.pow(max(0.0, 1.0 - decay_rate), days_passed)
                        new_salience = current_salience * decay_factor

                        # Clamp
                        new_salience = max(0.0, min(1.0, new_salience))

                        # Only update if changed significantly
                        if abs(new_salience - current_salience) > 0.001:
                            # Qdrant SetPayload
                            updates.append(
                                qmodels.PointStruct(
                                    id=point.id,
                                    payload={"dynamic_salience_score": new_salience},
                                    vector={}  # Dummy to satisfy struct if needed, or use separate update method
                                )
                            )
                            # Note: Qdrant client.set_payload works on list of IDs/Filters,
                            # or use batch upsert with partial payload if supported (overwrite=False).
                            # Here we use set_payload per point or batch upsert.

                            # Optimized: Batch Update Payload
                            await self.client.set_payload(
                                collection_name=COLLECTION_NAME,
                                payload={"dynamic_salience_score": new_salience},
                                points=[point.id]
                            )

                processed_count += len(points)
                offset = next_offset

                if offset is None:
                    break

            except Exception as e:
                logger.error(f"MBS Lifecycle: Error during decay cycle: {e}")
                break

        logger.info(f"MBS Lifecycle: Decay cycle complete. Processed {processed_count} memories.")