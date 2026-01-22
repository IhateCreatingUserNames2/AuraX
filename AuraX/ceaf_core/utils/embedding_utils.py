# ceaf_core/utils/embedding_utils.py

import asyncio
import logging
import os
from typing import List, Optional, Union, Dict, Any
import numpy as np

import litellm

# Attempt to import SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)

# --- Configuration from Environment Variables (Defaults for the client instance) ---
DEFAULT_EMBEDDING_PROVIDER = os.getenv("CEAF_EMBEDDING_PROVIDER", "sentence_transformers")
DEFAULT_EMBEDDING_MODEL_FOR_CLIENT = os.getenv("CEAF_DEFAULT_EMBEDDING_MODEL",
                                               "all-MiniLM-L6-v2")  # Default for client if no type context

# --- Type-Specific Embedding Model Configuration ---
# This maps a context type (e.g., memory type) to a specific model name.
# Models can be SentenceTransformer names or LiteLLM compatible strings.
# Provider for these models is determined by EmbeddingClient's initialized provider.
# For CEAF2, memory_type strings would be like "explicit", "emotional", etc.
EMBEDDING_MODELS_CONFIG: Dict[str, str] = {
    "explicit": os.getenv("CEAF_EMBEDDING_MODEL_EXPLICIT", "BAAI/bge-small-en-v1.5"),
    "emotional": os.getenv("CEAF_EMBEDDING_MODEL_EMOTIONAL", "all-MiniLM-L6-v2"),
    "procedural": os.getenv("CEAF_EMBEDDING_MODEL_PROCEDURAL", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
    "flashbulb": os.getenv("CEAF_EMBEDDING_MODEL_FLASHBULB", "nomic-ai/nomic-embed-text-v1.5"),
    # Requires trust_remote_code
    "somatic": os.getenv("CEAF_EMBEDDING_MODEL_SOMATIC", "all-MiniLM-L6-v2"),  # Using text model as proxy
    "liminal": os.getenv("CEAF_EMBEDDING_MODEL_LIMINAL", "mixedbread-ai/mxbai-embed-large-v1"),
    "generative": os.getenv("CEAF_EMBEDDING_MODEL_GENERATIVE", "all-MiniLM-L6-v2"),
    "goal_record": os.getenv("CEAF_EMBEDDING_MODEL_GOAL", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
    # Goals are often query-like
    "kg_entity_record": os.getenv("CEAF_EMBEDDING_MODEL_KG_ENTITY", "BAAI/bge-small-en-v1.5"),  # Entities often factual
    "kg_relation_record": os.getenv("CEAF_EMBEDDING_MODEL_KG_RELATION", "all-MiniLM-L6-v2"),
    # Relations more contextual
    "default_query": os.getenv("CEAF_EMBEDDING_MODEL_QUERY", DEFAULT_EMBEDDING_MODEL_FOR_CLIENT),  # For generic queries
    "DEFAULT_FALLBACK": DEFAULT_EMBEDDING_MODEL_FOR_CLIENT  # Ultimate fallback
}


class EmbeddingClient:
    """
    A client to generate text embeddings using either local SentenceTransformer models
    or API-based models via LiteLLM, with model selection based on context type.
    """

    def __init__(
            self,
            provider: Optional[str] = None,
            default_model_name: Optional[str] = None,  # Default model if context_type not in config
            litellm_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.provider = provider or DEFAULT_EMBEDDING_PROVIDER
        self.default_model_name = default_model_name or DEFAULT_EMBEDDING_MODEL_FOR_CLIENT
        self.litellm_kwargs = litellm_kwargs or {}
        self._st_model_cache: Dict[str, SentenceTransformer] = {}  # Cache for ST models

        if self.provider not in ["litellm", "sentence_transformers"]:
            logger.error(
                f"Unsupported embedding provider: {self.provider}. Must be 'litellm' or 'sentence_transformers'.")
            # Fallback or raise error
            self.provider = "sentence_transformers" if SENTENCE_TRANSFORMER_AVAILABLE else "litellm"
            logger.warning(f"Falling back to provider: {self.provider}")

        logger.info(
            f"Initializing EmbeddingClient with default provider: '{self.provider}', default model: '{self.default_model_name}'")

        if self.provider == "sentence_transformers" and not SENTENCE_TRANSFORMER_AVAILABLE:
            msg = "SentenceTransformer provider selected, but sentence-transformers library is not installed. Embedding generation via ST will fail."
            logger.error(msg)
            # This doesn't raise immediately, allowing LiteLLM to still function if it's the one used by config.
            # Calls using ST will fail later if _get_st_model is invoked.

    def _resolve_model_name(self, context_type: Optional[str]) -> str:
        """Resolves the model name based on context_type or uses the client's default."""
        if context_type and context_type in EMBEDDING_MODELS_CONFIG:
            return EMBEDDING_MODELS_CONFIG[context_type]
        return EMBEDDING_MODELS_CONFIG.get("DEFAULT_FALLBACK", self.default_model_name)

    def _get_st_model(self, model_name_str: str) -> SentenceTransformer:
        """Loads or retrieves a SentenceTransformer model from the cache."""
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            raise RuntimeError("SentenceTransformer library is not available.")
        if model_name_str not in self._st_model_cache:
            logger.info(f"Loading SentenceTransformer model: {model_name_str}")
            try:
                # Handle models like nomic-embed-text that require trust_remote_code
                trust_code = any(
                    kc in model_name_str for kc in ["nomic-ai/", "jinaai/"])  # Add other keywords if needed
                self._st_model_cache[model_name_str] = SentenceTransformer(model_name_str, device='cpu', trust_remote_code=trust_code)
                logger.info(f"Successfully loaded and cached SentenceTransformer model: {model_name_str}")
            except Exception as e:
                logger.error(
                    f"Failed to load SentenceTransformer model '{model_name_str}': {e}. Using client default '{self.default_model_name}' if available for ST.",
                    exc_info=True)
                # Fallback to client's default ST model if specific one fails
                if self.default_model_name not in self._st_model_cache:
                    logger.info(f"Loading client's default ST model as fallback: {self.default_model_name}")
                    default_trust_code = any(kc in self.default_model_name for kc in ["nomic-ai/", "jinaai/"])
                    self._st_model_cache[self.default_model_name] = SentenceTransformer(self.default_model_name,
                                                                                        trust_remote_code=default_trust_code)
                self._st_model_cache[model_name_str] = self._st_model_cache[
                    self.default_model_name]  # Point failed model to default
        return self._st_model_cache[model_name_str]

    async def get_embedding(self, text: str, context_type: Optional[str] = None, **kwargs) -> List[float]:
        if not text or not isinstance(text, str):
            logger.warning("get_embedding received empty or invalid text input.")
            # Determine dimension for zero vector
            resolved_model_name_for_dim = self._resolve_model_name(context_type)
            dim = 384  # Default
            if self.provider == "sentence_transformers" and SENTENCE_TRANSFORMER_AVAILABLE:
                try:
                    st_model_for_dim = self._get_st_model(resolved_model_name_for_dim)
                    if hasattr(st_model_for_dim, 'get_sentence_embedding_dimension'):
                        dim = st_model_for_dim.get_sentence_embedding_dimension() or dim  # type: ignore
                except Exception:
                    pass  # Stick to default dim if model load fails for dim check
            return [0.0] * dim

        actual_model_name = self._resolve_model_name(context_type)
        logger.debug(
            f"get_embedding: Text='{text[:30]}...', ContextType='{context_type}', ResolvedModel='{actual_model_name}', Provider='{self.provider}'")

        if self.provider == "sentence_transformers":
            st_model_instance = self._get_st_model(actual_model_name)
            try:
                loop = asyncio.get_event_loop()
                embedding_array = await loop.run_in_executor(None, st_model_instance.encode, text)
                return embedding_array.tolist()
            except Exception as e:
                logger.error(f"Error with SentenceTransformer '{actual_model_name}' for text '{text[:50]}...': {e}",
                             exc_info=True)
                raise
        elif self.provider == "litellm":
            try:
                final_litellm_kwargs = {**self.litellm_kwargs, **kwargs}
                response = await litellm.aembedding(
                    model=actual_model_name, input=[text], **final_litellm_kwargs
                )
                if response.data and len(response.data) > 0:
                    return response.data[0].embedding
                else:
                    logger.error(f"LiteLLM model '{actual_model_name}' returned no data for text: '{text[:50]}...'")
                    raise ValueError("LiteLLM returned no embedding data.")
            except Exception as e:
                logger.error(f"Error with LiteLLM model '{actual_model_name}' for text '{text[:50]}...': {e}",
                             exc_info=True)
                raise
        else:
            raise RuntimeError(f"Invalid embedding provider: {self.provider}")

    async def get_embeddings(self, texts: List[str], context_type: Optional[str] = None, **kwargs) -> List[List[float]]:
        if not texts or not all(isinstance(t, str) and t for t in texts):  # Ensure all texts are non-empty strings
            logger.warning("get_embeddings received empty list or list with invalid/empty texts.")
            return [await self.get_embedding("", context_type, **kwargs) for _ in texts]  # Return zero vectors

        actual_model_name = self._resolve_model_name(context_type)
        logger.debug(
            f"get_embeddings: Count={len(texts)}, ContextType='{context_type}', ResolvedModel='{actual_model_name}', Provider='{self.provider}'")

        if self.provider == "sentence_transformers":
            st_model_instance = self._get_st_model(actual_model_name)
            try:
                loop = asyncio.get_event_loop()
                embeddings_array = await loop.run_in_executor(None, st_model_instance.encode, texts)
                return [emb.tolist() for emb in embeddings_array]
            except Exception as e:
                logger.error(f"Error batch ST '{actual_model_name}': {e}", exc_info=True)
                raise
        elif self.provider == "litellm":
            try:
                final_litellm_kwargs = {**self.litellm_kwargs, **kwargs}
                response = await litellm.aembedding(
                    model=actual_model_name, input=texts, **final_litellm_kwargs
                )
                if response.data and len(response.data) == len(texts):
                    return [item.embedding for item in response.data]
                else:
                    logger.error(f"LiteLLM batch model '{actual_model_name}' returned mismatched/no data.")
                    raise ValueError("LiteLLM returned mismatched or no embedding data for batch.")
            except Exception as e:
                logger.error(f"Error batch LiteLLM '{actual_model_name}': {e}", exc_info=True)
                raise
        else:
            raise RuntimeError(f"Invalid embedding provider: {self.provider}")


# --- Similarity Utilities ---
def cosine_similarity_np(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two numpy vectors."""
    if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
        logger.debug("cosine_similarity_np: Received non-numpy array input.")
        return 0.0
    if vec1.shape != vec2.shape or vec1.ndim != 1:  # Ensure 1D and same shape
        logger.debug(f"cosine_similarity_np: Shape mismatch or not 1D. v1: {vec1.shape}, v2: {vec2.shape}")
        return 0.0  # Or handle resizing/error appropriately

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return float(dot_product / (norm_vec1 * norm_vec2))


def compute_adaptive_similarity(embedding1: Optional[List[float]], embedding2: Optional[List[float]]) -> float:
    """
    Compute similarity between embeddings (List[float]), handling None and dimension differences.
    """
    if embedding1 is None or embedding2 is None:
        return 0.0

    np_emb1 = np.array(embedding1, dtype=np.float32)
    np_emb2 = np.array(embedding2, dtype=np.float32)

    if np_emb1.shape[0] == 0 or np_emb2.shape[0] == 0:  # Handle empty embeddings
        return 0.0

    if np_emb1.shape[0] != np_emb2.shape[0]:
        min_dim = min(np_emb1.shape[0], np_emb2.shape[0])
        # logger.warning(f"Comparing embeddings of different dimensions: {np_emb1.shape[0]} vs {np_emb2.shape[0]}. Truncating to {min_dim}.")
        truncated_emb1 = np_emb1[:min_dim]
        truncated_emb2 = np_emb2[:min_dim]

        # Simple penalty for dimension mismatch. Could be more sophisticated.
        dim_difference_ratio = abs(np_emb1.shape[0] - np_emb2.shape[0]) / max(np_emb1.shape[0], np_emb2.shape[0])
        similarity_penalty = 0.2 * dim_difference_ratio  # Up to 20% penalty based on relative difference

        base_similarity = cosine_similarity_np(truncated_emb1, truncated_emb2)
        return max(0.0, base_similarity - similarity_penalty)

    return cosine_similarity_np(np_emb1, np_emb2)


# --- Singleton Instance ---
_embedding_client_instance: Optional[EmbeddingClient] = None


def get_embedding_client(
        provider: Optional[str] = None,
        default_model_name: Optional[str] = None,
        force_reinitialize: bool = False,
        litellm_kwargs: Optional[Dict[str, Any]] = None
) -> EmbeddingClient:
    global _embedding_client_instance

    effective_provider = provider or DEFAULT_EMBEDDING_PROVIDER
    effective_default_model_name = default_model_name or DEFAULT_EMBEDDING_MODEL_FOR_CLIENT
    effective_litellm_kwargs = litellm_kwargs or {}

    if _embedding_client_instance is None or force_reinitialize or \
            _embedding_client_instance.provider != effective_provider or \
            _embedding_client_instance.default_model_name != effective_default_model_name or \
            _embedding_client_instance.litellm_kwargs != effective_litellm_kwargs:
        logger.info(
            f"Creating/Re-creating EmbeddingClient: Provider='{effective_provider}', DefaultModel='{effective_default_model_name}'")
        try:
            _embedding_client_instance = EmbeddingClient(
                provider=effective_provider,
                default_model_name=effective_default_model_name,
                litellm_kwargs=effective_litellm_kwargs
            )
        except Exception as e:
            logger.critical(f"Failed to initialize EmbeddingClient: {e}.", exc_info=True)
            _embedding_client_instance = None  # Ensure it's None on failure
            raise

    if _embedding_client_instance is None:
        raise RuntimeError("EmbeddingClient singleton is None after attempted initialization.")

    return _embedding_client_instance
