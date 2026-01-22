# ceaf_core/services/state_manager.py
import os
import json
import logging
import asyncio
from typing import Optional, Type, TypeVar, Dict, Any, List

from redis import asyncio as aioredis
from pydantic import BaseModel

from ceaf_core.genlang_types import VirtualBodyState, MotivationalDrives

logger = logging.getLogger("StateManager")

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

T = TypeVar("T", bound=BaseModel)


class StateManager:
    """
    Manages volatile agent state (The Nervous System) using Redis.
    Handles persistence of Body State, Drives, and short-term Context.
    """
    _redis: Optional[aioredis.Redis] = None

    @classmethod
    async def get_redis(cls) -> aioredis.Redis:
        if cls._redis is None:
            logger.info(f"StateManager: Connecting to Redis at {REDIS_URL}...")
            cls._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        return cls._redis

    @classmethod
    async def close(cls):
        if cls._redis:
            await cls._redis.close()
            cls._redis = None

    # --- Generic Helpers ---

    async def _get_model(self, key: str, model_cls: Type[T]) -> T:
        redis = await self.get_redis()
        data = await redis.get(key)
        if data:
            try:
                return model_cls.model_validate_json(data)
            except Exception as e:
                logger.error(f"StateManager: Failed to parse {key}: {e}")
        # Return default instance if missing or error
        return model_cls()

    async def _set_model(self, key: str, model: BaseModel, ttl: int = None):
        redis = await self.get_redis()
        data = model.model_dump_json()
        if ttl:
            await redis.setex(key, ttl, data)
        else:
            await redis.set(key, data)

    # --- Agent Specific State ---

    async def get_working_memory(self, agent_id: str) -> List[Dict]:
        key = f"agent:{agent_id}:wm"
        redis = await self.get_redis() # <--- Garante conexão
        data = await redis.get(key)
        return json.loads(data) if data else []

    async def save_working_memory(self, agent_id: str, wm: List[Dict]):
        key = f"agent:{agent_id}:wm"
        # Aplica Decay antes de salvar
        for item in wm:
            item['energy'] *= 0.92  # Decay rate

        # Ordena por energia e corta o excesso (Esquecimento Ativo)
        wm.sort(key=lambda x: x['energy'], reverse=True)
        wm = wm[:7]  # Limite de Miller

        redis = await self.get_redis()  # Garante a conexão
        await redis.set(key, json.dumps(wm))  # Usa a conexão garantida


    async def get_body_state(self, agent_id: str) -> VirtualBodyState:
        """Retrieves the virtual physiological state (Fatigue, Saturation)."""
        key = f"agent:{agent_id}:body_state"
        return await self._get_model(key, VirtualBodyState)

    async def save_body_state(self, agent_id: str, state: VirtualBodyState):
        key = f"agent:{agent_id}:body_state"
        # Body state persists indefinitely until reset
        await self._set_model(key, state)

    async def get_drives(self, agent_id: str) -> MotivationalDrives:
        """Retrieves the motivational drives (Curiosity, Mastery, etc.)."""
        key = f"agent:{agent_id}:drives"
        return await self._get_model(key, MotivationalDrives)

    async def save_drives(self, agent_id: str, drives: MotivationalDrives):
        key = f"agent:{agent_id}:drives"
        await self._set_model(key, drives)

    async def get_hormonal_baseline(self, agent_id: str) -> Dict[str, float]:
        """Retrieves current hormonal baselines (e.g., Cortisol level)."""
        redis = await self.get_redis()
        key = f"agent:{agent_id}:hormones"
        data = await redis.get(key)
        if data:
            return json.loads(data)
        return {"cortisol": 0.0, "dopamine": 0.5, "oxytocin": 0.5}

    async def update_hormonal_baseline(self, agent_id: str, updates: Dict[str, float]):
        """Updates specific hormones in the baseline."""
        current = await self.get_hormonal_baseline(agent_id)
        current.update(updates)
        redis = await self.get_redis()
        key = f"agent:{agent_id}:hormones"
        await redis.set(key, json.dumps(current))

    # --- Distributed Locking (for Phase 3) ---

    @staticmethod
    def lock(resource_key: str, timeout: int = 10):
        """Context manager for distributed Redis locks."""
        # This requires the redis-py lock implementation
        # return StateManager._redis.lock(resource_key, timeout=timeout)
        # Placeholder for Phase 3 implementation
        pass