# ceaf_core/services/notification_service.py
import os
import json
import logging
import asyncio
from redis import asyncio as aioredis

logger = logging.getLogger("NotificationService")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class NotificationService:
    """
    Publishes workflow state updates to Redis channels.
    API WebSockets subscribe to these channels to push real-time updates to clients.
    """
    _redis = None

    @classmethod
    async def get_redis(cls):
        if cls._redis is None:
            cls._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        return cls._redis

    @classmethod
    async def publish_update(cls, session_id: str, status: str, details: str = None):
        """Called by Temporal Activities to signal progress."""
        redis = await cls.get_redis()
        channel = f"session:{session_id}:status"

        message = {
            "status": status,  # e.g., "THINKING", "DREAMING", "TYPING"
            "details": details,
            "timestamp": asyncio.get_event_loop().time()
        }

        await redis.publish(channel, json.dumps(message))
        # logger.debug(f"Published update to {channel}: {status}")

    @classmethod
    async def listen_to_updates(cls, session_id: str):
        """Async generator for FastAPI WebSockets."""
        redis = await cls.get_redis()
        pubsub = redis.pubsub()
        channel = f"session:{session_id}:status"

        await pubsub.subscribe(channel)

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield message["data"]
        finally:
            await pubsub.unsubscribe(channel)