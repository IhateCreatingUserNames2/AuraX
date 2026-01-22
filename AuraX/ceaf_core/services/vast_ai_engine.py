# ceaf_core/services/vast_ai_engine.py
import asyncio
import os
import aiohttp
import logging
import ssl
import json
from typing import Optional, Dict, Any

logger = logging.getLogger("VastAIEngine")


class VastAIEngine:
    """
    Client for the remote soul_engine.py running on Vast.AI.
    Handles the transmission of prompts AND RepE steering vectors.
    """

    def __init__(self, endpoint_name=None, model_name=None, timeout=3000):
        # 1. Get URL from Env (Set in docker-compose)
        env_url = os.getenv("VASTAI_ENDPOINT", "http://LOCALHOST:1111")

        # Clean URL
        base_url = env_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        # 2. Target the specific RepE endpoint
        self.endpoint_url = base_url + "/generate_with_soul"
        self.timeout = timeout

        # 3. SSL Context (Vast.AI IPs often don't have valid certs)
        self.ssl_ctx = ssl.create_default_context()
        self.ssl_ctx.check_hostname = False
        self.ssl_ctx.verify_mode = ssl.CERT_NONE

        logger.info(f"🚀 VastEngine Client Initialized pointing to: {self.endpoint_url}")

    async def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7,
                       vector_data: Optional[Dict[str, Any]] = None):
        """
        Sends generation request with optional RepE steering.
        """
        # Prepare Payload for soul_engine.py
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # RepE Parameters
            "concept": vector_data.get('concept') if vector_data else None,
            "intensity": vector_data.get('intensity', 0.0) if vector_data else 0.0,
            "layer_idx": vector_data.get('layer_idx', 16) if vector_data else 16
        }

        if vector_data:
            logger.info(f"💉 Sending RepE Injection to Vast.AI: {vector_data}")

        return await self._send_request(payload)

    async def _send_request(self, payload: Dict[str, Any]) -> str:
        try:
            timeout_settings = aiohttp.ClientTimeout(total=self.timeout)

            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ssl_ctx),
                                             timeout=timeout_settings) as session:
                async with session.post(self.endpoint_url, json=payload) as resp:

                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"❌ Vast.AI HTTP {resp.status}: {text}")
                        return f"[ERROR: Vast.AI returned {resp.status}]"

                    result = await resp.json()
                    full_text = result.get("text", "").strip()

                    # Basic Cleanup (Remove prompt echo if present)
                    sent_prompt = payload.get("prompt", "").strip()
                    if full_text.startswith(sent_prompt):
                        full_text = full_text[len(sent_prompt):].strip()

                    return full_text

        except asyncio.TimeoutError:
            logger.error("❌ Vast.AI Timeout.")
            return "[ERROR: Vast.AI Timeout]"
        except Exception as e:
            logger.error(f"❌ Vast.AI Connection Error: {e}")
            return f"[ERROR: Connection Failed: {str(e)}]"