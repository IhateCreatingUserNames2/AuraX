# ceaf_core/services/llm_service.py
import os
import logging
import asyncio
from typing import Optional, Dict, Any
import litellm

from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

# Import the Vast Client
from ceaf_core.services.vast_ai_engine import VastAIEngine
from ceaf_core.models import LLMConfig

from ceaf_core.utils.embedding_utils import get_embedding_client

logger = logging.getLogger("LLMService")


class LLMService:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.inference_mode = os.getenv("INFERENCE_MODE", "openrouter")  # 'vastai' or 'openrouter'
        self.vast_engine = None

        if self.inference_mode == "vastai":
            self.vast_engine = VastAIEngine(timeout=self.config.timeout_seconds)

        self.embedding_client = get_embedding_client()

    async def ainvoke(
            self,
            model: str,
            prompt: str,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            vector_data: Optional[Dict[str, Any]] = None  # <--- Critical Argument
    ) -> str:

        eff_temp = temperature if temperature is not None else self.config.default_temperature
        eff_tokens = max_tokens if max_tokens is not None else self.config.max_tokens_output

        # --- ROUTE TO VAST.AI (WITH RepE) ---
        if self.inference_mode == "vastai" and self.vast_engine:
            # Note: We ignore the 'model' string arg here because Vast runs a specific loaded model
            return await self.vast_engine.generate(
                prompt=prompt,
                max_tokens=eff_tokens,
                temperature=eff_temp,
                vector_data=vector_data  # Pass the hormones!
            )

        # --- FALLBACK TO OPENROUTER (NO RepE) ---
        # If we are in OpenRouter mode, we just ignore vector_data because APIs don't support it.
        if vector_data and self.inference_mode != "vastai":
            logger.warning("⚠️ Vector data provided but Inference Mode is NOT Vast.AI. Hormones will be ignored.")

        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=eff_temp,
                max_tokens=eff_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return f"[Error: {e}]"