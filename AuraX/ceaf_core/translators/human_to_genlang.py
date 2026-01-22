# ceaf_core/translators/human_to_genlang.py

import asyncio
import json
import re
from pydantic import ValidationError
from ceaf_core.genlang_types import IntentPacket, GenlangVector
from ceaf_core.utils.embedding_utils import get_embedding_client
from ceaf_core.services.llm_service import LLMService
from ceaf_core.utils.common_utils import extract_json_from_text
import logging
from ceaf_core.models import SystemPrompts, LLMConfig

logger = logging.getLogger("CEAFv3_System")


class HumanToGenlangTranslator:
    def __init__(self, prompts: SystemPrompts = None, llm_config: LLMConfig = None):
        self.embedding_client = get_embedding_client()
        # Cria o serviço de LLM passando a config, se disponível
        self.llm_service = LLMService(config=llm_config)
        self.prompts = prompts or SystemPrompts()

    def update_prompts(self, new_prompts: SystemPrompts):
        self.prompts = new_prompts

    async def translate(self, query: str, metadata: dict) -> IntentPacket:
        """
        Versão V1.2: Usa o prompt dinâmico configurado em SystemPrompts
        para garantir uma análise de intenção consistente.
        """
        logger.info(f"--- [HTG Translator v1.2] Analisando query humana: '{query[:50]}...' ---")

        analysis_prompt = self.prompts.htg_analysis.replace("{query}", query)

        # Formatação segura do prompt dinâmico
        try:
            analysis_prompt = self.prompts.htg_analysis.format(query=query)
        except KeyError:
            # Fallback se o usuário quebrou o template (ex: removeu {query} ou usou chaves erradas)
            logger.warning("Prompt HTG mal formatado pelo usuário (KeyError). Usando concatenação simples.")
            analysis_prompt = self.prompts.htg_analysis + f"\nUser Query: {query}"

        # Usa o modelo rápido definido na configuração do LLM
        # Nota: Certifique-se que LLMService tenha acesso a .config ou ajuste conforme sua implementação de LLMService
        if hasattr(self.llm_service, 'config') and self.llm_service.config:
            model_to_use = self.llm_service.config.fast_model
        else:
            # Fallback para constante global se a config não estiver acessível
            model_to_use = self.llm_service.config.fast_model

        analysis_json = None
        analysis_str = await self.llm_service.ainvoke(model_to_use, analysis_prompt, temperature=0.0)

        try:
            extracted_json = extract_json_from_text(analysis_str)
            if isinstance(extracted_json, dict):
                required_keys = ["core_query", "intent_description", "emotional_tone_description", "key_entities"]
                if all(key in extracted_json for key in required_keys):
                    analysis_json = extracted_json
                else:
                    logger.warning(
                        f"HTG Translator: Invalid JSON structure (missing keys). Raw: '{analysis_str[:150]}'")
            else:
                logger.warning(
                    f"HTG Translator: Failed to extract a dictionary from LLM response. Raw: '{analysis_str[:150]}'")

        except Exception as e:
            logger.error(f"HTG Translator: Exception during JSON parsing. Error: {e}. Raw: '{analysis_str[:150]}'")

        # Fallback aprimorado: Se a análise falhar, usa a query bruta, mas ainda tenta extrair keywords.
        if not analysis_json:
            logger.error("HTG Translator: Falha na análise da LPU. Usando fallback aprimorado.")
            fallback_keywords = list(set(re.findall(r'\b\w{3,15}\b', query.lower())))
            analysis_json = {
                "core_query": query,
                "intent_description": "unknown_intent",
                "emotional_tone_description": "unknown_emotion",
                "key_entities": fallback_keywords[:3]  # Pega até 3 palavras-chave
            }

        texts_to_embed = [
                             analysis_json.get("core_query", query),
                             analysis_json.get("intent_description", "unknown"),
                             analysis_json.get("emotional_tone_description", "unknown")
                         ] + analysis_json.get("key_entities", [])

        embeddings = await self.embedding_client.get_embeddings(texts_to_embed, context_type="default_query")

        query_vector = GenlangVector(vector=embeddings[0], source_text=analysis_json.get("core_query", query),
                                     model_name=self.embedding_client._resolve_model_name("default_query"))
        intent_vector = GenlangVector(vector=embeddings[1], source_text=analysis_json.get("intent_description"),
                                      model_name=self.embedding_client._resolve_model_name("default_query"))
        emotional_vector = GenlangVector(vector=embeddings[2],
                                         source_text=analysis_json.get("emotional_tone_description"),
                                         model_name=self.embedding_client._resolve_model_name("default_query"))
        entity_vectors = [GenlangVector(vector=emb, source_text=text,
                                        model_name=self.embedding_client._resolve_model_name("default_query")) for
                          text, emb in zip(analysis_json.get("key_entities", []), embeddings[3:])]

        intent_packet = IntentPacket(
            query_vector=query_vector,
            intent_vector=intent_vector,
            emotional_valence_vector=emotional_vector,
            entity_vectors=entity_vectors,
            metadata=metadata
        )

        logger.info(
            f"--- [HTG Translator] Análise completa. Intenção: '{intent_vector.source_text}', Entidades: {[e.source_text for e in entity_vectors]} ---")
        return intent_packet