# ceaf_core/services/user_profiling_service.py
import logging
import time
import json
from typing import Optional
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.genlang_types import UserRepresentation
from ceaf_core.modules.memory_blossom.memory_types import ExplicitMemory, ExplicitMemoryContent, MemorySourceType, \
    MemorySalience

logger = logging.getLogger("UserProfileService")


class UserProfilingService:
    def __init__(self, memory_service: MBSMemoryService):
        self.memory_service = memory_service

    def _get_memory_id(self, user_id: str) -> str:
        return f"user_profile_{user_id}"

    async def get_user_profile(self, user_id: str) -> UserRepresentation:
        """Carrega o perfil do usuário do MBS ou cria um novo."""
        mem_id = self._get_memory_id(user_id)

        try:
            mem = await self.memory_service.get_memory_by_id(mem_id)
            if mem and hasattr(mem, 'content') and mem.content.structured_data:
                return UserRepresentation(**mem.content.structured_data)
        except Exception as e:
            logger.warning(f"Erro ao carregar perfil do usuário {user_id}: {e}")

        # Retorna novo se não existir
        return UserRepresentation(user_id=user_id)

    async def update_user_profile(self, user_id: str, updates: dict):
        """Atualiza campos específicos e salva."""
        profile = await self.get_user_profile(user_id)

        # Atualiza campos
        for k, v in updates.items():
            if hasattr(profile, k):
                setattr(profile, k, v)

        profile.last_updated = time.time()
        profile.interaction_count += 1

        # Salva no MBS
        mem_id = self._get_memory_id(user_id)

        content = ExplicitMemoryContent(
            text_content=f"Perfil do Usuário {user_id}. Estilo: {profile.communication_style}. Emoção: {profile.emotional_state}.",
            structured_data=profile.model_dump()
        )

        memory = ExplicitMemory(
            memory_id=mem_id,
            content=content,
            memory_type="explicit",  # Ou criar um tipo 'user_model' específico
            source_type=MemorySourceType.INTERNAL_REFLECTION,
            salience=MemorySalience.HIGH,
            keywords=["user_profile", "meta_memory"],
            agent_id="system"  # Perfil é do sistema, não de um agente específico (ou pode ser por agente)
        )

        await self.memory_service.add_specific_memory(memory)
        logger.info(f"Perfil do usuário {user_id} atualizado.")