# api/deps.py
from typing import AsyncGenerator, Callable
from fastapi import Depends, HTTPException, status, Path

from database.models import DatabaseSetup, AgentRepository
from ceaf_core.services.state_manager import StateManager
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.system import CEAFSystem


# --- Core Infrastructure Dependencies ---

async def get_db() -> AsyncGenerator:
    """Provides a transactional database session."""
    session_maker = DatabaseSetup.get_session_maker()
    async with session_maker() as session:
        yield session


async def get_repository() -> AgentRepository:
    """Provides the Data Access Layer."""
    return AgentRepository()


async def get_memory_service() -> MBSMemoryService:
    """Provides the Vector Memory Service."""
    return MBSMemoryService()


async def get_state_manager() -> StateManager:
    """Provides the Redis State Manager."""
    return StateManager()


# --- Application Logic Dependencies ---

async def get_ceaf_factory() -> Callable[[dict], CEAFSystem]:
    """
    Returns a factory function to create CEAFSystem instances manually.
    Useful for WebSockets or background tasks where path parameters aren't standard.
    """

    def _factory(agent_config: dict) -> CEAFSystem:
        return CEAFSystem(agent_config)

    return _factory


async def get_active_ceaf_system(
        agent_id: str = Path(..., description="The ID of the agent to interact with"),
        repo: AgentRepository = Depends(get_repository)
) -> CEAFSystem:
    """
    REAL APP DEPENDENCY:
    1. Extracts 'agent_id' from the URL.
    2. Queries PostgreSQL to validate the agent exists.
    3. Builds the configuration dictionary.
    4. Returns the initialized CEAFSystem client.

    Raises:
        HTTP 404: If the agent does not exist.
    """
    agent = await repo.get_agent(agent_id)

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID '{agent_id}' not found."
        )

    # Convert SQL Model to Config Dict
    agent_config = {
        "agent_id": agent.id,
        "name": agent.name,
        "persona": agent.detailed_persona,
        "model": agent.model,
        # Map other DB fields to config as needed
        "settings": agent.settings if hasattr(agent, "settings") else {}
    }

    return CEAFSystem(agent_config)