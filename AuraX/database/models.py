# database/models.py
import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Any

from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, Text, Float, Integer, select, update, delete
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

# Environment configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://ceaf_user:ceaf_pass@localhost:5432/ceaf_db")

Base = declarative_base()


# --- Relational Models ---

class User(Base):
    __tablename__ = 'users'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    credits = Column(Integer, default=1000, nullable=False)

    # Relationships
    agents = relationship("Agent", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("ChatSession", back_populates="user")


class WhatsAppUser(Base):
    __tablename__ = 'whatsapp_users'

    phone_number = Column(String, primary_key=True, index=True)
    aura_user_id = Column(String, ForeignKey('users.id'), unique=True, nullable=False)
    aura_auth_token = Column(String, nullable=False)
    selected_agent_id = Column(String, ForeignKey('agents.id'), nullable=True)

    user = relationship("User")
    selected_agent = relationship("Agent")


class Agent(Base):
    __tablename__ = 'agents'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    name = Column(String, nullable=False)
    persona = Column(String, nullable=False)
    detailed_persona = Column(Text)
    avatar_url = Column(String, nullable=True)
    model = Column(String, default="openrouter/openai/gpt-4o-mini")
    settings = Column(JSON, default=dict)

    # Visibility & Marketplace
    is_public = Column(Integer, default=0)
    is_public_template = Column(Integer, default=0, nullable=False)
    parent_agent_id = Column(String, ForeignKey('agents.id'), nullable=True)
    version = Column(String, default="1.0.0", nullable=False)
    changelog = Column(Text, nullable=True)
    clone_count = Column(Integer, default=0, nullable=False)

    usage_cost = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Parameters
    memory_temperature = Column(Float, default=0.7)
    coherence_bias = Column(Float, default=0.6)
    novelty_bias = Column(Float, default=0.4)

    # Relationships
    user = relationship("User", back_populates="agents")
    versions = relationship("Agent", back_populates="parent", remote_side=[id])
    parent = relationship("Agent", remote_side=[parent_agent_id])
    sessions = relationship("ChatSession", back_populates="agent", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = 'sessions'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    agent_id = Column(String, ForeignKey('agents.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    state = Column(JSON, default=dict, nullable=False)

    user = relationship("User", back_populates="sessions")
    agent = relationship("Agent", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = 'messages'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey('sessions.id'), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


class CreditTransaction(Base):
    __tablename__ = 'credit_transactions'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    agent_id = Column(String, nullable=True)
    amount = Column(Integer, nullable=False)
    model_used = Column(String, nullable=True)
    description = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")


# --- Async Repository ---

class DatabaseSetup:
    _engine = None
    _sessionmaker = None

    @classmethod
    def get_engine(cls):
        if cls._engine is None:
            cls._engine = create_async_engine(DATABASE_URL, echo=False, future=True)
        return cls._engine

    @classmethod
    def get_session_maker(cls):
        if cls._sessionmaker is None:
            engine = cls.get_engine()
            cls._sessionmaker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        return cls._sessionmaker

    @classmethod
    async def init_db(cls):
        """Create tables if they don't exist."""
        async with cls.get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


class AgentRepository:
    """Async repository for relational data operations."""

    def __init__(self):
        self.session_maker = DatabaseSetup.get_session_maker()

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        async with self.session_maker() as session:
            result = await session.execute(select(User).where(User.id == user_id))
            return result.scalars().first()

    async def create_agent(self, agent_id: str, user_id: str, name: str, persona: str,
                           detailed_persona: str, model: str = None, is_public: bool = False,
                           settings: dict = None) -> Agent:
        async with self.session_maker() as session:
            agent = Agent(
                id=agent_id,
                user_id=user_id,
                name=name,
                persona=persona,
                detailed_persona=detailed_persona,
                model=model or "openrouter/openai/gpt-4o-mini",
                is_public=1 if is_public else 0,
                settings=settings or {}
            )
            session.add(agent)
            await session.commit()
            await session.refresh(agent)
            return agent

    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        async with self.session_maker() as session:
            result = await session.execute(select(Agent).where(Agent.id == agent_id))
            return result.scalars().first()

    async def list_user_agents(self, user_id: str) -> List[Agent]:
        async with self.session_maker() as session:
            result = await session.execute(select(Agent).where(Agent.user_id == user_id))
            return result.scalars().all()

    async def delete_agent(self, agent_id: str, user_id: str) -> bool:
        async with self.session_maker() as session:
            result = await session.execute(
                select(Agent).where(Agent.id == agent_id, Agent.user_id == user_id)
            )
            agent = result.scalars().first()
            if agent:
                await session.delete(agent)
                await session.commit()
                return True
            return False

    async def save_message(self, session_id: str, role: str, content: str) -> Message:
        async with self.session_maker() as session:
            message = Message(
                session_id=session_id,
                role=role,
                content=content
            )
            session.add(message)
            await session.commit()
            return message

    async def create_or_get_session(self, session_id: str, user_id: str, agent_id: str) -> ChatSession:
        async with self.session_maker() as session:
            # Tenta buscar
            result = await session.execute(select(ChatSession).where(ChatSession.id == session_id))
            existing_session = result.scalars().first()

            if existing_session:
                return existing_session

            # Se não existe, cria
            new_session = ChatSession(
                id=session_id,
                user_id=user_id,
                agent_id=agent_id,
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow()
            )
            session.add(new_session)
            await session.commit()
            return new_session


    async def get_session_messages(self, session_id: str, limit: int = 50) -> List[Message]:
        async with self.session_maker() as session:
            result = await session.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
            )
            return result.scalars().all()

    async def get_recently_active_agent_ids(self, hours: int = 48) -> List[str]:
        async with self.session_maker() as session:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            result = await session.execute(
                select(ChatSession.agent_id)
                .where(ChatSession.last_active >= cutoff_time)
                .distinct()
            )
            return result.scalars().all()