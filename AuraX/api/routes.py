# api/routes.py
import logging
import json
import asyncio
import uuid

import jwt
import bcrypt
from datetime import datetime, timedelta
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import status, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Dependencies
from api.deps import get_repository, get_active_ceaf_system, get_ceaf_factory, get_db
from database.models import AgentRepository, User, Agent
from sqlalchemy import select
from ceaf_core.system import CEAFSystem
from ceaf_core.services.notification_service import NotificationService


# Configurações de Segurança
JWT_SECRET = "your-secret-key-change-this-in-production" # Mova para .env
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
security = HTTPBearer()

logger = logging.getLogger("API")

app = FastAPI(title="CEAF V4 API (Production)", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Schemas ---
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    status: str
    xi: float
    session_id: Optional[str] = None
    telemetry: Optional[dict] = None

class CloneAgentRequest(BaseModel):
    source_agent_id: str
    custom_name: str = None
    clone_memories: bool = True

# Schemas Pydantic para Auth e CRUD
class UserRegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str

class UserLoginRequest(BaseModel):
    username: str
    password: str

class CreateAgentRequest(BaseModel):
    name: str
    persona: str
    detailed_persona: str
    model: str = "openrouter/openai/gpt-4o-mini"
    settings: dict = {}

# Funções Auxiliares de Auth
def create_access_token(user_id: str, username: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode = {"user_id": user_id, "username": username, "exp": expire}
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"user_id": payload["user_id"], "username": payload["username"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expirado")
    except (jwt.InvalidTokenError, jwt.PyJWTError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido")


# --- REST Endpoints ---

@app.post("/agents/clone", tags=["Agent Management"])
async def clone_agent_endpoint(
        request: CloneAgentRequest,
        current_user: dict = Depends(verify_token),
        repo: AgentRepository = Depends(get_repository)
):
    # 1. Buscar o agente original (pode ser público ou do usuário)
    # Aqui vamos simular buscando na lista de prebuilt se for um ID conhecido, ou no banco
    # Para simplificar, vamos assumir que o ID vem da lista de prebuilt mockada

    # Mock dos dados do agente original (em produção, buscaria no DB)
    source_agent_data = {
        "name": "Agente Clonado",
        "persona": "Um clone inteligente.",
        "detailed_persona": "Você é um assistente clonado.",
        "model": "openrouter/openai/gpt-4o-mini"
    }

    # Se quiser fazer direito, busque no DB:
    # source_agent = await repo.get_agent(request.source_agent_id)
    # if source_agent: source_agent_data = ...

    new_agent_id = str(uuid.uuid4())
    name = request.custom_name or source_agent_data["name"]

    new_agent = await repo.create_agent(
        agent_id=new_agent_id,
        user_id=current_user["user_id"],
        name=name,
        persona=source_agent_data["persona"],
        detailed_persona=source_agent_data["detailed_persona"],
        model=source_agent_data["model"]
    )

    return {
        "agent_id": new_agent.id,
        "name": new_agent.name,
        "message": "Agente clonado com sucesso"
    }


@app.post("/agents/{agent_id}/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(
        request: ChatRequest,
        agent_id: str,
        current_user: dict = Depends(verify_token),
        repo: AgentRepository = Depends(get_repository)
):
    # 1. Validar Propriedade do Agente
    agent = await repo.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agente não encontrado.")

    # Permite se for dono OU se for um agente público (caso queira liberar chat com agentes publicos)
    if agent.user_id != current_user["user_id"] and agent.is_public == 0:
        raise HTTPException(status_code=403, detail="Acesso negado.")

    # 2. Gerenciar Sessão (Cria se for nulo)
    session_id = request.session_id
    if not session_id:
        session_id = str(uuid.uuid4())

    # 3. Garantir que a sessão existe no Banco (Para evitar erro de Foreign Key nas mensagens)
    await repo.create_or_get_session(session_id, current_user["user_id"], agent.id)

    # 4. Preparar Sistema CEAF
    agent_config = {
        "agent_id": agent.id,
        "name": agent.name,
        "persona": agent.detailed_persona,
        "model": agent.model,
        "settings": agent.settings
    }
    ceaf_system = CEAFSystem(agent_config)

    # 5. Recuperar Histórico
    messages = await repo.get_session_messages(session_id, limit=10)
    chat_history = [{"role": m.role, "content": m.content} for m in reversed(messages)]

    # 6. Executar Workflow (Temporal)
    # Nota: Se o worker não estiver rodando, isso vai dar timeout aqui ou no futuro
    try:
        result = await ceaf_system.process(
            query=request.message,
            session_id=session_id,
            chat_history=chat_history
        )
    except Exception as e:
        logger.error(f"Erro no processamento do CEAF: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno no núcleo cognitivo: {str(e)}")

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("response", "Erro desconhecido no agente"))

    # 7. Persistir Interação no Banco Relacional
    await repo.save_message(session_id, "user", request.message)
    await repo.save_message(session_id, "assistant", result["response"])

    # Retorna também o session_id para o frontend saber na próxima mensagem
    return ChatResponse(
        response=result["response"],
        status="success",
        xi=result.get("xi", 0.0),
        session_id=session_id  # O front precisa disso para atualizar o state.sessionId
    )


@app.post("/auth/register", status_code=status.HTTP_201_CREATED, tags=["Authentication"])
async def register_user(request: UserRegisterRequest, db=Depends(get_db)):
    # Nota: get_db vem de api.deps
    from database.models import User
    from sqlalchemy import select

    # Verifica se usuário já existe
    result = await db.execute(select(User).where((User.email == request.email) | (User.username == request.username)))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Email ou usuário já existe")

    # Hash da senha e criação
    hashed_pw = bcrypt.hashpw(request.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_user = User(email=request.email, username=request.username, password_hash=hashed_pw)

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    token = create_access_token(new_user.id, new_user.username)
    return {"user_id": new_user.id, "username": new_user.username, "access_token": token}


@app.post("/agents", status_code=status.HTTP_201_CREATED, tags=["Agent Management"])
async def create_agent_endpoint(
        request: CreateAgentRequest,
        current_user: dict = Depends(verify_token),
        repo: AgentRepository = Depends(get_repository)
):
    agent_id = str(uuid.uuid4())

    new_agent = await repo.create_agent(
        agent_id=agent_id,
        user_id=current_user["user_id"],
        name=request.name,
        persona=request.persona,
        detailed_persona=request.detailed_persona,
        model=request.model,
        settings=request.settings
    )

    # CORREÇÃO: Retornar o objeto completo que o script.js espera
    return {
        "agent_id": new_agent.id,
        "name": new_agent.name,
        "persona": new_agent.persona,
        "model": new_agent.model,
        "message": "Agente criado com sucesso.",
        "status": "ready"
    }


@app.get("/agents", tags=["Agent Management"])
async def list_user_agents(
        current_user: dict = Depends(verify_token),
        repo: AgentRepository = Depends(get_repository)
):
    """Lista todos os agentes pertencentes ao usuário logado."""
    agents = await repo.list_user_agents(current_user["user_id"])
    return agents


@app.delete("/agents/{agent_id}", tags=["Agent Management"])
async def delete_agent_endpoint(
        agent_id: str,
        current_user: dict = Depends(verify_token),
        repo: AgentRepository = Depends(get_repository)
):
    """Deleta o agente do banco de dados."""
    success = await repo.delete_agent(agent_id, current_user["user_id"])
    if not success:
        raise HTTPException(status_code=404, detail="Agente não encontrado ou acesso negado.")

    # TODO (Fase 4): Disparar evento para limpar Qdrant e Redis async
    return {"message": "Agente deletado."}


@app.post("/auth/login", tags=["Authentication"])
async def login_user(request: UserLoginRequest, db=Depends(get_db)):
    from database.models import User
    from sqlalchemy import select

    result = await db.execute(select(User).where(User.username == request.username))
    user = result.scalars().first()

    if not user or not bcrypt.checkpw(request.password.encode('utf-8'), user.password_hash.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    token = create_access_token(user.id, user.username)
    return {"user_id": user.id, "username": user.username, "access_token": token}



# --- WebSocket Endpoint ---


@app.get("/auth/me", tags=["Authentication"])
async def get_current_user_info(
        current_user: dict = Depends(verify_token),
        db=Depends(get_db)
):
    """Retorna os dados do usuário logado para o Frontend."""
    result = await db.execute(select(User).where(User.id == current_user["user_id"]))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")

    return {
        "user_id": user.id,
        "username": user.username,
        "email": user.email,
        "credits": user.credits
    }


@app.get("/models/openrouter", tags=["System"])
async def list_available_models(current_user: dict = Depends(verify_token)):
    """Lista modelos disponíveis para o frontend."""
    return {
        "Free / Low Cost": [
            {"name": "openrouter/google/gemini-2.0-flash-lite-preview-02-05:free", "cost_display": "Free", "context": 32000},
            {"name": "openrouter/deepseek/deepseek-chat:free", "cost_display": "Free", "context": 32000},
        ],
        "Smart / Reasoning": [
            {"name": "openrouter/openai/gpt-4o-mini", "cost_display": "$", "context": 128000},
            {"name": "openrouter/anthropic/claude-3.5-sonnet", "cost_display": "$$$", "context": 200000},
        ],
        "Uncensored / Creative": [
            {"name": "openrouter/nousresearch/hermes-3-llama-3.1-405b", "cost_display": "$$", "context": 8000},
        ]
    }

@app.get("/prebuilt-agents/list", tags=["Agent Management"])
async def list_prebuilt_agents(current_user: dict = Depends(verify_token)):
    """Retorna uma lista estática ou do banco de agentes públicos."""
    # Por enquanto, retornamos uma lista vazia ou mockada para o front não travar
    return [
        {
            "id": "public_socrates",
            "name": "Sócrates",
            "short_description": "O pai da filosofia ocidental. Focado em questionamento socrático.",
            "system_type": "ceaf",
            "archetype": "Philosopher",
            "rating": 4.8,
            "download_count": 120
        },
        {
            "id": "public_freud",
            "name": "Sigmund",
            "short_description": "Psicanalista focado em sonhos e subconsciente.",
            "system_type": "ceaf",
            "archetype": "Analyst",
            "rating": 4.5,
            "download_count": 85
        }
    ]


@app.get("/agents/{agent_id}", tags=["Agent Management"])
async def get_agent_details(
        agent_id: str,
        current_user: dict = Depends(verify_token),
        repo: AgentRepository = Depends(get_repository)
):
    agent = await repo.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agente não encontrado")

    # Se o agente não for do usuário e não for público, bloqueia
    if agent.user_id != current_user["user_id"] and agent.is_public == 0:
        raise HTTPException(status_code=403, detail="Acesso negado")

    return {
        "agent_id": agent.id,
        "name": agent.name,
        "persona": agent.persona,
        "detailed_persona": agent.detailed_persona,
        "model": agent.model,
        "settings": agent.settings
    }


@app.get("/agents/{agent_id}/config", tags=["Agent Configuration"])
async def get_agent_config(
        agent_id: str,
        current_user: dict = Depends(verify_token),
        repo: AgentRepository = Depends(get_repository)
):
    agent = await repo.get_agent(agent_id)
    if not agent or agent.user_id != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Agente não encontrado")

    # Retorna o settings do banco, ou um default se estiver vazio
    # Aqui estamos mapeando o JSON do banco para a estrutura que o Front espera
    default_config = {
        "llm_config": {"smart_model": agent.model, "fast_model": agent.model, "creative_model": agent.model,
                       "default_temperature": 0.7, "max_tokens_output": 1000},
        "memory_config": {"semantic_score_weight": 0.6, "keyword_score_weight": 0.4, "base_decay_rate": 0.01},
        "mcl_config": {"agency_threshold": 2.0, "baseline_coherence_bias": 0.7, "baseline_novelty_bias": 0.3},
        "drives_config": {"passive_decay_rate": 0.03, "passive_curiosity_increase": 0.05,
                          "passive_connection_increase": 0.08},
        "body_config": {"fatigue_accumulation_multiplier": 0.3, "fatigue_recovery_rate": 0.03},
        "prompts": {"gth_rendering": "", "htg_analysis": "", "agency_planning": ""}
    }

    # Merge com o que está salvo no banco (agent.settings)
    saved_settings = agent.settings if agent.settings else {}

    # Lógica simples de merge (em produção usaríamos deepmerge)
    final_config = default_config.copy()
    final_config.update(saved_settings)

    return final_config


@app.patch("/agents/{agent_id}/config", tags=["Agent Configuration"])
async def update_agent_config(
        agent_id: str,
        config_update: dict,
        current_user: dict = Depends(verify_token),
        db=Depends(get_db)  # Precisamos do DB direto para update, ou adicionar método no repo
):
    # Atualiza o campo settings no banco
    from database.models import Agent
    from sqlalchemy import update

    stmt = (
        update(Agent)
        .where(Agent.id == agent_id, Agent.user_id == current_user["user_id"])
        .values(settings=config_update)
    )
    result = await db.execute(stmt)
    await db.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Agente não encontrado ou erro ao atualizar")

    return {"status": "success", "message": "Configuração atualizada"}

@app.websocket("/ws/chat/{agent_id}/{session_id}")
async def websocket_chat(
        websocket: WebSocket,
        agent_id: str,
        session_id: str,
        repo: AgentRepository = Depends(get_repository),
        # For WebSockets, we use the factory because handling 404s inside Depends
        # during a WS handshake can be tricky (it closes the connection abruptly).
        system_factory=Depends(get_ceaf_factory)
):
    await websocket.accept()

    try:
        # Manual lookup to close with correct code if missing
        agent = await repo.get_agent(agent_id)
        if not agent:
            await websocket.close(code=4004, reason="Agent not found")
            return

        # Instantiate using factory
        ceaf = system_factory({
            "agent_id": agent.id,
            "name": agent.name,
            "persona": agent.detailed_persona,
            "model": agent.model
        })

        while True:
            # 1. Receive
            data = await websocket.receive_text()

            # 2. Stream Internal Monologue (Thinking...)
            async def stream_updates():
                async for status_json in NotificationService.listen_to_updates(session_id):
                    # Sends: {"status": "THINKING", "details": "Querying Memory...", ...}
                    await websocket.send_text(f"STATUS::{status_json}")

            listener_task = asyncio.create_task(stream_updates())

            try:
                # 3. Process
                messages = await repo.get_session_messages(session_id, limit=10)
                history = [{"role": m.role, "content": m.content} for m in reversed(messages)]

                result = await ceaf.process(data, session_id, chat_history=history)

                # 4. Save
                await repo.save_message(session_id, "user", data)
                await repo.save_message(session_id, "assistant", result["response"])

                # 5. Respond
                await websocket.send_text(json.dumps({
                    "type": "final_response",
                    "content": result["response"],
                    "xi": result.get("xi", 0.0)
                }))

            finally:
                listener_task.cancel()
                try:
                    await listener_task
                except asyncio.CancelledError:
                    pass

    except WebSocketDisconnect:
        logger.info(f"WS Disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WS Error: {e}", exc_info=True)
        await websocket.close(code=1011)