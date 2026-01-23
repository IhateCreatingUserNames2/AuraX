import torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
import numpy as np
import os
from typing import Optional, List
from pydantic import BaseModel

# --- CONFIGURAÇÃO ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PORT = int(os.getenv("PORT", "54073"))

app = FastAPI(
    title="Soul Engine API",
    description="LLM com Steering Vectors (Hormônios Cognitivos)",
    version="2.0"
)

# CORS - Permite acesso de qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SoulEngine")


# --- MODELOS PYDANTIC ---
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 800
    temperature: float = 0.7
    concept: Optional[str] = None
    intensity: float = 0.0
    layer_idx: int = 20
    top_p: float = 0.95
    repetition_penalty: float = 1.1


class GenerateResponse(BaseModel):
    text: str
    model: str
    tokens_generated: int


# --- CARREGAMENTO DO MODELO ---
logger.info(f"🧠 Carregando {MODEL_ID} em 8-bit...")

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=False
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

logger.info("✅ Modelo Carregado com Sucesso!")

# --- CARREGAMENTO DE VETORES (HORMÔNIOS) ---
vectors = {}
vector_dir = os.path.join(os.getcwd(), "vectors")

try:
    if os.path.exists(vector_dir):
        vector_files = [f for f in os.listdir(vector_dir) if f.endswith('.npy')]
        for vf in vector_files:
            concept_name = vf.replace('.npy', '')
            v = np.load(os.path.join(vector_dir, vf))
            vectors[concept_name] = torch.tensor(v, dtype=torch.float16)
            logger.info(f"🧪 Vetor carregado: {concept_name}")
    else:
        logger.warning(f"⚠️ Diretório de vetores não encontrado: {vector_dir}")
except Exception as e:
    logger.error(f"❌ Erro ao carregar vetores: {e}")

# Lista global para segurar os hooks ativos
active_hooks = []


# --- FUNÇÕES DE STEERING ---
def steering_hook(module, input, output, steering_vector, intensity):
    """Hook que injeta o vetor de steering na camada."""
    if isinstance(output, tuple):
        hidden_states = output[0]
    else:
        hidden_states = output

    target_device = hidden_states.device
    target_dtype = hidden_states.dtype

    steering_vector_ready = steering_vector.to(
        device=target_device,
        dtype=target_dtype
    )

    if hidden_states.shape[1] > 0:
        hidden_states[:, :, :] += intensity * steering_vector_ready

    if isinstance(output, tuple):
        return (hidden_states,) + output[1:]
    return hidden_states


def clear_hooks():
    """Remove todas as injeções ativas."""
    global active_hooks
    for h in active_hooks:
        h.remove()
    active_hooks = []


# --- ENDPOINTS ---
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "message": "Soul Engine is running",
        "model": MODEL_ID,
        "available_concepts": list(vectors.keys()),
        "total_layers": len(model.model.layers)
    }


@app.get("/health")
async def health():
    """Endpoint de saúde para monitoramento."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(model.device),
        "vectors_loaded": len(vectors)
    }


@app.get("/concepts")
async def list_concepts():
    """Lista todos os conceitos/vetores disponíveis."""
    return {
        "concepts": list(vectors.keys()),
        "total": len(vectors)
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Endpoint de geração sem steering.
    Compatível com OpenAI-style requests.
    """
    global active_hooks

    try:
        clear_hooks()

        # Preparação do prompt
        if "<|im_start|>" not in request.prompt:
            messages = [
                {"role": "system", "content": "You are Aura, an advanced AI assistant."},
                {"role": "user", "content": request.prompt}
            ]
            text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text_input = request.prompt

        inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

        # Configuração de stop tokens
        stop_token_ids = [tokenizer.eos_token_id]
        if "<|im_end|>" in tokenizer.get_vocab():
            stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))

        # Geração
        do_sample = request.temperature >= 1e-4
        temp = request.temperature if do_sample else 1.0

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=temp,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_token_ids,
                repetition_penalty=request.repetition_penalty,
                top_p=request.top_p
            )

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return GenerateResponse(
            text=text,
            model=MODEL_ID,
            tokens_generated=len(generated_ids)
        )

    except Exception as e:
        logger.error(f"🔥 Erro na geração: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        clear_hooks()


@app.post("/generate_with_soul", response_model=GenerateResponse)
async def generate_soul(request: GenerateRequest):
    """
    Endpoint de geração COM steering vectors (hormônios).
    """
    global active_hooks

    try:
        clear_hooks()

        # Validação do conceito
        if request.concept and request.concept not in vectors:
            raise HTTPException(
                status_code=400,
                detail=f"Conceito '{request.concept}' não encontrado. Disponíveis: {list(vectors.keys())}"
            )

        # Validação da camada
        if not (0 <= request.layer_idx < len(model.model.layers)):
            raise HTTPException(
                status_code=400,
                detail=f"Layer index {request.layer_idx} inválido. Máximo: {len(model.model.layers) - 1}"
            )

        # Aplicação do steering vector
        if request.concept and request.intensity != 0:
            logger.info(
                f"💉 Injetando '{request.concept}' "
                f"(Intensidade: {request.intensity}) na camada {request.layer_idx}"
            )
            vec = vectors[request.concept]
            layer = model.model.layers[request.layer_idx]
            hook = layer.register_forward_hook(
                lambda m, i, o: steering_hook(m, i, o, vec, request.intensity)
            )
            active_hooks.append(hook)

        # Preparação do prompt
        if "<|im_start|>" not in request.prompt:
            messages = [
                {"role": "system", "content": "You are Aura, an advanced AI assistant."},
                {"role": "user", "content": request.prompt}
            ]
            text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text_input = request.prompt

        inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

        # Stop tokens
        stop_token_ids = [tokenizer.eos_token_id]
        if "<|im_end|>" in tokenizer.get_vocab():
            stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))

        # Geração
        do_sample = request.temperature >= 1e-4
        temp = request.temperature if do_sample else 1.0

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=temp,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_token_ids,
                repetition_penalty=request.repetition_penalty,
                top_p=request.top_p
            )

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return GenerateResponse(
            text=text,
            model=MODEL_ID,
            tokens_generated=len(generated_ids)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🔥 Erro crítico: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        clear_hooks()


# --- EXCEPTION HANDLERS ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Erro não tratado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )


# --- STARTUP EVENT ---
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("🚀 Soul Engine API Started Successfully!")
    logger.info(f"📍 Model: {MODEL_ID}")
    logger.info(f"🔌 Port: {PORT}")
    logger.info(f"🧪 Concepts loaded: {len(vectors)}")
    logger.info(f"🎛️ Total layers: {len(model.model.layers)}")
    logger.info("=" * 60)


# --- MAIN ---
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )