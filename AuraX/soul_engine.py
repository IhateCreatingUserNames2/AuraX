import torch
import uvicorn
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
import numpy as np
import os

# --- CONFIGURAÇÃO ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
app = FastAPI()

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SoulEngine")

# --- CARREGAMENTO DO MODELO (MODO TURBO 8-BIT) ---
print(f"🧠 Carregando {MODEL_ID} em 8-bit...")

# Isso reduz o uso de VRAM de 15GB para ~8GB, deixando a 3090 livre para processar rápido
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=False
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("✅ Modelo Carregado com Sucesso!")

# --- CARREGAMENTO DE VETORES (HORMÔNIOS) ---
vectors = {}
try:
    # Tenta carregar vetores locais se existirem
    vector_files = [f for f in os.listdir('.') if f.endswith('.npy')]
    for vf in vector_files:
        concept_name = vf.replace('.npy', '')
        v = np.load(vf)
        # Mantemos na CPU por enquanto, movemos sob demanda
        vectors[concept_name] = torch.tensor(v, dtype=torch.float16)
        print(f"🧪 Vetor carregado: {concept_name}")
except Exception as e:
    print(f"⚠️ Aviso ao carregar vetores: {e}")

# Lista global para segurar os hooks ativos
active_hooks = []


def steering_hook(module, input, output, steering_vector, intensity):
    """
    Hook Blindado: Injeta o vetor na camada, adaptando-se automaticamente
    ao dispositivo (CPU/GPU) e tipo de dado da camada.
    """
    if isinstance(output, tuple):
        hidden_states = output[0]
    else:
        hidden_states = output

    # Descobre onde essa camada está vivendo
    target_device = hidden_states.device
    target_dtype = hidden_states.dtype

    # Prepara o vetor para bater com o local e o tipo da camada
    steering_vector_ready = steering_vector.to(device=target_device, dtype=target_dtype)

    # Injeta a "dose" hormonal em todos os tokens da sequência
    if hidden_states.shape[1] > 0:
        hidden_states[:, :, :] += intensity * steering_vector_ready

    if isinstance(output, tuple):
        return (hidden_states,) + output[1:]
    return hidden_states


def clear_hooks():
    """Remove todas as injeções ativas para limpar o modelo."""
    global active_hooks
    for h in active_hooks:
        h.remove()
    active_hooks = []


@app.post("/generate_with_soul")
async def generate_soul(request: Request):
    global active_hooks

    try:
        data = await request.json()

        prompt = data.get("prompt", "")
        max_tokens = int(data.get("max_tokens", 500))
        temperature = float(data.get("temperature", 0.7))

        # Parâmetros de Injeção
        concept_name = data.get("concept", None)
        intensity = float(data.get("intensity", 0.0))
        layer_idx = int(data.get("layer_idx", 16))

        # Ajuste de temperatura para evitar erro de divisão por zero
        if temperature < 1e-4:
            do_sample = False
            temperature = 1.0
        else:
            do_sample = True

        # --- 1. APLICAÇÃO DOS GANCHOS (STEERING) ---
        clear_hooks()  # Limpa resquícios anteriores

        if concept_name and concept_name in vectors and intensity != 0:
            logger.info(f"💉 Injetando '{concept_name}' (Força: {intensity}) na camada {layer_idx}")
            vec = vectors[concept_name]

            # Proteção de índice de camada
            if 0 <= layer_idx < len(model.model.layers):
                layer = model.model.layers[layer_idx]

                # Registra o hook
                hook = layer.register_forward_hook(
                    lambda m, i, o: steering_hook(m, i, o, vec, intensity)
                )
                active_hooks.append(hook)
            else:
                logger.error(f"❌ Índice de camada inválido: {layer_idx}")

        # --- 2. PREPARAÇÃO DA GERAÇÃO ---
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        # Definição dos STOP TOKENS (Freio de Mão)
        # O Qwen usa <|im_end|> para sinalizar que parou de falar
        stop_token_ids = [tokenizer.eos_token_id]
        if "<|im_end|>" in tokenizer.get_vocab():
            stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))

        # --- 3. GERAÇÃO ---
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,

                # CORREÇÃO: Adicione estes parâmetros para parar a alucinação
                eos_token_id=stop_token_ids,  # Importante: Para quando termina a frase
                repetition_penalty=1.1,  # Importante: Evita loops infinitos
                top_p=0.95
            )

        # Decodifica
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"text": text}

    except Exception as e:
        logger.error(f"🔥 Erro Crítico no Server: {e}", exc_info=True)
        return {"text": f"Error: {str(e)}"}

    finally:
        # Garante que o modelo fica limpo para a próxima requisição
        clear_hooks()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1111)