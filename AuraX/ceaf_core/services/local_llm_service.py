# ceaf_core/services/local_llm_service.py

import torch
import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("LocalAuraEngine")

# Variável global para segurar o modelo na memória
_GLOBAL_MODEL_INSTANCE = None
_GLOBAL_TOKENIZER_INSTANCE = None


class LocalAuraEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        # Padrão Singleton: Garante que só existe UM objeto desse na memória
        if cls._instance is None:
            cls._instance = super(LocalAuraEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path="Qwen/Qwen2.5-1.5B-Instruct"):
        # Se já foi inicializado, não faz nada (economiza RAM e evita crash)
        if self._initialized:
            return

        self.active_hooks = []
        self.layer_idx = 14
        self.device = "cpu"
        self.target_dtype = torch.float32

        # Limita threads para evitar travamento da CPU do i7 7700
        torch.set_num_threads(4)

        logger.warning(f"⚠️ [SINGLETON] Inicializando Motor Local (Qwen 1.5B) na CPU...")

        try:
            # Usa as variáveis globais se existirem
            global _GLOBAL_MODEL_INSTANCE, _GLOBAL_TOKENIZER_INSTANCE

            if _GLOBAL_MODEL_INSTANCE is None:
                logger.info(f"💾 Carregando binários do modelo para a RAM...")
                _GLOBAL_TOKENIZER_INSTANCE = AutoTokenizer.from_pretrained(model_path)
                _GLOBAL_MODEL_INSTANCE = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=self.target_dtype,
                    low_cpu_mem_usage=True
                ).to(self.device)
                logger.info("✅ Modelo carregado na memória pela PRIMEIRA VEZ.")
            else:
                logger.info("♻️ Reutilizando modelo já carregado na memória.")

            # Vincula a instância local às globais
            self.tokenizer = _GLOBAL_TOKENIZER_INSTANCE
            self.model = _GLOBAL_MODEL_INSTANCE

            # Marca como inicializado para não rodar isso de novo
            self._initialized = True

        except Exception as e:
            logger.error(f"❌ Erro fatal: {e}")
            raise e

    def apply_hormone(self, vector_npy, intensity):
        """Injeta o vetor de honestidade no stream residual."""
        self.remove_hooks()

        steering_vector = torch.from_numpy(vector_npy).to(self.device).to(self.target_dtype)

        def hook_fn(module, input, output):
            if output[0].shape[1] > 0:
                output[0][:, -1, :] += intensity * steering_vector
            return output

        # Proteção robusta de camadas
        num_layers = len(self.model.model.layers)
        target_idx = min(self.layer_idx, num_layers - 1)
        target_layer = self.model.model.layers[target_idx]

        handle = target_layer.register_forward_hook(hook_fn)
        self.active_hooks.append(handle)

    def remove_hooks(self):
        for h in self.active_hooks: h.remove()
        self.active_hooks = []

    async def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.6,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)