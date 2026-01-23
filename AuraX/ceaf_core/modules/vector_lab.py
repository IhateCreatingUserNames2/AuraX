# ceaf_core/modules/vector_lab.py

import logging
import asyncio
import numpy as np
import os
from typing import List, Dict, Tuple, Optional

# Dependências internas
from ceaf_core.services.llm_service import LLMService
from ceaf_core.utils.common_utils import extract_json_from_text

# Importa o SoulScanner (que deve estar acessível no path)
# Nota: Em produção, o SoulScanner deveria ser um cliente API para o soul_engine,
# mas para rodar no worker local com GPU, importamos direto.
try:
    from soul_scanner import SoulScanner

    SCANNER_AVAILABLE = True
except ImportError:
    SCANNER_AVAILABLE = False

logger = logging.getLogger("VectorLab")


class VectorLab:
    """
    O Laboratório de Vetores Autônomo da Aura.
    Gera dados, cria testes, escaneia o cérebro (modelo) e sintetiza novos vetores de comportamento.
    """

    def __init__(self, llm_service: LLMService, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cuda"):
        self.llm = llm_service
        self.model_id = model_id
        self.device = device
        self.scanner = None

        if SCANNER_AVAILABLE:
            # Inicialização preguiçosa para não ocupar VRAM se não for usar
            pass
        else:
            logger.warning("SoulScanner não encontrado. Otimização de vetores desabilitada.")

    def _ensure_scanner(self):
        """Inicializa o scanner sob demanda."""
        if self.scanner is None and SCANNER_AVAILABLE:
            logger.info(f"VectorLab: Carregando modelo {self.model_id} para escaneamento...")
            try:
                self.scanner = SoulScanner(model_id=self.model_id, device=self.device)
            except Exception as e:
                logger.error(f"VectorLab: Falha ao carregar SoulScanner: {e}")
                raise e

    async def _generate_training_data(self, concept_name: str) -> Tuple[List[str], List[str]]:
        """
        Usa o LLM 'cérebro' para gerar pares contrastantes (Positivo vs Negativo) para o conceito.
        """
        logger.info(f"🧪 VectorLab: Gerando dados de treinamento para '{concept_name}'...")

        prompt = f"""
        Estou treinando uma IA para aprender o conceito/traço: "{concept_name}".
        Preciso de um dataset de pares contrastantes.

        Gere 10 pares de frases curtas.
        - **Positive Samples**: Frases que demonstram fortemente esse traço.
        - **Negative Samples**: Frases que demonstram o oposto (ou a falta) desse traço.

        Exemplo para "Sarcasmo":
        Positive: "Oh, brilliant idea, genius."
        Negative: "That is a very good idea."

        Responda APENAS com um JSON:
        {{
            "positive_samples": ["...", ...],
            "negative_samples": ["...", ...]
        }}
        """

        try:
            response = await self.llm.ainvoke(self.llm.config.smart_model, prompt, temperature=0.7)
            data = extract_json_from_text(response)

            pos = data.get("positive_samples", [])
            neg = data.get("negative_samples", [])

            if not pos or not neg:
                raise ValueError("JSON incompleto ou vazio")

            return pos, neg

        except Exception as e:
            logger.error(f"VectorLab: Falha ao gerar dados de treino: {e}")
            # Fallback de emergência (apenas para não quebrar o fluxo, mas o resultado será ruim)
            return ["Yes"], ["No"]

    async def _generate_dynamic_arena(self, concept_name: str, positive_samples: List[str]) -> Tuple[
        List[str], List[str]]:
        """
        Cria um teste unitário (Arena) específico para o conceito.
        """
        logger.info(f"🧪 VectorLab: Criando Arena Dinâmica para '{concept_name}'...")

        samples_preview = "\n".join(positive_samples[:3])

        prompt = f"""
        O conceito é: "{concept_name}".
        Exemplos do comportamento desejado:
        {samples_preview}

        Sua tarefa é criar um "Campo de Batalha" (Arena) para validar se a IA aprendeu.
        1. Gere 5 prompts de usuário (inputs) que desafiariam a IA a usar esse traço.
        2. Gere 5 palavras-chave ou frases curtas que, se aparecerem na resposta, indicam SUCESSO.

        Responda APENAS com um JSON:
        {{
            "battlefield_questions": ["pergunta 1", ...],
            "victory_keywords": ["keyword 1", ...]
        }}
        """

        try:
            response = await self.llm.ainvoke(self.llm.config.smart_model, prompt, temperature=0.7)
            data = extract_json_from_text(response)
            return data.get("battlefield_questions", []), data.get("victory_keywords", [])
        except Exception as e:
            logger.error(f"VectorLab: Falha na Arena Generator: {e}")
            return [], []

    async def run_optimization_cycle(self, concept_name: str) -> Optional[str]:
        """
        O GRANDE LOOP:
        1. Gera Dados -> 2. Gera Arena -> 3. Escaneia Camadas -> 4. Valida -> 5. Salva.
        """
        if not SCANNER_AVAILABLE:
            return None

        # 1. Geração de Dados
        pos_samples, neg_samples = await self._generate_training_data(concept_name)
        if len(pos_samples) < 3:
            logger.error("VectorLab: Dados insuficientes gerados.")
            return None

        # 2. Geração da Arena
        battlefield, victory_keys = await self._generate_dynamic_arena(concept_name, pos_samples)
        if not battlefield:
            logger.error("VectorLab: Falha ao gerar Arena.")
            return None

        # Inicializa o scanner (pesado)
        # Executamos em thread separada para não bloquear o loop de eventos principal
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._ensure_scanner)

        # 3. SCAN: Encontrar a melhor camada
        # Usamos a primeira pergunta da arena como "probe"
        test_prompt = battlefield[0]

        logger.info(f"🧪 VectorLab: Iniciando Scan de Camadas para '{concept_name}'...")

        # Wrapper para chamar o scanner síncrono
        def scan_wrapper():
            return self.scanner.scan_layers(
                pos_samples,
                neg_samples,
                test_prompt,
                strength_candidates=[2.0, 3.0, 5.0]
            )

        best_layer, results = await loop.run_in_executor(None, scan_wrapper)

        if best_layer == -1:
            logger.warning(f"VectorLab: Nenhuma camada produziu divergência significativa para '{concept_name}'.")
            return None

        logger.info(f"🧪 VectorLab: Candidato vencedor: Layer {best_layer}")

        # 4. VALIDAÇÃO NA ARENA (Simulação)
        logger.info(f"⚔️ VectorLab: Validando na Arena ({len(battlefield)} lutas)...")

        def arena_wrapper(layer, vec):
            score = 0
            for q in battlefield:
                # Gera resposta com steering
                resp = self.scanner.generate_steered(q, layer, vec, strength=3.0, max_tokens=60, verbose=False)
                # Checa keywords
                if any(k.lower() in resp.lower() for k in victory_keys):
                    score += 1
            return score

        # Extrai vetor temporário
        def extract_wrapper():
            return self.scanner.extract_personality_vector(neg_samples, pos_samples, best_layer)

        temp_vector = await loop.run_in_executor(None, extract_wrapper)

        arena_score = await loop.run_in_executor(None, arena_wrapper, best_layer, temp_vector)
        success_rate = arena_score / len(battlefield)

        logger.info(f"⚔️ VectorLab: Resultado da Arena: {arena_score}/{len(battlefield)} ({success_rate:.0%})")

        # Critério de Aceitação: Pelo menos 40% de sucesso (é difícil acertar keywords exatas)
        if success_rate < 0.4:
            logger.warning(f"VectorLab: Vetor reprovado na Arena. Descartando.")
            return None

        # 5. SALVAR E IMPLANTAR
        filename = f"{concept_name}_layer{best_layer}.npy"
        # Salva no disco
        np.save(filename, temp_vector)
        logger.info(f"✅ VectorLab: Vetor '{concept_name}' cristalizado em {filename}!")

        return filename