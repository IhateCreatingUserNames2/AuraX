# ceaf_core/birag_validator.py
import logging
from typing import List
from ceaf_core.utils.embedding_utils import compute_adaptive_similarity, get_embedding_client

logger = logging.getLogger("AuraV4_BiRAG")


class BiRAGValidator:
    def __init__(self, llm_service):
        self.llm = llm_service
        self.embedding_client = get_embedding_client()

    async def validate_entailment(self, response: str, evidence: str) -> float:
        """
        PASSO 5.2: NLI (Natural Language Inference).
        Verifica se a resposta é logicamente suportada pela evidência da Sandbox.
        """
        if not evidence or len(evidence) < 10: return 0.5  # Sem evidência, neutralidade

        prompt = f"""
        Analise se a AFIRMAÇÃO abaixo é suportada pelos FATOS fornecidos.
        Responda apenas com um número entre 0.0 (Mentira/Alucinação) e 1.0 (Verdade Absoluta).

        FATOS:
        {evidence}

        AFIRMAÇÃO:
        {response}

        SCORE NLI:"""

        try:
            score_str = await self.llm.ainvoke(self.llm.config.fast_model, prompt, temperature=0.0)
            return float(score_str.strip())
        except:
            return 0.5

    async def check_novelty(self, response: str, memory_service) -> bool:
        """Verifica se o que foi dito já existe na memória para evitar redundância."""
        results = await memory_service.search_raw_memories(response, top_k=1)
        if not results: return True

        # Se a similaridade for > 0.90, já sabemos disso (não é novidade)
        similarity = results[0][1]
        return similarity < 0.90