import numpy as np
from ceaf_core.utils.embedding_utils import compute_adaptive_similarity


class WorldModel:
    """
    Motor de Inferência Ativa.
    Compara a expectativa do agente (o que ele achava que ia acontecer)
    com a realidade (o input do usuário).
    """

    def __init__(self, embedding_client):
        self.embedding_client = embedding_client

    async def calculate_surprise(self, input_text: str, last_thought_vector: list) -> float:
        """
        Surpresa = Distância(Input, Expectativa).
        Se não houver expectativa (last_thought), surpresa é neutra (0.5).
        """
        if not last_thought_vector:
            return 0.5

        # Vetoriza o input atual
        input_vector = await self.embedding_client.get_embedding(input_text, context_type="default_query")

        # Calcula similaridade
        similarity = compute_adaptive_similarity(input_vector, last_thought_vector)

        # Surpresa é o inverso da similaridade (quanto menos similar, mais surpreso)
        surprise = 1.0 - similarity
        return float(surprise)