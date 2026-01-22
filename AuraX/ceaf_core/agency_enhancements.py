# NOVO ARQUIVO: ceaf_v3/agency_enhancements.py
# Contém os avaliadores de primitivas otimizados e não-LLM para o AgencyModule.

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List

# Inicializa o analisador de sentimento uma vez para reutilização
sentiment_analyzer = SentimentIntensityAnalyzer()


async def eval_narrative_continuity(candidate_embedding: np.ndarray, persona_embedding: np.ndarray) -> float:
    """
    Avalia a continuidade narrativa calculando a similaridade de cosseno.
    Retorna um score de 0.0 (totalmente diferente) a 1.0 (idêntico).
    """
    if persona_embedding is None or candidate_embedding is None:
        return 0.5  # Retorna um valor neutro se os embeddings não estiverem disponíveis

    # Garante que os embeddings sejam arrays 2D para a função
    similarity = cosine_similarity(candidate_embedding.reshape(1, -1), persona_embedding.reshape(1, -1))

    # O score de similaridade já está entre -1 e 1, normalizamos para 0-1
    return (similarity[0][0] + 1) / 2


async def eval_specificity(text: str) -> float:
    """
    Avalia a especificidade usando uma métrica estatística simples: o comprimento médio das palavras.
    Textos com palavras mais longas são considerados mais específicos/complexos.
    Retorna um score normalizado entre 0.0 e 1.0.
    """
    words = text.split()
    if not words:
        return 0.0

    avg_word_length = sum(len(word) for word in words) / len(words)

    # Normaliza o score. Assumimos que um comprimento médio de 8+ é muito específico.
    # Esta é uma heurística simples que pode ser aprimorada.
    score = min(avg_word_length / 8.0, 1.0)
    return score


async def eval_emotional_resonance(text: str) -> float:
    """
    Avalia a ressonância emocional usando VADER para análise de sentimento.
    Retorna a magnitude do sentimento (positivo ou negativo) como um score de 0.0 a 1.0.
    """
    sentiment_scores = sentiment_analyzer.polarity_scores(text)
    # Usamos o 'compound score', que varia de -1 (negativo) a 1 (positivo).
    # O valor absoluto nos dá a magnitude da emoção, independentemente da polaridade.
    resonance_score = abs(sentiment_scores['compound'])
    return resonance_score