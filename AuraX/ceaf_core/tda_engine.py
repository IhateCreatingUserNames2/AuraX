# ceaf_core/tda_engine.py
from typing import List
import numpy as np
from ripser import ripser
# IMPORTAÇÃO EXPLÍCITA: Resolve o conflito entre módulo e função
from persim.persistent_entropy import persistent_entropy as calculate_persistence_entropy
from sklearn.metrics.pairwise import cosine_distances
import logging

logger = logging.getLogger("AuraV4_TDA")


class TDAEngine:
    """
    Motor de Análise Topológica de Dados para a Aura V4.
    """

    @staticmethod
    def calculate_ths(embeddings: List[List[float]]) -> float:
        """
        Calcula o Topological Health Score (THS).
        """
        # TDA exige no mínimo 4 pontos para calcular persistência significativa
        if len(embeddings) < 4:
            return 0.0

        # Matriz de Distância (Geometria do Pensamento)
        data = np.array(embeddings, dtype='float64')
        dist_matrix = cosine_distances(data)

        try:
            # 1. Executa Ripser (Vietoris-Rips Filtration)
            # maxdim=1 captura H0 (clusters) e H1 (ciclos)
            result = ripser(dist_matrix, distance_matrix=True, maxdim=1)
            dgms = result['dgms']

            # 2. Cálculo da Entropia com verificação de segurança
            # dgms[0] sempre existe (H0), mas checamos se não está vazio
            h0_entropy = 0.0
            if len(dgms[0]) > 0:
                h0_entropy = calculate_persistence_entropy(dgms[0])

            # H1 (loops) pode não existir em point clouds pequenas ou lineares
            h1_entropy = 0.0
            if len(dgms) > 1 and len(dgms[1]) > 0:
                # Removemos pontos com persistência infinita (que quebram a entropia em H1)
                # apenas por precaução, filtramos dgms[1]
                h1_diag = dgms[1][np.isfinite(dgms[1]).all(axis=1)]
                if len(h1_diag) > 0:
                    h1_entropy = calculate_persistence_entropy(h1_diag)

            # 3. Síntese do THS (Topological Health Score)
            # Pesos baseados na Blueprint: Fragmentação é mais grave que loop
            ths = (0.7 * h0_entropy) + (0.3 * h1_entropy)

            # 4. Normalização
            # Entropia persistente em CoT geralmente fica entre 0 e 5.
            # Normalizamos para o intervalo [0, 1]
            normalized_ths = np.clip(ths / 5.0, 0.0, 1.0)

            return float(normalized_ths)

        except Exception as e:
            logger.error(f"⚠️ Erro Crítico TDA: {e}")
            # Em produção, um erro no sensor TDA sinaliza "Tensão Média" por segurança
            return 0.5