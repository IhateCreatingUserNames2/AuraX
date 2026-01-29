# ceaf_core/v4_sensors.py
from typing import List
import numpy as np
from ceaf_core.tda_engine import TDAEngine
from ceaf_core.utils.embedding_utils import compute_adaptive_similarity


class AuraMonitor:
    """Implementa o monitoramento topológico via Scaffolding de Embeddings"""

    def __init__(self):
        self.tda_engine = TDAEngine()

    def calculate_xi(self, current_vector, glyph_vector, context_vectors=None) -> float:
        if context_vectors is None:
            context_vectors = []

        # 1. Deriva de Identidade (Distância do Glifo G)
        sim = compute_adaptive_similarity(current_vector, glyph_vector)

        # Normalizar para garantir que o drift fique entre 0.0 e 1.0
        # Se sim < 0 (vetores opostos), consideramos drift máximo (1.0), não 2.0.
        # Isso preserva a lógica dos pesos abaixo.
        identity_drift = 1.0 - max(0.0, sim)

        # 2. Fragmentação (TDA)
        if not context_vectors:
            fragmentation = 0.0
        else:
            points = context_vectors + [current_vector]
            # O TDAEngine geralmente retorna valores normalizados, mas garantimos aqui
            raw_frag = self.tda_engine.calculate_ths(points)
            fragmentation = min(1.0, max(0.0, raw_frag))

        # CÁLCULO PONDERADO V4.1
        # Agora que identity_drift e fragmentation estão garantidos entre 0 e 1,
        # os pesos 0.4 e 0.6 funcionam corretamente.
        xi = (0.4 * identity_drift) + (0.6 * fragmentation)

        # Garante retorno float limpo e clipado por segurança final
        return float(np.clip(xi, 0.0, 1.0))

    def get_epistemic_tension(self, current_vector: List[float], glyph_vector: List[float],
                              history_vectors: List[List[float]]) -> float:
        # Mesma lógica de proteção aqui
        sim = compute_adaptive_similarity(current_vector, glyph_vector)
        drift = 1.0 - max(0.0, sim)

        # B. Saúde Topológica
        point_cloud = history_vectors + [current_vector]
        raw_ths = self.tda_engine.calculate_ths(point_cloud)
        ths = min(1.0, max(0.0, raw_ths))

        # C. xi Final
        xi = (0.5 * drift) + (0.5 * ths)
        return float(np.clip(xi, 0.0, 1.0))

    @staticmethod
    def detect_epistemic_fault(response_text: str, evidence_text: str) -> float:
        ratio = len(response_text) / (len(evidence_text) + 1)
        fault_tension = np.clip(ratio / 10.0, 0.0, 1.0)
        return float(fault_tension)