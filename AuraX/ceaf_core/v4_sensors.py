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
        # Reduzimos o peso da deriva para não punir agentes "novos"
        identity_drift = 1.0 - compute_adaptive_similarity(current_vector, glyph_vector)

        # 2. Fragmentação (TDA)
        if not context_vectors:
            # Se não há contexto, a fragmentação é baixa
            fragmentation = 0.0
        else:
            points = context_vectors + [current_vector]
            fragmentation = self.tda_engine.calculate_ths(points)

        # CÁLCULO PONDERADO V4.1 (Mais permissivo no início)
        # Diminuímos a influência da identidade se o agente não tiver memórias coesas
        xi = (0.4 * identity_drift) + (0.6 * fragmentation)

        # LOG DE DEBUG PARA CALIBRAÇÃO
        # print(f"DEBUG TDA: Drift={identity_drift:.2f}, Frag={fragmentation:.2f}, XI={xi:.2f}")

        return float(np.clip(xi, 0.0, 1.0))

    # REMOVIDO @staticmethod pois usamos self.tda_engine
    def get_epistemic_tension(self, current_vector: List[float], glyph_vector: List[float],
                              history_vectors: List[List[float]]) -> float:
        # A. Deriva de Identidade
        drift = 1.0 - compute_adaptive_similarity(current_vector, glyph_vector)

        # B. Saúde Topológica
        # CORRIGIDO: de TDA_engine para self.tda_engine
        point_cloud = history_vectors + [current_vector]
        ths = self.tda_engine.calculate_ths(point_cloud)

        # C. xi Final
        xi = (0.5 * drift) + (0.5 * ths)
        return float(xi)

    @staticmethod
    def detect_epistemic_fault(response_text: str, evidence_text: str) -> float:
        ratio = len(response_text) / (len(evidence_text) + 1)
        fault_tension = np.clip(ratio / 10.0, 0.0, 1.0)
        return float(fault_tension)