# ceaf_core/modules/geometric_brain.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer, util


class GeometricBrain:
    def __init__(self):
        # Carrega o modelo de embeddings (rápido)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.capacity = 7
        # Epsilon ajustado: Como vamos normalizar a distância,
        # o threshold antigo de 1.1 (em escala 0-2) vira 0.55 (em escala 0-1)
        self.epsilon = 0.55

    def compute_gating(self, input_text, current_wm_vectors):
        # Garante que new_vec é float32
        new_vec = self.encoder.encode([input_text])[0].astype(np.float32)

        if not current_wm_vectors:
            return "ACCEPT", 0.0, new_vec, None

        # Lógica Híbrida do AuraFlux
        wm_matrix = np.array(current_wm_vectors, dtype=np.float32)

        # Cold Start (Cosine)
        if len(current_wm_vectors) < 4:
            cos_scores = util.cos_sim(new_vec, wm_matrix)[0].cpu().numpy()
            max_sim = float(np.max(cos_scores))

            # CORREÇÃO 1: Remover multiplicação por 2.0 e garantir range [0, 1]
            # Similaridade 1.0 -> Distância 0.0
            # Similaridade 0.0 -> Distância 1.0
            dist = 1.0 - max(0.0, max_sim)

            action = "REINFORCE" if max_sim > 0.85 else "ACCEPT"
            return action, dist, new_vec, None

            # Manifold (PCA)
        # Empilha vetores antigos com o novo
        all_vecs = np.vstack([wm_matrix, new_vec])

        # PCA reduz a dimensionalidade
        pca = PCA(n_components=min(len(all_vecs), 7))
        # Normaliza para projetar na esfera unitária
        proj = normalize(pca.fit_transform(all_vecs), axis=1)

        # Calcula distância euclidiana do novo ponto (último) para os anteriores
        # A distância Euclidiana na esfera unitária vai de 0 a 2.0
        raw_dists = np.linalg.norm(proj[:-1] - proj[-1], axis=1)
        min_raw_dist = np.min(raw_dists)
        idx = np.argmin(raw_dists)

        # CORREÇÃO 2: Normalizar a distância Euclidiana (0..2) para Xi (0..1)
        # Dividimos por 2.0 para manter a consistência
        xi = min_raw_dist / 2.0
        xi = float(np.clip(xi, 0.0, 1.0))

        # Ação baseada na proximidade (se está muito perto, reforça. Se longe, aceita novo)
        # Usamos o epsilon ajustado (0.55) comparado com o Xi normalizado
        action = "REINFORCE" if xi < self.epsilon else "ACCEPT"

        return action, xi, new_vec, idx