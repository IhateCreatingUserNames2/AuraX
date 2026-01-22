import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer, util


class GeometricBrain:
    def __init__(self):
        # Carrega o modelo de embeddings (rápido)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.capacity = 7
        self.epsilon = 1.1

    def compute_gating(self, input_text, current_wm_vectors):
        # Garante que new_vec é float32
        new_vec = self.encoder.encode([input_text])[0].astype(np.float32)

        if not current_wm_vectors:
            return "ACCEPT", 0.0, new_vec, None

        # Lógica Híbrida do AuraFlux
        # CORREÇÃO: Forçar dtype=np.float32 ao criar a matriz
        wm_matrix = np.array(current_wm_vectors, dtype=np.float32)

        # Cold Start (Cosine)
        if len(current_wm_vectors) < 4:
            cos_scores = util.cos_sim(new_vec, wm_matrix)[0].cpu().numpy()
            max_sim = float(np.max(cos_scores))
            dist = (1.0 - max_sim) * 2.0
            action = "REINFORCE" if max_sim > 0.85 else "ACCEPT"
            return action, dist, new_vec, None # Cold start não tem target_idx, retorna None

        # Manifold (PCA)
        all_vecs = np.vstack([wm_matrix, new_vec])
        pca = PCA(n_components=min(len(all_vecs), 7))
        proj = normalize(pca.fit_transform(all_vecs), axis=1)

        dists = np.linalg.norm(proj[:-1] - proj[-1], axis=1)
        min_dist = np.min(dists)
        idx = np.argmin(dists)

        action = "REINFORCE" if min_dist < self.epsilon else "ACCEPT"
        return action, min_dist, new_vec, idx