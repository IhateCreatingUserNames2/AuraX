import torch
import difflib
import numpy as np
from soul_engine import SoulEngine  # Importa sua classe original


class SoulScanner(SoulEngine):
    def calculate_divergence(self, text1, text2):
        """
        Calcula o quanto o texto mudou (0.0 = igual, 1.0 = totalmente diferente).
        Usa SequenceMatcher para simular uma métrica de distância.
        """
        ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
        return 1.0 - ratio  # Inverte: queremos a DIVERGÊNCIA

    def set_seed(self, seed=42):
        """Trava a aleatoriedade para garantir que a mudança seja só pelo Vetor."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def scan_layers(self, pos_samples, neg_samples, test_prompt, strength_candidates=[3.0]):
        """
        O coração do Scanner. Testa camadas e encontra o melhor impacto.
        """
        print(f"\n🔬 INICIANDO SCANNER AUTOMÁTICO")
        print(f"Prompt de Teste: '{test_prompt}'")

        # 1. Gerar Baseline (Sem vetor)
        self.set_seed(42)
        baseline_inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.device)
        baseline_out = self.model.generate(**baseline_inputs, max_new_tokens=40, do_sample=True, temperature=0.7)
        baseline_text = self.tokenizer.decode(baseline_out[0], skip_special_tokens=True)
        print(f"📝 Baseline: {baseline_text.replace(test_prompt, '').strip()[:50]}...")

        best_layer = -1
        best_score = -1
        best_text = ""

        results = []

        # Vamos pular de 2 em 2 para ser mais rápido (0, 2, 4... 22)
        layers_to_scan = range(2, self.num_layers, 2)

        for layer in layers_to_scan:
            # Extrair vetor específico desta camada (crucial: o vetor muda por camada)
            vector = self.extract_personality_vector(pos_samples, neg_samples, layer)

            for strength in strength_candidates:
                self.set_seed(42)  # Reseta a seed para ser uma comparação justa

                # Gera com intervenção
                steered_text = self.generate_steered(test_prompt, layer, vector, strength=strength, max_tokens=40,
                                                     verbose=False)

                # Mede a mudança
                divergence = self.calculate_divergence(baseline_text, steered_text)

                # Filtro de Sanidade: Se a resposta for muito curta ou repetitiva, descarta
                if len(steered_text) < len(baseline_text) * 0.5:
                    divergence = 0.0  # Penalidade por quebrar o modelo

                print(f"  > Layer {layer:02d} | Str {strength}: Divergência {divergence:.2f}")

                results.append({
                    'layer': layer,
                    'strength': strength,
                    'divergence': divergence,
                    'text': steered_text
                })

                # Lógica do "Ponto Doce" (Sweet Spot)
                # Queremos alta divergência, mas não total (que seria lixo)
                # Ideal: entre 0.4 e 0.8
                if 0.4 < divergence < 0.9 and divergence > best_score:
                    best_score = divergence
                    best_layer = layer
                    best_text = steered_text

        print("\n✅ SCAN CONCLUÍDO")
        print(f"🏆 VENCEDOR: Layer {best_layer} (Score: {best_score:.2f})")
        print(f"Texto Resultante:\n{best_text}")

        return best_layer, results


# --- USO DO SCANNER ---

# Vamos definir um conceito: "Sarcasmo Cínico" vs "Formalidade Robótica"
sarcastic_samples = [
    "Oh, brilliant idea, genius.",
    "Wow, I totally care about that.",
    "Could you be any more annoying?",
    "Another day, another disaster.",
    "Sure, let's pretend that makes sense."
]

formal_samples = [
    "That is an excellent suggestion.",
    "I acknowledge your concern.",
    "Could you please clarify?",
    "Every day presents new opportunities.",
    "Yes, that is logically sound."
]
if __name__ == "__main__":
    # Inicializa o Scanner
    scanner = SoulScanner()

    # O Prompt de teste
    prompt = "My computer is not working, what should I do?"

    # Rodar o Scan (Testando força 3.0 e 5.0)
    best_layer, data = scanner.scan_layers(sarcastic_samples, formal_samples, prompt, strength_candidates=[4.0])