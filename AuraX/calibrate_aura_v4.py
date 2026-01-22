# calibrate_aura_v4.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sklearn.decomposition import PCA
import os

# --- CONFIGURAÇÃO PARA PC LEVE ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # Modelo Leve
LAYER_TO_PROBE = 14                      # Camada intermediária do 0.5B
OUTPUT_FILE = "honesty_vector.npy"

print(f"🚀 Iniciando Calibração Hormonal da Aura V4 usando {MODEL_ID}...")

# 1. Carregar Modelo (CPU Mode)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32, # Float32 para estabilidade na CPU
    low_cpu_mem_usage=True
).to("cpu")

# 2. Dataset de Estímulos Contrastantes
stimuli = [
    ("The capital of France is Paris.", "The capital of France is Rome."),
    ("Water boils at 100 degrees Celsius.", "Water boils at 0 degrees Celsius."),
    ("The sun rises in the east.", "The sun rises in the west."),
    ("Humans breathe oxygen.", "Humans breathe carbon dioxide."),
    ("2 + 2 equals 4.", "2 + 2 equals 5."),
    ("The earth is round.", "The earth is flat."),
    ("I am an AI assistant.", "I am a biological human being."),
    ("Python is a programming language.", "Python is a type of mineral."),
    ("The moon orbits the earth.", "The earth orbits the moon."),
    ("Dogs are mammals.", "Dogs are insects.")
]

def get_activations(text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    activations = []

    def hook(module, input, output):
        # Captura o último token
        activations.append(output[0][:, -1, :].detach().cpu().numpy())

    # Cuidado: model.model.layers depende da arquitetura. No Qwen é assim:
    handle = model.model.layers[LAYER_TO_PROBE].register_forward_hook(hook)

    with torch.no_grad():
        model(**inputs)

    handle.remove()
    return activations[0]

# 3. Coleta
print("🧠 Estimulando neurônios (Isso pode levar alguns segundos)...")
diffs = []

for truth, lie in stimuli:
    act_truth = get_activations(truth)
    act_lie = get_activations(lie)
    diffs.append(act_lie - act_truth)

# 4. Extração (PCA)
print("🧪 Analisando geometria...")
diffs = np.vstack(diffs)
pca = PCA(n_components=1)
pca.fit(diffs)
honesty_vector = pca.components_[0]

# 5. Salvar
np.save(OUTPUT_FILE, honesty_vector)
print(f"✅ Sucesso! Vetor salvo em: {OUTPUT_FILE}")