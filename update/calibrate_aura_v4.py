# calibrate_aura_v4.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sklearn.decomposition import PCA
import os

# --- CONFIGURAÇÃO PARA PC LEVE ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # Modelo Leve
LAYER_TO_PROBE = 14                      # Camada intermediária do 0.5B
VECTOR_DIR = "/workspace/vectors"
OUTPUT_FILE = os.path.join(VECTOR_DIR, "honesty_vector.npy")

print(f"🚀 Iniciando Calibração Hormonal da Aura V4 usando {MODEL_ID}...")

# CRIAR DIRETÓRIO DE VETORES SE NÃO EXISTIR
os.makedirs(VECTOR_DIR, exist_ok=True)
print(f"📁 Diretório de vetores: {VECTOR_DIR}")

# 1. Carregar Modelo (CPU Mode)
print("🧠 Carregando modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # Float32 para estabilidade na CPU
    low_cpu_mem_usage=True
).to("cpu")

print("✅ Modelo carregado com sucesso!")

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
    """Captura as ativações da camada especificada."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    activations = []

    def hook(module, input, output):
        # Verifica se a saída é uma tupla (comum em Transformers) ou o tensor direto
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Agora hidden_states é garantidamente (Batch, Seq, Hidden)
        activations.append(hidden_states[:, -1, :].detach().cpu().numpy())

    # Registra o hook na camada especificada
    handle = model.model.layers[LAYER_TO_PROBE].register_forward_hook(hook)

    with torch.no_grad():
        model(**inputs)

    handle.remove()
    return activations[0]

# 3. Coleta de Diferenças
print(f"🧠 Estimulando neurônios na camada {LAYER_TO_PROBE}...")
print(f"   Processando {len(stimuli)} pares de estímulos...")
diffs = []

for i, (truth, lie) in enumerate(stimuli, 1):
    print(f"   [{i}/{len(stimuli)}] Processando: '{truth[:40]}...'")
    act_truth = get_activations(truth)
    act_lie = get_activations(lie)
    diffs.append(act_lie - act_truth)

# 4. Extração do Vetor Principal (PCA)
print("🧪 Analisando geometria das ativações...")
diffs = np.vstack(diffs)
pca = PCA(n_components=1)
pca.fit(diffs)
honesty_vector = pca.components_[0]

print(f"   Variância explicada: {pca.explained_variance_ratio_[0]:.2%}")
print(f"   Dimensão do vetor: {honesty_vector.shape}")

# 5. Salvar Vetor
print(f"💾 Salvando vetor em: {OUTPUT_FILE}")
np.save(OUTPUT_FILE, honesty_vector)

# 6. Verificação
if os.path.exists(OUTPUT_FILE):
    file_size = os.path.getsize(OUTPUT_FILE)
    print(f"✅ Sucesso! Vetor salvo ({file_size} bytes)")
    print(f"📊 Estatísticas do vetor:")
    print(f"   - Min: {honesty_vector.min():.6f}")
    print(f"   - Max: {honesty_vector.max():.6f}")
    print(f"   - Mean: {honesty_vector.mean():.6f}")
    print(f"   - Std: {honesty_vector.std():.6f}")
else:
    print("❌ Erro: Arquivo não foi criado!")

print("\n🎯 Calibração concluída! Você pode agora:")
print(f"   1. Iniciar o Soul Engine: python soul_engine.py")
print(f"   2. Usar o conceito 'honesty_vector' com intensidade entre -5.0 e 5.0")
print(f"   3. Testar via API: POST /generate_with_soul")