import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO V6 (Hard-Grounding CPU Edition)
# -----------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B"
DEVICE = "cpu"
DTYPE = torch.float32

# Ajustes baseados no seu Log anterior:
MAX_RECURSION_DEPTH = 10  # Damos mais tempo para ele pensar
TENSION_THRESHOLD = 0.15  # Aumentado de 0.005 para 0.15 (Baseado no seu log onde a média foi ~0.15)
DECAY_RATE = 0.9  # Fator de calma: força a tensão a diminuir a cada passo

print(f"Carregando {MODEL_ID} (V6 Hard-Grounding)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map=None,
    torch_dtype=DTYPE,
    trust_remote_code=True
).to(DEVICE)
model.eval()


def get_hard_token_input(hidden_state):
    """
    HARD TOKEN: Aterramento Sólido.
    Pega apenas a palavra MAIS provável (Top-1) e usa seu embedding real.
    Isso elimina o ruído "borrado" que estava confundindo o modelo 0.5B.
    """
    with torch.no_grad():
        logits = model.lm_head(hidden_state)
        # Pega o ID da palavra vencedora
        top_token_id = torch.argmax(logits, dim=-1)
        # Pega o embedding real dessa palavra no dicionário do modelo
        token_embedding = model.get_input_embeddings()(top_token_id)

    return token_embedding, top_token_id


def calculate_cosine_tension(state_new, state_old):
    # Mede a mudança de "opinião" (direção do vetor)
    v1 = state_new.view(-1)
    v2 = state_old.view(-1)
    cosine_sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
    tension = 1.0 - cosine_sim.item()
    return tension


def generate_songlike(prompt, max_new_tokens=20):
    print(f"\nPrompt: {prompt}")
    print("-" * 60)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    # Cache do contexto inicial
    with torch.no_grad():
        base_embeddings = model.get_input_embeddings()(input_ids)

    for i in range(max_new_tokens):

        # Passo 1: O Pensamento Inicial (A_0)
        with torch.no_grad():
            outputs = model(inputs_embeds=base_embeddings, output_hidden_states=True)
            A_n = outputs.hidden_states[-1][:, -1:, :]

        xi = 0.0
        recursions = 0
        final_token_id = None

        # --- LOOP RECURSIVO (Conflito Interno) ---
        for n in range(MAX_RECURSION_DEPTH):
            recursions = n + 1

            # A. HARD GROUNDING: "Se eu dissesse X, o que eu sentiria depois?"
            # Em vez de injetar um vetor abstrato, injetamos a palavra provável
            virtual_input, pred_id = get_hard_token_input(A_n)

            # B. Simulação Mental: [Contexto] + [Palavra Provável]
            recursive_input = torch.cat([base_embeddings, virtual_input], dim=1)

            with torch.no_grad():
                outputs = model(inputs_embeds=recursive_input, output_hidden_states=True)
                A_next = outputs.hidden_states[-1][:, -1:, :]

            # C. Medir Tensão (Mudou o estado mental?)
            raw_xi = calculate_cosine_tension(A_next, A_n)

            # Aplica um decaimento artificial para garantir convergência eventual
            # (Ajuda modelos pequenos a não ficarem presos em loop infinito)
            xi = raw_xi * (DECAY_RATE ** n)

            A_n = A_next
            final_token_id = pred_id

            # Convergência
            if xi < TENSION_THRESHOLD:
                break

        # --- SAÍDA ---
        word = tokenizer.decode(final_token_id[0])
        clean_word = word.replace('\n', '\\n')

        # Marcador visual de intensidade
        bar = "|" * int(xi * 100)
        print(f"Token {i + 1}: '{clean_word}' \t| Rec: {recursions} | Tensão: {xi:.4f} {bar}")

        # Atualiza o contexto real para a próxima palavra
        with torch.no_grad():
            new_token_embed = model.get_input_embeddings()(final_token_id)
            base_embeddings = torch.cat([base_embeddings, new_token_embed], dim=1)

    print("-" * 60)
    print("Geração Concluída.")


# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
prompt_text = "The silence of the universe implies"
generate_songlike(prompt_text, max_new_tokens=15)