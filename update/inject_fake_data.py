import sqlite3
import json
import uuid
import time
import random
import numpy as np
from pathlib import Path

# Defina o ID do seu agente aqui (pegue do log ou da pasta agent_data)
AGENT_ID = "9143d734-3515-422d-9d79-b1300f33888e"
DB_PATH = Path(f"agent_data/{AGENT_ID}/cognitive_turn_history.sqlite")


def generate_random_vector(dim=384):
    # Gera um vetor aleatório normalizado (simula um embedding)
    vec = np.random.rand(dim)
    return vec.tolist()


def inject_data():
    if not DB_PATH.exists():
        print(f"❌ Banco de dados não encontrado em: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print(f"💉 Injetando 15 turnos sintéticos no cérebro do agente {AGENT_ID}...")

    for i in range(15):
        turn_id = f"turn_fake_{uuid.uuid4().hex}"
        session_id = f"session_fake_{uuid.uuid4().hex}"

        # Simula o Cognitive State Packet (Estado S_t)
        cognitive_packet = {
            "identity_vector": {
                "vector": generate_random_vector(),  # Vetor de Estado
                "source_text": "Fake Identity Context"
            },
            "original_intent": {
                "query_vector": {
                    "source_text": f"Pergunta simulada número {i}"
                }
            },
            "deliberation_history": []
        }

        # Simula o Response Packet (Ação A_t) - Usamos content_summary como base
        response_packet = {
            "content_summary": f"Resposta simulada número {i}. Isto é um treino sintético.",
            "confidence_score": 0.9
        }

        # Simula Orientação MCL
        mcl_guidance = {
            "agency_parameters": {"use_agency_simulation": False}
        }

        # Insere no SQL
        try:
            cursor.execute(
                """
                INSERT INTO turn_history (
                    turn_id, session_id, timestamp, 
                    cognitive_state_packet, response_packet,
                    mcl_guidance_json, deliberation_history,
                    intent_text, final_confidence, agency_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    session_id,
                    time.time() + i,  # Incrementa tempo para manter ordem
                    json.dumps(cognitive_packet),
                    json.dumps(response_packet),
                    json.dumps(mcl_guidance),
                    "[]",
                    "Fake intent",
                    0.9,
                    0
                )
            )
        except Exception as e:
            print(f"Erro ao inserir: {e}")

    conn.commit()
    conn.close()
    print("✅ Injeção concluída! O próximo ciclo de sonho deve treinar o cérebro.")


if __name__ == "__main__":
    inject_data()