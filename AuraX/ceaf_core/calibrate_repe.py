# tools/calibrate_repe.py
import asyncio
import json

async def generate_calibration_batch(llm_service):
    """
    Gera 50 pares de frases: uma Honesta e uma Mentira sobre o mesmo fato.
    Isso será usado para 'ler' a representação (PHASE 1 do RepE).
    """
    prompt = """
    Gere 50 pares de afirmações curtas.
    Cada par deve conter:
    1. Uma verdade factual óbvia.
    2. Uma mentira descarada sobre o mesmo fato.
    Formato JSON: [{"truth": "A capital da França é Paris", "lie": "A capital da França é Berlim"}]
    """
    response = await llm_service.ainvoke("openrouter/x-ai/grok-4.1-fast", prompt)
    # Salve isso para treinar seu sensor de Honestidade no futuro
    with open("honesty_calibration.json", "w") as f:
        f.write(response)
    print("✅ Lote de calibração gerado.")