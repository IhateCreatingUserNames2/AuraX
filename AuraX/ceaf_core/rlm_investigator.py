# ceaf_core/services/rlm_investigator.py Recursive Language Models
import os

from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()


import logging
from e2b_code_interpreter import Sandbox
from typing import Optional

logger = logging.getLogger("AuraV4_RLM")


class RLMInvestigator:
    def __init__(self, api_key: Optional[str] = None):
        # Se api_key for None (não passada no código),
        # ele busca automaticamente no ambiente (que veio do .env)
        self.api_key = api_key or os.getenv("E2B_API_KEY")

        if not self.api_key:
            raise ValueError("Erro: E2B_API_KEY não encontrada. Verifique o arquivo .env")

        # Define a variável de ambiente para a biblioteca e2b ler internamente
        os.environ["E2B_API_KEY"] = self.api_key

    async def investigate(self, query: str, context_data: str) -> str:
        # Verifica se a chave existe antes de tentar criar o Sandbox
        if not os.getenv("E2B_API_KEY"):
            logger.error("Tentativa de investigação abortada: E2B_API_KEY ausente.")
            return "Erro: Configuração de Sandbox ausente."

        logger.info("RLM: Iniciando investigação em Sandbox E2B...")

        # Script de filtragem (mesmo da versão anterior, garantindo repr() para segurança)
        investigation_script = f"""
import re
context = {repr(context_data)}
query = {repr(query)}

findings = []
lines = [l.strip() for l in context.split('\\n') if l.strip()]
query_words = set(query.lower().split())

for line in lines:
    if any(word in line.lower() for word in query_words):
        findings.append(line)

print("\\n".join(findings[:10]) if findings else "Sem evidências.")
"""

        try:
            # CORREÇÃO: Usar Sandbox.create() conforme a documentação
            with Sandbox.create() as sandbox:
                execution = sandbox.run_code(investigation_script)

                if execution.error:
                    logger.error(f"RLM Sandbox Error: {execution.error}")
                    return f"Erro na investigação: {execution.error}"

                return execution.text

        except Exception as e:
            logger.error(f"Falha ao conectar com E2B: {e}", exc_info=True)
            return "Investigação falhou por conectividade."