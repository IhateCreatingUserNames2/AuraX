# Em: ceaf_core/services/evolution_log_service.py

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("EvolutionLogger")


class EvolutionLogger:
    """
    Registra um snapshot do estado evolutivo do agente a cada turno em um arquivo .jsonl.
    """

    def __init__(self, persistence_path: Path):
        log_dir = persistence_path / "logs"
        log_dir.mkdir(exist_ok=True, parents=True)
        self.log_file_path = log_dir / "evolution_log.jsonl"
        logger.info(f"EvolutionLogger inicializado. Logs serão salvos em: {self.log_file_path}")

    def log_turn_state(self, turn_data: Dict[str, Any]):
        """
        Recebe um dicionário com o estado do turno e o escreve como uma linha no arquivo de log.
        """
        try:
            # Adiciona metadados essenciais para rastreamento
            log_entry = {
                "log_timestamp_utc": datetime.utcnow().isoformat(),
                **turn_data  # Desempacota todos os dados do turno aqui
            }

            # Converte para JSON e escreve no arquivo
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")  # default=str lida com tipos não serializáveis

        except Exception as e:
            logger.error(f"Falha ao escrever no log de evolução: {e}", exc_info=True)