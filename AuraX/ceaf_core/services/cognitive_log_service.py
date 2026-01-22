# ARQUIVO COMPLETO: ceaf_core/services/cognitive_log_service.py

import sqlite3
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("CognitiveLogService")

LOG_DB_FILENAME = "cognitive_turn_history.sqlite"


class CognitiveLogService:
    """
    Armazena e consulta o histórico de pacotes Genlang de cada turno.
    Usa SQLite com JSON para armazenamento estruturado e consultável.
    """

    def __init__(self, persistence_path: Path):
        self.db_path = persistence_path / LOG_DB_FILENAME
        self._initialize_db()
        logger.info(f"CognitiveLogService inicializado com banco de dados em: {self.db_path.resolve()}")

    def _get_db_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize_db(self):
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                # <<< MUDANÇA: ADICIONADA A COLUNA deliberation_history >>>
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS turn_history (
                        turn_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        cognitive_state_packet TEXT NOT NULL,
                        response_packet TEXT NOT NULL,
                        mcl_guidance_json TEXT,
                        deliberation_history TEXT, 
                        intent_text TEXT,
                        final_confidence REAL,
                        agency_used INTEGER
                    )
                """)
                # <<< FIM DA MUDANÇA >>>
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON turn_history (timestamp);")
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Falha ao inicializar o banco de dados do CognitiveLogService: {e}", exc_info=True)
            raise

    def log_turn(
            self,
            turn_id: str,
            session_id: str,
            cognitive_state_packet: Dict[str, Any],
            response_packet: Dict[str, Any],
            mcl_guidance: Dict[str, Any]
    ):
        """Registra os pacotes Genlang e a orientação do MCL de um turno completo."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                agency_used_flag = 1 if mcl_guidance.get("agency_parameters", {}).get("use_agency_simulation") else 0
                mcl_guidance_json_str = json.dumps(mcl_guidance, default=str)

                # <<< MUDANÇA: EXTRAI E SERIALIZA O deliberation_history >>>
                deliberation_history_list = cognitive_state_packet.get("deliberation_history", [])
                deliberation_history_json_str = json.dumps(deliberation_history_list)
                # <<< FIM DA MUDANÇA >>>

                # <<< MUDANÇA: ATUALIZA O INSERT STATEMENT >>>
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
                        time.time(),
                        json.dumps(cognitive_state_packet, default=str),
                        json.dumps(response_packet, default=str),
                        mcl_guidance_json_str,
                        deliberation_history_json_str,  # Salva o novo dado
                        cognitive_state_packet.get("original_intent", {}).get("query_vector", {}).get("source_text"),
                        response_packet.get("confidence_score"),
                        agency_used_flag
                    ),
                )
                # <<< FIM DA MUDANÇA >>>
                conn.commit()
                logger.debug(f"Turno '{turn_id}' registrado no CognitiveLogService com deliberation_history.")
        except sqlite3.Error as e:
            logger.error(f"Falha ao registrar turno '{turn_id}' no log cognitivo: {e}")

    def get_recent_turns(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Recupera os N turnos mais recentes para análise de forma robusta."""
        results = []
        try:
            with self._get_db_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                # <<< MUDANÇA: ATUALIZA O SELECT STATEMENT >>>
                cursor.execute(
                    "SELECT * FROM turn_history ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
                # <<< FIM DA MUDANÇA >>>
                rows = cursor.fetchall()

                for row in rows:
                    try:
                        row_dict = dict(row)
                        row_dict["cognitive_state_packet"] = json.loads(row_dict["cognitive_state_packet"])
                        row_dict["response_packet"] = json.loads(row_dict["response_packet"])

                        if row_dict.get("mcl_guidance_json"):
                            row_dict["mcl_guidance"] = json.loads(row_dict["mcl_guidance_json"])
                        else:
                            row_dict["mcl_guidance"] = {}

                        # <<< MUDANÇA: DESERIALIZA O deliberation_history >>>
                        if row_dict.get("deliberation_history"):
                            row_dict["deliberation_history"] = json.loads(row_dict["deliberation_history"])
                        else:
                            row_dict["deliberation_history"] = []
                        # <<< FIM DA MUDANÇA >>>

                        results.append(row_dict)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(
                            f"Pulando turno de log corrompido (ID: {row.get('turn_id', 'N/A')}) devido a erro de parsing: {e}"
                        )
                        continue

        except sqlite3.Error as e:
            logger.error(f"Falha na consulta ao banco de dados de logs: {e}")

        return results