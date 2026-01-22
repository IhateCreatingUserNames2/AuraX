# ceaf_core/services/persistent_log_service.py

import sqlite3
import json
import time
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

DEFAULT_LOG_DB_PATH = "./data/ceaf_persistent_logs.sqlite"
LOG_DB_FILE = Path(os.getenv("CEAF_PERSISTENT_LOG_DB_PATH", DEFAULT_LOG_DB_PATH))
LOG_TABLE_NAME = "ceaf_logs"

class PersistentLogService:
    """
    A service for persistently storing and retrieving structured log events
    from the CEAF system, using an SQLite database.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path or LOG_DB_FILE)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
        logger.info(f"PersistentLogService initialized with database at: {self.db_path.resolve()}")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Establishes and returns a database connection."""
        # check_same_thread=False is suitable for many async frameworks if the connection
        # is managed per request/task or if operations are short-lived.
        # For more complex scenarios, a proper connection pool or fully async driver (aiosqlite)
        # might be needed, but for this logging service, this should be acceptable.
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _initialize_db(self):
        """Creates the log table if it doesn't already exist."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {LOG_TABLE_NAME} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        event_type TEXT NOT NULL,
                        source_agent TEXT,
                        session_id TEXT,
                        turn_id TEXT,
                        data_payload TEXT, -- JSON formatted string
                        tags TEXT -- Comma-separated string or JSON array string
                    )
                """)
                # Add indexes for common query fields
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_event_type ON {LOG_TABLE_NAME} (event_type);")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_timestamp ON {LOG_TABLE_NAME} (timestamp);")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_session_id ON {LOG_TABLE_NAME} (session_id);")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_turn_id ON {LOG_TABLE_NAME} (turn_id);")
                conn.commit()
            logger.debug(f"Database table '{LOG_TABLE_NAME}' initialized/verified.")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database table '{LOG_TABLE_NAME}': {e}", exc_info=True)
            raise

    def log_event(
        self,
        event_type: str,
        data_payload: Dict[str, Any],
        source_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[int]:
        """
        Logs a structured event to the persistent store.

        Args:
            event_type: A string identifying the type of event (e.g., "ORA_LLM_REQUEST", "MCL_ANALYSIS").
            data_payload: A dictionary containing the core data of the event. Will be stored as JSON.
            source_agent: The name of the agent or module that generated the event.
            session_id: The session ID associated with this event, if applicable.
            turn_id: The turn ID (or invocation ID) associated with this event, if applicable.
            tags: A list of string tags for categorization and easier querying.

        Returns:
            The row ID of the inserted log entry, or None if insertion failed.
        """
        current_timestamp = time.time()
        data_payload_json = json.dumps(data_payload, default=str) # default=str for non-serializable types
        tags_str = ",".join(sorted(list(set(tags)))) if tags else None

        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    INSERT INTO {LOG_TABLE_NAME}
                    (timestamp, event_type, source_agent, session_id, turn_id, data_payload, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        current_timestamp,
                        event_type,
                        source_agent,
                        session_id,
                        turn_id,
                        data_payload_json,
                        tags_str,
                    ),
                )
                conn.commit()
                last_row_id = cursor.lastrowid
                logger.debug(f"Logged event: Type='{event_type}', Source='{source_agent}', ID={last_row_id}")
                return last_row_id
        except sqlite3.Error as e:
            logger.error(
                f"Failed to log event (Type: {event_type}, Source: {source_agent}): {e}",
                exc_info=True
            )
            return None

    def query_logs(
        self,
        event_type: Optional[str] = None,
        source_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        time_window_start_ts: Optional[float] = None,
        time_window_end_ts: Optional[float] = None,
        tags_contain_any: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        order_by_timestamp_desc: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Queries the persistent log store based on various criteria.

        Args:
            event_type: Filter by event type.
            source_agent: Filter by source agent.
            session_id: Filter by session ID.
            turn_id: Filter by turn ID.
            time_window_start_ts: Unix timestamp for the start of the query window.
            time_window_end_ts: Unix timestamp for the end of the query window.
            tags_contain_any: List of tags; returns logs that have at least one of these tags.
            limit: Maximum number of log entries to return.
            offset: Number of log entries to skip (for pagination).
            order_by_timestamp_desc: If True, order by timestamp descending (newest first).

        Returns:
            A list of dictionaries, where each dictionary represents a log entry.
            The 'data_payload' will be a dictionary (parsed from JSON), and 'tags' will be a list.
        """
        conditions: List[str] = []
        params: List[Any] = []

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if source_agent:
            conditions.append("source_agent = ?")
            params.append(source_agent)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if turn_id:
            conditions.append("turn_id = ?")
            params.append(turn_id)
        if time_window_start_ts:
            conditions.append("timestamp >= ?")
            params.append(time_window_start_ts)
        if time_window_end_ts:
            conditions.append("timestamp <= ?")
            params.append(time_window_end_ts)

        if tags_contain_any and tags_contain_any:
            tag_conditions = []
            for tag in tags_contain_any:
                # This assumes tags are stored comma-separated.
                # For more robust tag querying, a separate tags table and many-to-many relationship
                # or SQLite's JSON1 extension with json_each would be better.
                # Simple LIKE for now. Add leading/trailing comma to query and stored tags for better matching.
                tag_conditions.append("',' || tags || ',' LIKE ?")
                params.append(f"%,{tag},%") # Match ",tag,"
            if tag_conditions:
                conditions.append(f"({ ' OR '.join(tag_conditions) })")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        order_clause = f"ORDER BY timestamp {'DESC' if order_by_timestamp_desc else 'ASC'}"
        limit_clause = "LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        query = f"SELECT id, timestamp, event_type, source_agent, session_id, turn_id, data_payload, tags FROM {LOG_TABLE_NAME} {where_clause} {order_clause} {limit_clause}"

        results: List[Dict[str, Any]] = []
        try:
            with self._get_db_connection() as conn:
                # Make rows returned as dictionaries
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                for row in rows:
                    row_dict = dict(row)
                    try:
                        row_dict["data_payload"] = json.loads(row_dict["data_payload"]) if row_dict["data_payload"] else {}
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse data_payload for log ID {row_dict['id']}. Payload: {row_dict['data_payload']}")
                        row_dict["data_payload"] = {"error": "failed_to_parse_json", "raw": row_dict["data_payload"]}

                    tags_str = row_dict.get("tags")
                    row_dict["tags"] = [t.strip() for t in tags_str.split(',') if t.strip()] if tags_str else []
                    results.append(row_dict)
            logger.debug(f"Query executed: '{query[:100]}...'. Params: {params}. Found {len(results)} results.")
        except sqlite3.Error as e:
            logger.error(f"Failed to query logs: {e}. Query: {query}", exc_info=True)
        return results

    def get_total_log_count(self) -> int:
        """Returns the total number of log entries."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {LOG_TABLE_NAME}")
                count = cursor.fetchone()
                return count[0] if count else 0
        except sqlite3.Error as e:
            logger.error(f"Failed to get total log count: {e}", exc_info=True)
            return 0

    def close(self):
        """
        This method isn't strictly necessary for sqlite3 as connections are typically
        managed with context managers (`with ... as conn:`).
        If a persistent connection were held by the class instance, this would close it.
        """
        logger.info("PersistentLogService close called (typically no-op for connection-per-operation).")
