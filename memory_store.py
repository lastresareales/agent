import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DB_PATH = Path("memory.sqlite3")
DB_PATH_ENV_VAR = "ENTITY_RECOGNITION_MEMORY_DB"


def get_db_path():
    return Path(os.getenv(DB_PATH_ENV_VAR, DEFAULT_DB_PATH))


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path=None):
    path = Path(db_path) if db_path is not None else get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path=None):
    with connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS learned_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                relation TEXT NOT NULL,
                target TEXT NOT NULL,
                source_label TEXT NOT NULL,
                target_label TEXT NOT NULL,
                source_text TEXT NOT NULL,
                confidence REAL,
                occurrences INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(source, relation, target, source_text)
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_learned_facts_source ON learned_facts(source)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_learned_facts_target ON learned_facts(target)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_learned_facts_relation ON learned_facts(relation)")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'chat',
                importance REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_conversation_memories_role ON conversation_memories(role)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_conversation_memories_source ON conversation_memories(source)")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversation_memories_created_at ON conversation_memories(created_at)"
        )


def learn_from_extraction(extraction, db_path=None):
    init_db(db_path)
    entities_by_text = {entity["text"]: entity for entity in extraction.get("entities", [])}
    learned = []
    now = utc_now()

    with connect(db_path) as connection:
        for edge in extraction.get("graph", {}).get("edges", []):
            source_entity = entities_by_text.get(edge["source"])
            target_entity = entities_by_text.get(edge["target"])
            if source_entity is None or target_entity is None:
                continue

            confidence_values = [
                value
                for value in (source_entity.get("confidence"), target_entity.get("confidence"))
                if value is not None
            ]
            confidence = min(confidence_values) if confidence_values else None
            row = {
                "source": edge["source"],
                "relation": edge["relation"],
                "target": edge["target"],
                "source_label": source_entity["label"],
                "target_label": target_entity["label"],
                "source_text": extraction["text"],
                "confidence": confidence,
                "created_at": now,
                "updated_at": now,
            }
            connection.execute(
                """
                INSERT INTO learned_facts (
                    source, relation, target, source_label, target_label, source_text,
                    confidence, created_at, updated_at
                )
                VALUES (
                    :source, :relation, :target, :source_label, :target_label, :source_text,
                    :confidence, :created_at, :updated_at
                )
                ON CONFLICT(source, relation, target, source_text)
                DO UPDATE SET
                    confidence = COALESCE(excluded.confidence, learned_facts.confidence),
                    occurrences = learned_facts.occurrences + 1,
                    updated_at = excluded.updated_at
                """,
                row,
            )
            learned.append({key: row[key] for key in ("source", "relation", "target", "confidence")})

    return learned


def search_facts(query=None, limit=25, db_path=None):
    init_db(db_path)
    limit = max(1, min(int(limit), 100))

    parameters = {"limit": limit}
    where_clause = ""
    if query:
        parameters["query"] = f"%{query.lower()}%"
        where_clause = """
            WHERE lower(source) LIKE :query
               OR lower(relation) LIKE :query
               OR lower(target) LIKE :query
               OR lower(source_text) LIKE :query
        """

    with connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT
                id, source, relation, target, source_label, target_label, source_text,
                confidence, occurrences, created_at, updated_at
            FROM learned_facts
            {where_clause}
            ORDER BY updated_at DESC, occurrences DESC, id DESC
            LIMIT :limit
            """,
            parameters,
        ).fetchall()

    return [dict(row) for row in rows]


def estimate_importance(content):
    lowered = content.lower()
    important_markers = [
        "remember",
        "my name",
        "i am",
        "i'm",
        "i like",
        "i prefer",
        "i want",
        "my goal",
        "important",
    ]
    if any(marker in lowered for marker in important_markers):
        return 0.9
    if content.strip().endswith("?"):
        return 0.35
    return 0.6


def save_message(role, content, source="chat", importance=None, db_path=None):
    init_db(db_path)
    now = utc_now()
    score = estimate_importance(content) if importance is None else importance
    with connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO conversation_memories (role, content, source, importance, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (role, content, source, score, now),
        )
        memory_id = cursor.lastrowid

    return {
        "id": memory_id,
        "role": role,
        "content": content,
        "source": source,
        "importance": score,
        "created_at": now,
    }


def search_messages(query=None, limit=25, db_path=None):
    init_db(db_path)
    limit = max(1, min(int(limit), 100))
    parameters = {"limit": limit}
    where_clause = ""
    if query:
        parameters["query"] = f"%{query.lower()}%"
        where_clause = "WHERE lower(content) LIKE :query OR lower(source) LIKE :query"

    with connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT id, role, content, source, importance, created_at
            FROM conversation_memories
            {where_clause}
            ORDER BY importance DESC, created_at DESC, id DESC
            LIMIT :limit
            """,
            parameters,
        ).fetchall()

    return [dict(row) for row in rows]


def memory_summary(db_path=None):
    init_db(db_path)
    with connect(db_path) as connection:
        facts = connection.execute("SELECT COUNT(*) AS count FROM learned_facts").fetchone()["count"]
        occurrences = connection.execute(
            "SELECT COALESCE(SUM(occurrences), 0) AS count FROM learned_facts"
        ).fetchone()["count"]
        messages = connection.execute("SELECT COUNT(*) AS count FROM conversation_memories").fetchone()["count"]

    return {"facts": facts, "occurrences": occurrences, "messages": messages}
