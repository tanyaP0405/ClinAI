from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PATIENT_FIELDS = (
    "patient_id",
    "name",
    "age",
    "gender",
    "conversation",
    "note",
    "summary",
    "timeline",
    "keywords",
    "prescriptions",
)


class PatientRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL DEFAULT 'NA',
                    age TEXT NOT NULL DEFAULT 'NA',
                    gender TEXT NOT NULL DEFAULT 'NA',
                    conversation TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    timeline TEXT NOT NULL DEFAULT '[]',
                    keywords TEXT NOT NULL DEFAULT '',
                    prescriptions TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_name TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chat_session
                    ON chat_messages(session_id, id);
                """
            )

    def upsert_patient(self, record: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        values = {field: str(record.get(field, "") or "") for field in PATIENT_FIELDS}
        values["name"] = values["name"] or "NA"
        values["age"] = values["age"] or "NA"
        values["gender"] = values["gender"] or "NA"
        values["timeline"] = self._normalize_timeline(record.get("timeline"))

        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO patients (
                    patient_id, name, age, gender, conversation, note, summary,
                    timeline, keywords, prescriptions, created_at, updated_at
                ) VALUES (
                    :patient_id, :name, :age, :gender, :conversation, :note,
                    :summary, :timeline, :keywords, :prescriptions, :created_at,
                    :updated_at
                )
                ON CONFLICT(patient_id) DO UPDATE SET
                    name=excluded.name,
                    age=excluded.age,
                    gender=excluded.gender,
                    conversation=excluded.conversation,
                    note=excluded.note,
                    summary=excluded.summary,
                    timeline=excluded.timeline,
                    keywords=excluded.keywords,
                    prescriptions=excluded.prescriptions,
                    updated_at=excluded.updated_at
                """,
                {**values, "created_at": now, "updated_at": now},
            )
        return self.get_patient(values["patient_id"]) or values

    def get_patient(self, patient_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_patients(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT patient_id, name, age, gender, summary, keywords, updated_at
                FROM patients
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_field(self, patient_id: str, field: str, value: Any) -> bool:
        allowed = {"summary", "timeline", "prescriptions", "keywords"}
        if field not in allowed:
            raise ValueError(f"Unsupported field: {field}")
        if field == "timeline":
            value = self._normalize_timeline(value)
        with self.connection() as connection:
            result = connection.execute(
                f"UPDATE patients SET {field} = ?, updated_at = ? WHERE patient_id = ?",
                (str(value), datetime.now(timezone.utc).isoformat(), patient_id),
            )
        return result.rowcount > 0

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_name: str | None = None,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO chat_messages (
                    session_id, role, content, tool_name, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    tool_name,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_messages(
        self, session_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT role, content, tool_name
                FROM (
                    SELECT id, role, content, tool_name
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _normalize_timeline(value: Any) -> str:
        if isinstance(value, list):
            return json.dumps([str(item) for item in value])
        if not value:
            return "[]"
        text = str(value)
        try:
            parsed = json.loads(text)
            return json.dumps(parsed if isinstance(parsed, list) else [text])
        except json.JSONDecodeError:
            return json.dumps([text])
