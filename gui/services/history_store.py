import csv
import sqlite3
from datetime import datetime

from gui.services.paths import history_db_path


class HistoryStore:
    def __init__(self):
        self.db_path = history_db_path()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    decision TEXT NOT NULL,
                    profile_id TEXT,
                    audio_path TEXT NOT NULL,
                    notes TEXT
                )
                """
            )
            conn.commit()

    def add_entry(self, score, threshold, decision, profile_id, audio_path, notes=""):
        created_at = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO history (
                    created_at, score, threshold, decision,
                    profile_id, audio_path, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    float(score),
                    float(threshold),
                    decision,
                    profile_id,
                    audio_path,
                    notes,
                ),
            )
            conn.commit()

    def list_entries(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, score, threshold, decision,
                       profile_id, audio_path, notes
                FROM history
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_entries(self, entry_ids):
        if not entry_ids:
            return
        placeholders = ",".join("?" for _ in entry_ids)
        with self._connect() as conn:
            conn.execute(f"DELETE FROM history WHERE id IN ({placeholders})", entry_ids)
            conn.commit()

    def clear(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM history")
            conn.commit()

    def export_csv(self, output_path):
        rows = self.list_entries()
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "created_at",
                    "score",
                    "threshold",
                    "decision",
                    "profile_id",
                    "audio_path",
                    "notes",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
