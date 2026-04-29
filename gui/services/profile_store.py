import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from gui.services.paths import profiles_db_path, profiles_dir


def l2_normalize(x, axis=1, eps=1e-12):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def build_merged_embedding(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    embeddings = l2_normalize(embeddings, axis=1)
    return l2_normalize(embeddings.mean(axis=0, keepdims=True), axis=1)


class ProfileStore:
    def __init__(self):
        self.db_path = profiles_db_path()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    profile_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    embedding_path TEXT NOT NULL,
                    num_samples INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL,
                    backend_mode TEXT NOT NULL DEFAULT 'local_onnx',
                    model_fingerprint TEXT NOT NULL DEFAULT ''
                )
                """
            )
            self._ensure_schema(conn)
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_profiles_backend_active
                ON profiles (backend_mode, is_active, created_at DESC)
                """
            )
            conn.commit()

    @staticmethod
    def _ensure_schema(conn):
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(profiles)").fetchall()
        }
        if "backend_mode" not in columns:
            conn.execute(
                "ALTER TABLE profiles ADD COLUMN backend_mode TEXT NOT NULL DEFAULT 'local_onnx'"
            )
        if "model_fingerprint" not in columns:
            conn.execute(
                "ALTER TABLE profiles ADD COLUMN model_fingerprint TEXT NOT NULL DEFAULT ''"
            )

    def save_active_profile(
        self,
        display_name,
        embeddings,
        num_samples,
        backend_mode="local_onnx",
        model_fingerprint="",
    ):
        merged = build_merged_embedding(embeddings)

        profile_id = uuid4().hex
        embedding_path = Path(profiles_dir()) / f"{profile_id}.npy"
        np.save(embedding_path, merged.astype(np.float32))
        created_at = datetime.now().isoformat(timespec="seconds")

        with self._connect() as conn:
            conn.execute(
                "UPDATE profiles SET is_active = 0 WHERE backend_mode = ?",
                (str(backend_mode),),
            )
            conn.execute(
                """
                INSERT INTO profiles (
                    profile_id, display_name, embedding_path,
                    num_samples, created_at, is_active,
                    backend_mode, model_fingerprint
                ) VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    profile_id,
                    display_name,
                    str(embedding_path),
                    int(num_samples),
                    created_at,
                    str(backend_mode),
                    str(model_fingerprint),
                ),
            )
            conn.commit()

        return self.get_active_profile(
            backend_mode=backend_mode,
            model_fingerprint=model_fingerprint,
        )

    def get_active_profile(
        self,
        backend_mode=None,
        model_fingerprint=None,
        allow_model_mismatch=False,
    ):
        backend_mode = None if backend_mode is None else str(backend_mode)
        model_fingerprint = "" if model_fingerprint is None else str(model_fingerprint)

        with self._connect() as conn:
            if backend_mode is None:
                row = conn.execute(
                    """
                    SELECT profile_id, display_name, embedding_path,
                           num_samples, created_at, is_active,
                           backend_mode, model_fingerprint
                    FROM profiles
                    WHERE is_active = 1
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT profile_id, display_name, embedding_path,
                           num_samples, created_at, is_active,
                           backend_mode, model_fingerprint
                    FROM profiles
                    WHERE is_active = 1 AND backend_mode = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (backend_mode,),
                ).fetchone()
        if row is None:
            return None

        profile = dict(row)
        stored_fingerprint = str(profile.get("model_fingerprint", ""))
        mismatch = bool(model_fingerprint and stored_fingerprint != model_fingerprint)
        if mismatch and not allow_model_mismatch:
            return None

        embedding_path = Path(profile["embedding_path"])
        profile["embedding"] = (
            np.load(embedding_path).astype(np.float32) if embedding_path.is_file() else None
        )
        profile["model_fingerprint_mismatch"] = mismatch
        return profile

    def delete_active_profile(self, backend_mode=None):
        profile = self.get_active_profile(
            backend_mode=backend_mode,
            allow_model_mismatch=True,
        )
        if profile is None:
            return False

        embedding_path = Path(profile["embedding_path"])
        if embedding_path.is_file():
            embedding_path.unlink()

        with self._connect() as conn:
            conn.execute("DELETE FROM profiles WHERE profile_id = ?", (profile["profile_id"],))
            conn.commit()
        return True
