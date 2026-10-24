"""Transactional local persistence for reproducible simulation experiments."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ExperimentStore:
    """SQLite metadata store; simulation trajectories are intentionally ephemeral."""

    def __init__(self, path: str | Path = "data/experiments.sqlite3") -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level="IMMEDIATE")
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL CHECK(length(name) BETWEEN 1 AND 120),
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                model TEXT NOT NULL CHECK(model IN ('gbm','ou','jump_diffusion','garch')),
                parameters_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                metrics_json TEXT,
                status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),
                error TEXT
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS experiments_created_idx ON experiments(created_at DESC)")
            connection.execute("""CREATE TABLE IF NOT EXISTS path_forensics_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS path_forensics_created_idx ON path_forensics_runs(created_at DESC)")

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        for column, target in (("parameters_json", "parameters"), ("metadata_json", "metadata"), ("metrics_json", "metrics")):
            raw = record.pop(column)
            record[target] = json.loads(raw) if raw is not None else None
        return record

    def create(self, *, name: str, description: str | None, model: str,
               parameters: dict[str, Any], metadata: dict[str, Any]) -> str:
        experiment_id = str(uuid.uuid4())
        now = _now()
        with self._connect() as connection:
            connection.execute("""INSERT INTO experiments
                (id,name,description,created_at,updated_at,model,parameters_json,metadata_json,status)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (experiment_id, name, description, now, now, model, _json(parameters), _json(metadata), "pending"))
        return experiment_id

    def set_running(self, experiment_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute("UPDATE experiments SET status='running',updated_at=? WHERE id=? AND status='pending'",
                                        (_now(), experiment_id))
            if cursor.rowcount != 1:
                raise RuntimeError("experiment could not enter running state")

    def complete(self, experiment_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute("""UPDATE experiments SET status='completed',metrics_json=?,error=NULL,updated_at=?
                WHERE id=? AND status='running'""", (_json(metrics), _now(), experiment_id))
            if cursor.rowcount != 1:
                raise RuntimeError("experiment could not be marked completed")
        return self.get(experiment_id)

    def fail(self, experiment_id: str, message: str) -> None:
        with self._connect() as connection:
            connection.execute("""UPDATE experiments SET status='failed',error=?,updated_at=?
                WHERE id=? AND status IN ('pending','running')""", (message[:2000], _now(), experiment_id))

    def get(self, experiment_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if row is None:
            raise KeyError(experiment_id)
        return self._decode(row)

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM experiments ORDER BY created_at DESC,id DESC LIMIT ?", (limit,)).fetchall()
        return [self._decode(row) for row in rows]

    def delete(self, experiment_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM experiments WHERE id=?", (experiment_id,))
            return cursor.rowcount == 1

    def save_path_forensics(self, payload: dict[str, Any]) -> str:
        """Persist exact path inputs and generated values for an opted-in run."""
        run_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute("INSERT INTO path_forensics_runs(id,created_at,payload_json) VALUES (?,?,?)",
                               (run_id, _now(), _json(payload)))
            connection.execute("""DELETE FROM path_forensics_runs WHERE id NOT IN
                (SELECT id FROM path_forensics_runs ORDER BY created_at DESC,id DESC LIMIT 25)""")
        return run_id

    def get_path_forensics(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT created_at,payload_json FROM path_forensics_runs WHERE id=?",
                                     (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return {"run_id": run_id, "created_at": row["created_at"], **json.loads(row["payload_json"])}

    def list_path_forensics(self, limit: int = 25) -> list[dict[str, Any]]:
        """List retained forensic runs without exposing their full trajectory payloads."""
        with self._connect() as connection:
            rows = connection.execute("SELECT id,created_at,payload_json FROM path_forensics_runs ORDER BY created_at DESC,id DESC LIMIT ?",
                                      (max(1, min(int(limit), 25)),)).fetchall()
        result = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            trajectories = payload.get("trajectories") or []
            result.append({"run_id": row["id"], "created_at": row["created_at"],
                "model": payload.get("model"), "parameters": payload.get("parameters", {}),
                "path_count": len(trajectories), "step_count": max((len(path) for path in trajectories), default=0)})
        return result
