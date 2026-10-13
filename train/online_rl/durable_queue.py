"""SQLite-backed durable trajectory queue for asynchronous collectors and learners."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class DurableQueue:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        # The launcher normally initializes the database before spawning workers. Retrying here
        # also makes independently started collectors/learners safe under concurrent startup.
        last_error = None
        for attempt in range(20):
            try:
                with self._conn() as c:
                    c.execute("PRAGMA busy_timeout=30000")
                    c.execute("PRAGMA journal_mode=WAL")
                    c.execute("""CREATE TABLE IF NOT EXISTS messages(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trajectory_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        created REAL NOT NULL,
                        claimed_by TEXT,
                        claimed REAL,
                        done INTEGER NOT NULL DEFAULT 0,
                        UNIQUE(trajectory_id, role))""")
                last_error = None
                break
            except sqlite3.OperationalError as exc:
                last_error = exc
                if "locked" not in str(exc).lower():
                    raise
                time.sleep(0.1 * (attempt + 1))
        if last_error is not None:
            raise last_error

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        return c

    def put(self, trajectory_id: str, role: str, payload: Dict[str, Any]) -> bool:
        with self._conn() as c:
            cur = c.execute("INSERT OR IGNORE INTO messages(trajectory_id,role,payload,created) VALUES(?,?,?,?)",
                            (trajectory_id, role, json.dumps(payload, ensure_ascii=False), time.time()))
            return cur.rowcount > 0

    def claim(self, role: str, worker: str, stale_after: float = 1800) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("""SELECT * FROM messages WHERE role=? AND done=0
                AND (claimed_by IS NULL OR claimed<?) ORDER BY id LIMIT 1""", (role, now-stale_after)).fetchone()
            if row is None:
                c.execute("COMMIT"); return None
            c.execute("UPDATE messages SET claimed_by=?,claimed=? WHERE id=?", (worker, now, row["id"]))
            c.execute("COMMIT")
            out = dict(row); out["payload"] = json.loads(out["payload"])
            return out

    def ack(self, message_id: int, worker: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE messages SET done=1 WHERE id=? AND claimed_by=?", (message_id, worker))

    def mark_published(self, trajectory_id: str, role: str) -> int:
        """Finish a queued record whose policy version was published before a crash."""
        with self._conn() as c:
            cur = c.execute("UPDATE messages SET done=1 WHERE trajectory_id=? AND role=?",
                            (trajectory_id, role))
            return cur.rowcount

    def release(self, message_id: int, worker: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE messages SET claimed_by=NULL,claimed=NULL WHERE id=? AND claimed_by=?",
                      (message_id, worker))

    def stats(self) -> Dict[str, Dict[str, int]]:
        with self._conn() as c:
            rows = c.execute("SELECT role,done,COUNT(*) n FROM messages GROUP BY role,done").fetchall()
        out: Dict[str, Dict[str, int]] = {}
        for r in rows:
            d = out.setdefault(r["role"], {"pending": 0, "done": 0})
            d["done" if r["done"] else "pending"] = r["n"]
        return out

    def payload_for_episode(self, episode: int) -> Optional[Dict[str, Any]]:
        """Return the newest stored trajectory payload for an episode."""
        with self._conn() as c:
            rows = c.execute("SELECT payload FROM messages ORDER BY id DESC").fetchall()
        for row in rows:
            payload = json.loads(row["payload"])
            if int(payload.get("episode", -1)) == int(episode):
                return payload
        return None

    def reset_incomplete_claims(self) -> int:
        """Release claims left by workers that have already been stopped."""
        with self._conn() as c:
            cur = c.execute("UPDATE messages SET claimed_by=NULL,claimed=NULL WHERE done=0")
            return cur.rowcount
