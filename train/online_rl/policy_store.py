"""Atomic versioned role-policy store for asynchronous AdCoEvo."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

ROLES = ("planner", "optimizer", "inspector")


class PolicyStore:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.json"

    def initialize(self, adapters: Dict[str, str]) -> None:
        with (self.root / "publish.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if self.manifest.exists():
                    return
                versions = {}
                for role in ROLES:
                    target = self.root / role / "v000000"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(adapters[role], target)
                    versions[role] = 0
                self._write({"versions": versions, "generation": 0})
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def read(self) -> Dict:
        return json.loads(self.manifest.read_text(encoding="utf-8"))

    def adapter(self, role: str, version: Optional[int] = None) -> Path:
        if version is None:
            version = int(self.read()["versions"][role])
        return self.root / role / f"v{version:06d}"

    def publish(self, role: str, source: str, parent_version: int, metrics: Dict) -> int:
        with (self.root / "publish.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                manifest = self.read()
                current = int(manifest["versions"][role])
                if current != parent_version:
                    raise RuntimeError(f"stale publish {role}: current={current} parent={parent_version}")
                version = current + 1
                target = self.root / role / f"v{version:06d}"
                tmp = target.parent / f".{target.name}.tmp"
                if tmp.exists():
                    shutil.rmtree(tmp)
                shutil.copytree(source, tmp)
                # A crash may leave this version directory behind before the
                # manifest commit. The locked manifest still names the parent.
                if target.exists():
                    shutil.rmtree(target)
                os.replace(tmp, target)
                manifest["versions"][role] = version
                manifest["generation"] = int(manifest.get("generation", 0)) + 1
                manifest.setdefault("updates", []).append({"role": role, "version": version,
                                                            "parent_version": parent_version, "metrics": metrics})
                self._write(manifest)
                return version
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _write(self, obj: Dict) -> None:
        fd, tmp = tempfile.mkstemp(prefix="manifest.", suffix=".json", dir=self.root)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, self.manifest)
