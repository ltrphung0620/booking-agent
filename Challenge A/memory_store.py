"""Project rules in JSON, with revision checks, audit history and atomic writes."""

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class MemoryFileError(ValueError):
    pass


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    topic: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")
    value: str = Field(min_length=1, max_length=2000)
    status: Literal["active", "superseded"] = "active"
    created_at: str
    source_message: str = Field(min_length=1, max_length=6000)
    session_id: str
    supersedes: str | None = None
    superseded_by: str | None = None


class MemoryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    revision: int = Field(default=0, ge=0)
    records: list[Rule] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_history(self):
        by_id = {r.id: r for r in self.records}
        active_topics = [r.topic for r in self.records if r.status == "active"]
        if (len(by_id) != len(self.records) or self.revision != len(self.records)
                or len(active_topics) != len(set(active_topics))):
            raise ValueError("Memory history is inconsistent")
        for r in self.records:
            if r.status == "superseded":
                successor = by_id.get(r.superseded_by)
                if successor is None or successor.supersedes != r.id or successor.topic != r.topic:
                    raise ValueError("Missing replacement in memory history")
            elif r.superseded_by is not None:
                raise ValueError("Active rule cannot be superseded")
            if r.supersedes is not None:
                previous = by_id.get(r.supersedes)
                if previous is None or previous.superseded_by != r.id:
                    raise ValueError("Missing predecessor in memory history")
        return self


class ProjectMemoryStore:
    # Separate Streamlit sessions can own different store objects for the same file.
    _thread_lock = RLock()

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock, self.path.with_suffix(".lock").open("a+b") as lock:
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                lock.seek(0)
                if os.name == "nt":
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read(self) -> MemoryDocument:
        if not self.path.exists():
            return MemoryDocument()
        try:
            return MemoryDocument.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (ValidationError, UnicodeError) as exc:
            raise MemoryFileError("File memory không hợp lệ; cần kiểm tra file trước khi tiếp tục. "
                                  "Chương trình không tự xóa hoặc ghi đè dữ liệu lỗi.") from exc

    @staticmethod
    def _snapshot(doc: MemoryDocument) -> dict:
        # Deliberately exclude superseded rules and historical user messages.
        return {"revision": doc.revision, "active_rules": [
            {"id": r.id, "topic": r.topic, "value": r.value, "created_at": r.created_at}
            for r in doc.records if r.status == "active"
        ]}

    def snapshot(self) -> dict:
        with self._locked():
            return self._snapshot(self._read())

    def history(self) -> dict:
        with self._locked():
            return self._read().model_dump(mode="json")

    def remember(self, topic: str, value: str, *, source_message: str,
                 session_id: str, expected_revision: int) -> dict:
        candidate = Rule(id=uuid4().hex, topic=topic.strip(), value=value.strip(),
                         created_at=datetime.now(timezone.utc).isoformat(),
                         source_message=source_message, session_id=session_id)
        with self._locked():
            doc = self._read()
            if expected_revision != doc.revision:
                return {"status": "conflict", "reason": "Memory đã đổi; đọc revision mới trước khi cập nhật.",
                        **self._snapshot(doc)}
            old = next((r for r in doc.records if r.topic == candidate.topic and r.status == "active"), None)
            if old and old.value == candidate.value:
                return {"status": "ok", "changed": False, **self._snapshot(doc)}
            if old:
                old.status = "superseded"
                old.superseded_by = candidate.id
                candidate.supersedes = old.id
            doc.records.append(candidate)
            doc.revision += 1
            # Validate the entire replacement chain before touching the file.
            doc = MemoryDocument.model_validate(doc.model_dump())
            name = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                                 dir=self.path.parent, delete=False) as output:
                    name = output.name
                    output.write(doc.model_dump_json(indent=2))
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(name, self.path)
            finally:
                if name and Path(name).exists():
                    Path(name).unlink()
            return {"status": "ok", "changed": True, **self._snapshot(doc)}
