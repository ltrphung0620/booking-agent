"""run_code: execute model code in a separate process; only its result returns."""

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

from langchain_core.tools import BaseTool, StructuredTool

from log_tools import approx_tokens

RUNTIME_CHILD = Path(__file__).resolve().parent / "runtime_child.py"
KEEP_ENV = ("SYSTEMROOT",)  # Windows Python cannot start without it; nothing else is passed.

RUN_CODE_DESCRIPTION = """Run a short Python program over the booking-system log files. Only its result reaches you.
1. Code: the BODY of a Python function. Use `return` for the final, JSON-serialisable result. Standard library only.
2. Call tools only through the `tools` object (synchronous, no await). A denied or failed call raises ToolError.
3. You only see the return value and print() output, truncated to {cap_chars} characters.
4. Filter, count and aggregate inside the code. Never return raw log lines or whole files.
   Include verification numbers: files_listed, files_scanned, lines_scanned, failed.
5. Limits: {timeout_s:g} s timeout, {max_calls} tool calls, no network, no files outside tools.*. Errors come back as "Error: <Type>: <message>".
6. SDK:
   tools.list_files(glob: str) -> list[str]     # e.g. tools.list_files("logs/*.log")
   tools.read_file(path: str) -> str            # whole file as text
   Log line: "YYYY-MM-DD HH:MM:SS LEVEL service [ERR_CODE] key=value ..."
"""


def _pump(stream, lines: queue.Queue) -> None:
    """Reader thread: only moves lines; it never touches the harness."""
    try:
        for line in stream:
            lines.put(line)
    finally:
        lines.put(None)
        stream.close()


def _cap(text: str, cap_chars: int, dropped: int) -> tuple[str, bool]:
    if len(text) <= cap_chars and not dropped:
        return text, False
    cut = max(len(text) - cap_chars, 0) + dropped
    return text[:cap_chars] + f"\n…[harness cắt {cut} ký tự]", True


def run_code(code: str, dispatch: Callable[[str, dict], dict], *, timeout_s: float = 10,
             max_calls: int = 300, cap_chars: int = 2000) -> dict:
    """dispatch runs on the calling thread: Harness.execute already holds the RLock there."""
    started = time.monotonic()
    workdir = tempfile.mkdtemp(prefix="run_code_")
    stats = {"runtime_tool_calls": 0, "runtime_tokens": 0, "duration_ms": 0}
    proc = None

    def finish(result: dict) -> dict:
        stats["duration_ms"] = round((time.monotonic() - started) * 1000)
        return {**result, "stats": stats}

    try:
        with tempfile.TemporaryFile("w+", encoding="utf-8") as errors:
            proc = subprocess.Popen(
                [sys.executable, "-I", str(RUNTIME_CHILD)], cwd=workdir,
                env={k: os.environ[k] for k in KEEP_ENV if k in os.environ},
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                text=True, encoding="utf-8")
            lines: queue.Queue = queue.Queue()
            threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True).start()

            def send(message: dict) -> None:
                proc.stdin.write(json.dumps(message, ensure_ascii=True) + "\n")
                proc.stdin.flush()

            send({"type": "run", "code": code})
            while True:
                try:
                    line = lines.get(timeout=max(started + timeout_s - time.monotonic(), 0))
                except queue.Empty:
                    proc.kill()
                    return finish({"status": "error",
                                   "reason": f"Timeout: harness dừng runtime sau {timeout_s:g} s"})
                if line is None:
                    proc.wait(timeout=5)
                    return finish({"status": "error",
                                   "reason": f"Error: RuntimeCrashed: exit code {proc.returncode}"})
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    proc.kill()
                    return finish({"status": "error", "reason": "Error: ProtocolError: runtime wrote to the protocol stream"})
                kind = message.get("type")
                if kind == "call":
                    stats["runtime_tool_calls"] += 1
                    if stats["runtime_tool_calls"] > max_calls:
                        reply = {"ok": False, "error": f"QuotaExceeded: more than {max_calls} tool calls"}
                    else:
                        try:
                            result = dispatch(str(message.get("name")), message.get("args") or {})
                            stats["runtime_tokens"] += approx_tokens(json.dumps(result, ensure_ascii=False))
                            reply = {"ok": True, "value": result}
                        except Exception as exc:  # A harness failure becomes a ToolError in the code.
                            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    send({"type": "result", "id": message.get("id"), **reply})
                elif kind == "done":
                    logs, value = message.get("logs", "").rstrip("\n"), message.get("value")
                    parts = [logs] if logs else []
                    if value is not None or not parts:
                        parts.append(json.dumps(value, ensure_ascii=False, indent=2))
                    output, truncated = _cap("\n".join(parts), cap_chars, message.get("dropped", 0))
                    return finish({"status": "ok", "output": output, "truncated": truncated})
                elif kind == "error":
                    return finish({"status": "error", "reason": "Error: " + str(message.get("message"))[:300]})
    except OSError as exc:  # Broken pipe: the child died before reading a reply.
        return finish({"status": "error", "reason": f"Error: RuntimeCrashed: {type(exc).__name__}"})
    finally:
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            try:
                proc.stdin.close()  # stdout is closed by the reader thread at EOF.
            except OSError:
                pass
        shutil.rmtree(workdir, ignore_errors=True)


def make_run_code_tool(harness, *, timeout_s: float = 10, max_calls: int = 300,
                       cap_chars: int = 2000) -> BaseTool:
    def run(code: str) -> dict:
        return run_code(code, harness.execute_inner, timeout_s=timeout_s,
                        max_calls=max_calls, cap_chars=cap_chars)

    description = RUN_CODE_DESCRIPTION.format(timeout_s=timeout_s, max_calls=max_calls,
                                              cap_chars=cap_chars)
    return StructuredTool.from_function(run, name="run_code", description=description)
