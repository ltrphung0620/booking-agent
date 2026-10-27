"""Log-reading tools, their path policy and the investigation harness."""

import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import bootstrap  # noqa: F401 -- establishes the shared demo import path
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from constraints import Constraints
from harness import Harness
from mock_tools import MockFlightService

LOG_DIR = "logs"


def approx_tokens(text: str) -> int:
    """Same 4 chars/token rule as langchain's count_tokens_approximately."""
    return max(1, round(len(text) / 4))


def _escapes(path: str) -> str | None:
    if PureWindowsPath(path).is_absolute() or PureWindowsPath(path).drive or path.startswith(("/", "\\")):
        return f"PermissionDenied: absolute path {path}"
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if ".." in parts:
        return f"PermissionDenied: {path}"
    hidden = next((part for part in parts if part.startswith(".")), None)
    return f"PermissionDenied: {hidden}" if hidden else None


def path_policy(path: str) -> str | None:
    """Return a denial reason, or None when path is a readable location inside logs/."""
    reason = _escapes(path)
    if reason:
        return reason
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if not parts or parts[0] != LOG_DIR:
        return f"PermissionDenied: only {LOG_DIR}/ is readable"
    return None


def make_log_tools(root: Path) -> dict[str, BaseTool]:
    root = Path(root).resolve()
    logs = root / LOG_DIR

    def inside_logs(path: Path) -> bool:
        resolved = path.resolve()  # Also rejects symlinks pointing out of logs/.
        return resolved.is_relative_to(logs) and not any(
            part.startswith(".") for part in resolved.relative_to(root).parts)

    @tool
    def list_files(glob: str = "logs/*.log") -> dict:
        """List readable log files matching a glob relative to the log root, e.g. logs/*.log."""
        reason = _escapes(glob)
        if reason:
            return {"status": "denied", "reason": reason}
        files = sorted(p.relative_to(root).as_posix() for p in root.glob(glob)
                       if p.is_file() and inside_logs(p))
        return {"status": "ok", "files": files}

    @tool
    def read_file(path: str) -> dict:
        """Read one whole log file, e.g. logs/booking-2026-10-07.log."""
        reason = path_policy(path)
        if reason:
            return {"status": "denied", "reason": reason}
        target = root / path.replace("\\", "/")
        if not inside_logs(target):
            return {"status": "denied", "reason": f"PermissionDenied: {path}"}
        if not target.is_file():
            return {"status": "not_found", "reason": f"NotFound: {path}"}
        return {"status": "ok", "path": path, "content": target.read_text(encoding="utf-8")}

    return {t.name: t for t in (list_files, read_file)}


class InvestigationHarness(Harness):
    def __init__(self, root: Path, *, max_model_calls: int = 40, max_tool_calls: int = 60):
        # Reuse the base lock, trace and budgets; the flight constraints/service stay unused.
        super().__init__(Constraints(), MockFlightService(), allow_payment=False,
                         max_model_calls=max_model_calls, max_tool_calls=max_tool_calls)
        self.root = Path(root)
        self.tools = make_log_tools(root)
        self.sdk_tools = make_log_tools(root)  # Separate registry for tools.* inside run_code.

    def check_permission(self, name: str, args: dict) -> str | None:
        if name == "read_file":
            return path_policy(args["path"])
        if name == "list_files":
            return _escapes(args["glob"])
        return super().check_permission(name, args)

    def execute_inner(self, name: str, args: dict) -> dict:
        """Gate for tools.* calls from the code runtime; its quota lives in code_runner."""
        with self.service.lock:
            if name not in self.sdk_tools:
                result = {"status": "denied", "reason": f"PermissionDenied: tools.{name} does not exist"}
            else:
                try:
                    schema = self.sdk_tools[name].args_schema
                    if set(args) - set(schema.model_fields):
                        raise ValueError("unsupported argument")
                    clean = schema.model_validate(args).model_dump()
                    reason = self.check_permission(name, clean)
                    result = ({"status": "denied", "reason": reason} if reason
                              else self.sdk_tools[name].invoke(clean))
                except (ValidationError, ValueError, TypeError) as exc:
                    result = {"status": "error", "reason": f"InvalidArguments: {exc}"}
                except Exception as exc:
                    result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
            # Size only: the trace (and UI) must not hold raw log content.
            self.record("runtime_tool", name=name, args=args, status=result.get("status"),
                        tokens=approx_tokens(json.dumps(result, ensure_ascii=False)))
            return result
