"""Child process for run_code: runs model code, proxies tools.* to the harness."""

# Standard library only: this file runs under `python -I` with an empty environment.
import ast
import io
import json
import sys

PROTOCOL_IN, PROTOCOL_OUT = sys.stdin, sys.stdout
FILENAME = "<run_code>"
MAX_CAPTURE = 200_000  # print() beyond this is counted, not kept.


def send(message: dict) -> None:
    PROTOCOL_OUT.write(json.dumps(message, ensure_ascii=True) + "\n")
    PROTOCOL_OUT.flush()


def receive() -> dict:
    line = PROTOCOL_IN.readline()
    if not line:
        raise SystemExit(1)
    return json.loads(line)


class ToolError(Exception):
    pass


class CappedWriter(io.TextIOBase):
    def __init__(self):
        self.parts, self.kept, self.dropped = [], 0, 0

    def writable(self):
        return True

    def write(self, text):
        room = MAX_CAPTURE - self.kept
        self.parts.append(text[:room])
        self.kept += min(len(text), max(room, 0))
        self.dropped += max(len(text) - max(room, 0), 0)
        return len(text)

    def getvalue(self):
        return "".join(self.parts)


class Tools:
    """SDK seen by model code; every call is an RPC the harness checks."""

    def __init__(self):
        self._next_id = 0

    def _call(self, name, args, key):
        self._next_id += 1
        send({"type": "call", "id": self._next_id, "name": name, "args": args})
        reply = receive()
        if reply.get("type") != "result" or reply.get("id") != self._next_id:
            raise RuntimeError("ProtocolError: unexpected reply")
        if not reply["ok"]:
            raise ToolError(reply["error"])
        value = reply["value"]
        if value.get("status") != "ok":
            raise ToolError(value.get("reason") or value.get("status"))
        return value[key]

    def list_files(self, glob="logs/*.log"):
        return self._call("list_files", {"glob": glob}, "files")

    def read_file(self, path):
        return self._call("read_file", {"path": path}, "content")


def compile_body(code: str):
    """Wrap the model's code as the body of __main__(tools), keeping its line numbers."""
    body = ast.parse(code, FILENAME).body or [ast.Pass()]
    wrapper = ast.parse("def __main__(tools):\n    pass\n", FILENAME)
    wrapper.body[0].body = body
    return compile(ast.fix_missing_locations(wrapper), FILENAME, "exec")


def short_error(exc: BaseException) -> str:
    """'<Type>: <message> (line N)' only; tracebacks never reach the model."""
    line = getattr(exc, "lineno", None) if isinstance(exc, SyntaxError) else None
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == FILENAME:
            line = tb.tb_lineno
        tb = tb.tb_next
    text = f"{type(exc).__name__}: {exc.msg if isinstance(exc, SyntaxError) else exc}"
    return text[:240] + (f" (line {line})" if line else "")


def main():
    request = receive()
    captured = CappedWriter()
    sys.stdout = sys.stderr = captured
    sys.stdin = io.StringIO()
    try:
        scope = {"__name__": "__run_code__", "ToolError": ToolError}
        exec(compile_body(request["code"]), scope)
        value = scope["__main__"](Tools())
        message = {"type": "done", "value": json.loads(json.dumps(value, default=str)),
                   "logs": captured.getvalue(), "dropped": captured.dropped}
    except BaseException as exc:  # noqa: BLE001 -- report everything, including SystemExit.
        message = {"type": "error", "message": short_error(exc)}
    send(message)


if __name__ == "__main__":
    main()
