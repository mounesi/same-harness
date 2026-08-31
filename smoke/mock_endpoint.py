"""A fake OpenAI-compatible endpoint, so the harness can be exercised without a GPU.

This exists because every expensive failure mode we found in review was a plumbing bug —
a missing file, a variable exported to the wrong step, an interface that did not line up —
and every one of them would have been caught by running the thing once. A GPU is not
required to catch them; a server that speaks the same protocol is.

It serves the two routes the harness uses:
    GET  /v1/models             -> the served model name (run.sh preflight checks this)
    POST /v1/chat/completions   -> a scripted reply

Scripted behaviour, chosen by $MOCK_MODE:
    solve   the model emits a tool call writing the correct fix, then stops    -> resolved
    noop    the model answers in prose and never edits anything               -> NO_PATCH
    flaky   fails with 503 twice, then behaves like `solve`     -> exercises the retry path

Usage:  python3 smoke/mock_endpoint.py --port 8099 --model smoke-model
Stdlib only, single-threaded-safe, exits cleanly on SIGTERM.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("MOCK_MODEL", "smoke-model")
MODE = os.environ.get("MOCK_MODE", "solve")

_state_lock = threading.Lock()
_state = {"calls": 0, "fail_budget": 2}

# The patch the "solve" mode writes. The synthetic task ships a divide() that raises on a
# zero denominator; the hidden test asserts it returns None instead.
FIX_OLD = "    return a / b"
FIX_NEW = "    if b == 0:\n        return None\n    return a / b"


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _reply(messages: list, tools_seen: bool) -> dict:
    """Return one assistant message, driving a tiny edit-then-finish script."""
    # Has the harness already told us an edit succeeded? Then we are done.
    tool_results = [m for m in messages if m.get("role") == "tool"]
    if MODE == "noop":
        return {"role": "assistant", "content": "I have reviewed the code and see no issue."}

    if not tool_results:
        # First turn: read the file, so the transcript exercises a multi-turn loop.
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("call_1", "read_file", {"path": "calc/ops.py"})],
        }
    if len(tool_results) == 1:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _tool_call(
                    "call_2",
                    "edit_file",
                    {"path": "calc/ops.py", "old_str": FIX_OLD, "new_str": FIX_NEW},
                )
            ],
        }
    return {
        "role": "assistant",
        "content": "Guarded the zero denominator in calc/ops.py; divide() now returns None.",
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the smoke output readable
        if os.environ.get("MOCK_VERBOSE"):
            sys.stderr.write("mock: " + (fmt % args) + "\n")

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        else:
            self._send(404, {"error": {"message": "no route " + self.path}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": "no route " + self.path}})
            return
        length = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(length) or b"{}")

        with _state_lock:
            _state["calls"] += 1
            if MODE == "flaky" and _state["fail_budget"] > 0:
                _state["fail_budget"] -= 1
                self._send(503, {"error": {"message": "mock: transient upstream failure"}})
                return

        msg = _reply(req.get("messages") or [], bool(req.get("tools")))
        finish = "tool_calls" if msg.get("tool_calls") else "stop"
        self._send(
            200,
            {
                "id": "chatcmpl-mock-%d" % _state["calls"],
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.get("model") or MODEL,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": 128,
                    "completion_tokens": 32,
                    "total_tokens": 160,
                },
            },
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    globals()["MODEL"] = args.model
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    sys.stderr.write("mock: serving %s on :%d (mode=%s)\n" % (args.model, args.port, MODE))
    sys.stderr.flush()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
