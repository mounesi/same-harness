#!/usr/bin/env bash
# serve.sh — one command to browse every exported gamespec run.
#
#   ./suites/gamespec/serve.sh [port]      default port 8787
#
# Starts a plain static server on output/runs/ and prints the URL to open:
# output/runs/index.html lists every run, grouped by model; each links to that run's own
# page (config, attempts, floor result, a Play link into the badged game.html); Ctrl-C
# stops the server. Nothing here talks to Lambda — this only serves files already on disk
# from suites/gamespec/export_run.py (run by suites/gamespec/demo.sh at teardown, or by
# hand against a pulled run directory).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${1:-8787}"
OUT="$HERE/output/runs"
[[ -f "$OUT/index.html" ]] || {
  echo "no exported runs yet at $OUT — run suites/gamespec/demo.sh, or" >&2
  echo "  python3 suites/gamespec/export_run.py <pulled-run-dir>" >&2
  exit 1
}
echo "==> http://localhost:$PORT/index.html  (Ctrl-C to stop)" >&2
exec python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$OUT"
