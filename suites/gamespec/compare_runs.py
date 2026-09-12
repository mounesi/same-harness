#!/usr/bin/env python3
"""compare_runs.py — a side-by-side page for two or more exported gamespec runs.

    python3 suites/gamespec/compare_runs.py <run-dir-or-id> <run-dir-or-id> [more...] \\
        [--instance racing-v2] [--out output/runs/_compare]

Each argument is either a path under output/runs/<model>/<run_id> (as export_run.py
prints) or a bare <run_id>, resolved by searching output/runs/*/<run_id>. Requires those
runs to already be exported (run suites/gamespec/export_run.py first, or let
suites/gamespec/demo.sh do it at teardown).

Answers the question export_run.py's per-run page cannot: "how did these N attempts at
the same spec compare?" One page, one column per run, each with:
  - the deliverable, playable in an iframe (the badged, exported game.html)
  - its floor result
  - the run's model, date and key inference/cost numbers, so a difference in outcome can
    be read against a difference in configuration rather than guessed at

If exactly one instance id is common to every run given, the page compares that instance
directly; with more than one in common, pass --instance to pick one (the tool lists what
it found otherwise). Output: output/runs/_compare/<instance>__<run>_vs_<run>[_vs_...]/index.html.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT_DEFAULT = REPO / "output" / "runs"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_run import HTML_HEAD, HTML_TAIL, esc, fmt_dur, pill  # noqa: E402


def resolve(arg: str, out_root: Path) -> Path:
    """A path export_run.py printed, or a bare run_id — find its metadata.json either way."""
    p = Path(arg)
    if (p / "metadata.json").is_file():
        return p.resolve()
    hits = sorted(out_root.glob(f"*/{arg}/metadata.json"))
    if not hits:
        raise SystemExit(f"error: no exported run matches {arg!r} under {out_root} "
                         f"(export it first with suites/gamespec/export_run.py)")
    if len(hits) > 1:
        raise SystemExit(f"error: {arg!r} matches more than one exported run: "
                         + ", ".join(str(h.parent) for h in hits))
    return hits[0].parent


def load(run_dir: Path) -> dict:
    return json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))


def common_instances(metas: list[dict]) -> list[str]:
    sets = [{g["instance_id"] for g in m["games"] if g.get("rebuilt")} for m in metas]
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def game_for(meta: dict, instance_id: str) -> dict | None:
    for g in meta["games"]:
        if g["instance_id"] == instance_id and g.get("rebuilt"):
            return g
    return None


def rel_path(run_dir: Path, out_dir: Path, sub: str) -> str:
    """A relative path from the comparison page back to a file inside one run's folder."""
    import os
    return os.path.relpath(run_dir / sub, out_dir)


def render(instance_id: str, run_dirs: list[Path], metas: list[dict], out_dir: Path) -> str:
    title = f"{instance_id}: " + " vs ".join(m["model"] for m in metas)
    out = [HTML_HEAD.format(title=esc(title)),
          f'<p class="muted"><a href="../index.html">&larr; all runs</a></p>',
          f"<h1>{esc(instance_id)}</h1>",
          f'<p class="muted">{len(metas)} run(s), same spec, side by side.</p>']

    out.append('<div class="cols">')
    for run_dir, meta in zip(run_dirs, metas):
        g = game_for(meta, instance_id)
        fr = (g or {}).get("floor") or {}
        checks = fr.get("checks") or []
        game_href = rel_path(run_dir, out_dir, g["path"]) if g else None
        run_href = rel_path(run_dir, out_dir, "index.html")
        out.append('<div class="col">')
        out.append(f'<h2><a href="{esc(run_href)}">{esc(meta["model"])}</a></h2>')
        if game_href:
            passed_n = sum(1 for c in checks if c.get("ok"))
            status = ("passed" if fr.get("passed") else "failed") + f" {passed_n}/{len(checks)}"
            out.append(f'<iframe src="{esc(game_href)}" title="{esc(meta["model"])}"></iframe>')
            out.append(f'<p>{pill(bool(fr.get("passed")), "floor " + status)}</p>')
        else:
            out.append('<p class="muted">no deliverable for this instance in this run</p>')
        out.append('</div>')
    out.append('</div>')

    out.append('<h2>Configuration</h2><table><tr><th></th>')
    for meta in metas:
        out.append(f'<th>{esc(meta["model"])}</th>')
    out.append('</tr>')

    def row(label: str, get) -> None:
        out.append(f'<tr><th>{esc(label)}</th>' + "".join(f'<td>{esc(get(m))}</td>' for m in metas) + '</tr>')

    row("run id", lambda m: m["run_id"])
    row("started (UTC)", lambda m: m.get("started_at"))
    row("HF repo", lambda m: (m["manifest"].get("model") or {}).get("hf_repo"))
    row("quantization", lambda m: (m["manifest"].get("model") or {}).get("quantization"))
    row("hardware", lambda m: (m["manifest"].get("hardware") or {}).get("instance_type"))
    row("vLLM", lambda m: (m["manifest"].get("runtime") or {}).get("vllm_version"))
    row("temperature", lambda m: (m["manifest"].get("inference") or {}).get("temperature"))
    row("max_iters", lambda m: (m["manifest"].get("inference") or {}).get("max_iters"))
    out.append('</table>')

    out.append('<h2>Attempts</h2><table><tr><th></th>')
    for meta in metas:
        out.append(f'<th>{esc(meta["model"])}</th>')
    out.append('</tr>')

    def attempt_for(meta: dict) -> dict | None:
        for a in meta["attempts"]:
            if a["instance_id"] == instance_id:
                return a
        return None

    row("verdict", lambda m: (lambda a: (a["error_code"] if a else "-") if not (a and a["resolved"]) else "resolved")(attempt_for(m)))
    row("iterations", lambda m: (lambda a: a.get("iterations") if a else "-")(attempt_for(m)))
    row("LLM calls", lambda m: (lambda a: a.get("llm_calls") if a else "-")(attempt_for(m)))
    row("tokens (prompt/completion)", lambda m: (lambda a: (f'{(a.get("tokens") or {}).get("prompt")} / {(a.get("tokens") or {}).get("completion")}') if a else "-")(attempt_for(m)))
    row("wall clock", lambda m: (lambda a: fmt_dur((a.get("wall_clock_ms") or 0) / 1000) if a else "-")(attempt_for(m)))
    row("cost", lambda m: (lambda a: f'${((a.get("cost") or {}).get("usd") or 0):.3f}' if a else "-")(attempt_for(m)))
    out.append('</table>')

    out.append('<h2>Failed floor checks</h2><table><tr><th></th>')
    for meta in metas:
        out.append(f'<th>{esc(meta["model"])}</th>')
    out.append('</tr><tr><th>checks</th>')
    for meta in metas:
        g = game_for(meta, instance_id)
        fr = (g or {}).get("floor") or {}
        failed = [c["name"] for c in (fr.get("checks") or []) if not c.get("ok")]
        out.append(f'<td class="muted">{esc("; ".join(failed)) if failed else ("&mdash;" if g else "no deliverable")}</td>')
    out.append('</tr></table>')

    out.append(HTML_TAIL)
    return "\n".join(out)


def slug(run_id: str) -> str:
    return run_id.replace("/", "_")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="two or more exported run directories or bare run ids")
    ap.add_argument("--instance", help="which spec instance to compare (required if more than one is common)")
    ap.add_argument("--out-root", default=str(OUT_ROOT_DEFAULT), help="the output/runs/ root (default output/runs)")
    args = ap.parse_args()
    if len(args.runs) < 2:
        print("error: give at least two runs to compare", file=sys.stderr)
        return 2

    out_root = Path(args.out_root).resolve()
    run_dirs = [resolve(r, out_root) for r in args.runs]
    metas = [load(d) for d in run_dirs]

    common = common_instances(metas)
    if not common:
        print("error: these runs share no instance with a rebuilt deliverable", file=sys.stderr)
        return 1
    if args.instance:
        if args.instance not in common:
            print(f"error: {args.instance!r} is not common to every run given; common instances: {', '.join(common)}",
                  file=sys.stderr)
            return 1
        instance_id = args.instance
    elif len(common) == 1:
        instance_id = common[0]
    else:
        print(f"error: {len(common)} instances are common to these runs ({', '.join(common)}); "
              "pass --instance to pick one", file=sys.stderr)
        return 1

    out_dir = out_root / "_compare" / f"{instance_id}__{'_vs_'.join(slug(m['run_id']) for m in metas)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(render(instance_id, run_dirs, metas, out_dir), encoding="utf-8")
    print(out_dir / "index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
