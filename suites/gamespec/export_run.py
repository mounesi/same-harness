#!/usr/bin/env python3
"""export_run.py — lay one gamespec run out as a browsable folder under output/runs/.

    python3 suites/gamespec/export_run.py <pulled-run-dir> [--bundle-dir DIR] [--out output/runs]

Why: a run leaves its evidence scattered — a manifest, a results.jsonl, a patch per
attempt, trajectories, a sealed bundle in results/pulled/ — and the question an operator
asks a week later is simpler than that: "which model, when, on what, and what did it
build?" This puts everything for one run in one place, one folder per model, one per run:

    output/runs/<model>/<run_id>/
        README.md            the run on one page: model, date, hardware, harness, verdicts
        metadata.json        the same, machine-readable (plus the full manifest + records)
        game/<instance>/pass-<n>/game.html   the deliverable, rebuilt from its patch, badged with
                                             model / run / date; game.original.html is unmarked
        game/<instance>/pass-<n>/floor-report.json   the floor check, re-run here
        code.zip             game/ + patches/ zipped, for handing around
        run-manifest.json, results.jsonl, patches/, trajectories/, logs/   copied verbatim
        bundle/              the sealed bundle (tarball, manifest, checksum) when given
    output/runs/index.md     one row per exported run, regenerated on every export

output/ is git-ignored (CONTRACTS §7.4 keeps raw results out of git); this is a working
folder on disk, not a publication.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FLOOR_CHECK = REPO / "suites" / "gamespec" / "floor_check.py"


def load_records(run_dir: Path) -> list[dict]:
    p = run_dir / "results.jsonl"
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def rebuild(instance_id: str, diff: Path, out: Path) -> Path | None:
    if out.exists():
        shutil.rmtree(out)
    proc = subprocess.run(
        [sys.executable, "-m", "harness.adapters.gamespec", "rebuild", instance_id, str(diff), str(out)],
        cwd=REPO, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        (out.parent / f"{out.name}.rebuild-error.txt").parent.mkdir(parents=True, exist_ok=True)
        (out.parent / f"{out.name}.rebuild-error.txt").write_text(proc.stderr or proc.stdout, encoding="utf-8")
        return None
    game = Path(proc.stdout.strip().splitlines()[-1])
    return game if game.is_file() else None


def floor(game: Path, instance_id: str) -> dict | None:
    argv = [sys.executable, str(FLOOR_CHECK), str(game), "--json"]
    if instance_id != "racing-v1":
        argv += ["--spec", instance_id]
    proc = subprocess.run(argv, capture_output=True, text=True)
    try:
        return json.loads(proc.stdout)
    except ValueError:
        return {"schema": "gamespec-floor/v1", "passed": False, "checks": [],
                "error": (proc.stderr or proc.stdout)[-400:]}


BADGE = """
<!-- gamespec export badge: added at export time, not part of the graded submission -->
<style>#gamespec-badge{position:fixed;left:8px;bottom:8px;z-index:2147483647;font:12px/1.35 system-ui,sans-serif;
color:#fff;background:rgba(0,0,0,.62);padding:6px 9px;border-radius:6px;pointer-events:none;max-width:60vw;white-space:pre}</style>
<div id="gamespec-badge">built by @model@  ·  @hf_repo@\n@spec@  ·  @started@  ·  floor @floor@\nrun @run_id@</div>
"""


def watermark(html: str, meta: dict, game: dict) -> str:
    """The exported copy carries a badge naming the model, run and date. The graded artifact
    is what the model wrote; this is for whoever plays the export later and asks 'which model
    made this?' (blind human judging uses the unmarked copy beside it)."""
    m = meta["manifest"]
    fr = game.get("floor") or {}
    checks = fr.get("checks") or []
    fields = {
        "model": meta["model"], "hf_repo": (m.get("model") or {}).get("hf_repo") or "",
        "spec": f"{meta.get('suite')} / {game['instance_id']} pass {game['pass'].split('-')[-1]}",
        "started": (meta.get("started_at") or "")[:16].replace("T", " ") + " UTC",
        "floor": ("passed" if fr.get("passed") else "failed") + f" {sum(1 for c in checks if c.get('ok'))}/{len(checks)}",
        "run_id": meta["run_id"],
    }
    badge = BADGE
    for k, v in fields.items():
        badge = badge.replace("@" + k + "@", str(v).replace("<", "&lt;"))
    i = html.lower().rfind("</body>")
    return html[:i] + badge + html[i:] if i >= 0 else html + badge


def fmt_dur(seconds) -> str:
    try:
        s = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "?"
    return f"{s // 60} min {s % 60} s" if s >= 60 else f"{s} s"


def readme(meta: dict) -> str:
    m = meta["manifest"]
    model, hw, rt, hs, inf, pr, tm, fl, su = (m.get(k, {}) for k in
        ("model", "hardware", "runtime", "harness", "inference", "price", "timing", "flags", "suite"))
    rows = [
        ("run id", meta["run_id"]),
        ("model", f"`{model.get('name')}` — {model.get('hf_repo')} @ `{str(model.get('weight_revision', ''))[:12]}`"
                  f" ({model.get('quantization') or 'unquantised'}, {model.get('weight_file_count')} files, "
                  f"{(model.get('weight_bytes') or 0) / 1e9:.1f} GB)"),
        ("suite / spec(s)", f"{su.get('name')}: {', '.join(su.get('instance_ids') or [])}"),
        ("started (UTC)", tm.get("started_at") or m.get("created_at")),
        ("ended (UTC)", f"{tm.get('ended_at')} — {fmt_dur(tm.get('wall_clock_s'))} wall clock"),
        ("hardware", f"{hw.get('gpu_count')}× {hw.get('gpu_model')} (`{hw.get('instance_type')}`), "
                     f"region {hw.get('region') or 'unknown'}, driver {rt.get('nvidia_driver')}"),
        ("price", f"${(pr.get('effective_cents_per_hour') or 0) / 100:.2f}/h ({pr.get('source')})"),
        ("serving", f"vLLM {rt.get('vllm_version')}, TP {rt.get('tensor_parallel_size')}, "
                    f"max_model_len {rt.get('max_model_len')}, `{rt.get('extra_args')}`"),
        ("harness", f"v{hs.get('version')}, template `{hs.get('prompt_template_id')}`, adapter "
                    f"`{hs.get('adapter')}` v{hs.get('adapter_version')}, repo `{hs.get('repo_git_describe')}`"),
        ("inference", f"temperature {inf.get('temperature')}, max_iters {inf.get('max_iters')}, max_tokens "
                      f"{inf.get('max_tokens')}, seed {inf.get('seed')}, passes {inf.get('passes')}, "
                      f"task_timeout {inf.get('task_timeout_s')} s"),
        ("status", f"{m.get('status')}"
                   + ("; **nonconformant**: " + "; ".join(fl.get("nonconformant_reasons") or []) if fl.get("nonconformant") else "")
                   + ("; provenance incomplete: " + "; ".join(fl.get("provenance_incomplete_reasons") or []) if fl.get("provenance_incomplete") else "")),
        ("exported", meta["exported_at"]),
    ]
    out = [f"# {meta['run_id']}", "", "| | |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    out += ["", "## Attempts", "",
            "| instance | pass | verdict | error | iterations | LLM calls | tokens prompt / completion | wall clock | cost |",
            "|---|---|---|---|---|---|---|---|---|"]
    for a in meta["attempts"]:
        t = a.get("tokens") or {}
        out.append(f"| {a['instance_id']} | {a['pass_idx']} | {'**resolved**' if a['resolved'] else 'not resolved'} | "
                   f"{a['error_code']}{(' — ' + a['error_detail']) if a.get('error_detail') else ''} | {a.get('iterations')} | "
                   f"{a.get('llm_calls')} | {t.get('prompt')} / {t.get('completion')} | {fmt_dur((a.get('wall_clock_ms') or 0) / 1000)} | "
                   f"${(a.get('cost') or {}).get('usd', 0) or 0:.3f} |")
    out += ["", "## Floor check (re-run at export time)", ""]
    for g in meta["games"]:
        fr = g.get("floor") or {}
        checks = fr.get("checks") or []
        failed = [c["name"] for c in checks if not c.get("ok")]
        status = "PASSED" if fr.get("passed") else "FAILED"
        out.append(f"- `{g['path']}` — **{status}** ({len(checks) - len(failed)}/{len(checks)})"
                   + (": " + "; ".join(failed[:6]) if failed else "")
                   + (f" — {g['bytes']} bytes" if g.get("bytes") else ""))
    if not meta["games"]:
        out.append("- no deliverable could be rebuilt from the patches")
    out += ["", "## Files", "",
            "- `game/<instance>/pass-<n>/game.html` — the deliverable, rebuilt from `patches/<instance>/pass-<n>.diff` on the adapter's base tree, with a corner badge naming the model, run and date added at export; `game.original.html` is the unmarked file exactly as the model wrote it (use that one for blind judging); `floor-report.json` beside them",
            "- `code.zip` — `game/` and `patches/` zipped",
            "- `run-manifest.json` — full provenance (CONTRACTS.md §2); `results.jsonl` — one raw-result record per attempt (§3)",
            "- `trajectories/`, `logs/` — the model's full tool-call transcript and the harness log, when they were pulled",
            "- `bundle/` — the sealed bundle, its manifest and checksum, when present",
            "- `metadata.json` — everything on this page as data, plus the manifest and records verbatim"]
    return "\n".join(out) + "\n"


def write_index(out_root: Path) -> None:
    rows = []
    for meta_path in sorted(out_root.glob("*/*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        m = meta.get("manifest", {})
        verdicts = ", ".join(f"{a['instance_id']}/{a['pass_idx']}: {'resolved' if a['resolved'] else a['error_code']}"
                             for a in meta.get("attempts", []))
        floors = ", ".join(("PASS" if (g.get("floor") or {}).get("passed") else "FAIL") for g in meta.get("games", [])) or "-"
        rows.append((meta.get("started_at") or "", meta.get("model"), meta["run_id"],
                     m.get("suite", {}).get("name"), m.get("hardware", {}).get("instance_type"),
                     verdicts, floors, meta_path.parent.relative_to(out_root).as_posix()))
    lines = ["# Exported runs", "", "One folder per model, one per run. Newest first.", "",
             "| started (UTC) | model | run id | suite | instance | verdicts | floor | folder |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, reverse=True):
        lines.append(f"| {r[0]} | {r[1]} | `{r[2]}` | {r[3]} | {r[4]} | {r[5]} | {r[6]} | [{r[7]}]({r[7]}/README.md) |")
    (out_root / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a pulled run directory (run-manifest.json, results.jsonl, patches/)")
    ap.add_argument("--bundle-dir", help="directory holding the sealed bundle (results/pulled/<instance>)")
    ap.add_argument("--out", default=str(REPO / "output" / "runs"), help="export root (default output/runs)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "run-manifest.json"
    if not manifest_path.is_file():
        print(f"error: {manifest_path} not found", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest.get("run_id") or run_dir.name
    model = (manifest.get("model") or {}).get("name") or run_id.split("__")[0]
    out_root = Path(args.out).resolve()
    dest = out_root / model / run_id
    dest.mkdir(parents=True, exist_ok=True)

    # verbatim copies
    for name in ("run-manifest.json", "results.jsonl"):
        if (run_dir / name).is_file():
            shutil.copy2(run_dir / name, dest / name)
    for sub in ("patches", "trajectories", "logs"):
        if (run_dir / sub).is_dir():
            if (dest / sub).exists():
                shutil.rmtree(dest / sub)
            shutil.copytree(run_dir / sub, dest / sub)
    if args.bundle_dir:
        bdir = Path(args.bundle_dir)
        hits = [p for p in bdir.glob(f"{run_id}*") if p.is_file()]
        if hits:
            (dest / "bundle").mkdir(exist_ok=True)
            for p in hits:
                shutil.copy2(p, dest / "bundle" / p.name)

    # the deliverable(s)
    games = []
    for diff in sorted((run_dir / "patches").glob("*/pass-*.diff")) if (run_dir / "patches").is_dir() else []:
        iid, pas = diff.parent.name, diff.stem
        game = rebuild(iid, diff, dest / "game" / iid / pas)
        if game is None:
            games.append({"instance_id": iid, "pass": pas, "path": f"game/{iid}/{pas}/", "rebuilt": False})
            continue
        report = floor(game, iid)
        (game.parent / "floor-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        games.append({"instance_id": iid, "pass": pas, "path": f"game/{iid}/{pas}/game.html", "rebuilt": True,
                      "bytes": game.stat().st_size, "floor": report})
        # the base tree's helper files are not the model's work; keep only the deliverable + report
        for extra in ("README.md", "SPEC.md", "floor_check.py", ".gitignore"):
            p = game.parent / extra
            if p.exists():
                p.unlink()
        if (game.parent / ".git").exists():
            shutil.rmtree(game.parent / ".git")

    # badge the playable copies; keep the unmarked original beside each
    meta_for_badge = {"model": model, "run_id": run_id, "manifest": manifest,
                      "suite": (manifest.get("suite") or {}).get("name"),
                      "started_at": (manifest.get("timing") or {}).get("started_at") or manifest.get("created_at")}
    for g in games:
        if not g.get("rebuilt"):
            continue
        game = dest / g["path"]
        original = game.with_name("game.original.html")
        shutil.copy2(game, original)
        game.write_text(watermark(original.read_text(encoding="utf-8"), meta_for_badge, g), encoding="utf-8")

    with zipfile.ZipFile(dest / "code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("game", "patches"):
            for p in sorted((dest / sub).rglob("*")) if (dest / sub).is_dir() else []:
                if p.is_file():
                    z.write(p, p.relative_to(dest).as_posix())

    records = load_records(run_dir)
    attempts = [{
        "instance_id": r.get("instance_id"), "pass_idx": r.get("pass_idx"), "resolved": bool(r.get("resolved")),
        "error_code": r.get("error_code"), "error_detail": r.get("error_detail"), "iterations": r.get("iterations"),
        "llm_calls": r.get("llm_calls"), "tool_calls": r.get("tool_calls"), "tokens": r.get("tokens"),
        "cost": r.get("cost"), "wall_clock_ms": r.get("wall_clock_ms"), "started_at": r.get("started_at"),
        "ended_at": r.get("ended_at"), "grade": r.get("grade"),
    } for r in records]
    meta = {
        "schema": "gamespec-export/v1",
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id, "model": model,
        "started_at": (manifest.get("timing") or {}).get("started_at") or manifest.get("created_at"),
        "ended_at": (manifest.get("timing") or {}).get("ended_at"),
        "suite": (manifest.get("suite") or {}).get("name"),
        "instances": (manifest.get("suite") or {}).get("instance_ids"),
        "attempts": attempts, "games": games,
        "source_run_dir": str(run_dir), "manifest": manifest, "records": records,
    }
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (dest / "README.md").write_text(readme(meta), encoding="utf-8")
    write_index(out_root)
    print(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
