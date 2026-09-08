"""plots.py — render the video/publication figures for "The Harness Variable" (AI-P153).

analysis/aggregate.py stops at markdown and CSV, which is right for the paper and useless
for a camera. This turns the same `summary.json` into 1920x1080 PNGs. It computes nothing:
every number drawn here is read straight out of the aggregator's report, so a figure cannot
disagree with the table it came from.

    python3 analysis/aggregate.py --manifests run-A/run-manifest.json run-B/run-manifest.json
    python3 analysis/plots.py --summary analysis/tables/summary.json --out-dir analysis/figures

Figures (--only picks a subset):
    cost           cost per resolved task, per model            -- the headline number
    resolve        resolve rate with the min/max range over passes
    taxonomy       100% stacked failure breakdown (CONTRACTS.md 4)
    contamination  Verified -> Pro -> AgentTask slope, per model
    dots           one square per instance, resolved/failed/infra

Three honesty rules are enforced in the drawing code, not left to the narrator:

  1. If the aggregator marked the report `comparability.mixed`, every figure gets a red
     banner saying so. A mixed aggregate must not be able to leave this repo looking like
     a like-for-like comparison just because someone cropped the header off.
  2. A cost drawn from a run flagged `cost_approximate` is suffixed with an asterisk and a
     footnote. Provenance-incomplete runs are included by design (CONTRACTS.md 2.2); the
     figure says the number is approximate rather than quietly rounding it into the bar.
  3. The whiskers on the resolve-rate chart are labelled a RANGE, never a confidence
     interval. Decoding is greedy (temperature 0.0), so repeated passes are not independent
     samples and a CI over them would claim a precision the sampler cannot deliver.

`--demo` renders every figure from obviously synthetic data, watermarked SAMPLE DATA, so
chart design can be settled before a GPU is ever billed.

Requires matplotlib (the only non-stdlib dependency anywhere in this repo, and only here):
    python3 -m pip install matplotlib

Exit codes follow the rest of the repo: 0 ok, 1 usage/no data, 3 I/O.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Single source of truth for the taxonomy and the suite order: importing them from the
# aggregator means a code added there can never be silently dropped from a figure here.
from aggregate import ERROR_CODES, SUITE_ORDER, SUITE_SHORT, is_infra  # noqa: E402

FIGURES = ("cost", "resolve", "taxonomy", "contamination", "dots")

THEMES = {
    "dark": {
        "bg": "#0d1117", "panel": "#161b22", "fg": "#e6edf3", "muted": "#8b949e",
        "grid": "#30363d", "accent": "#58a6ff", "ok": "#3fb950", "ok_dim": "#1f6f36",
        "bad": "#f85149", "warn": "#d29922", "infra": "#6e7681",
        "banner_bg": "#8b1a1a", "banner_fg": "#ffffff",
    },
    "light": {
        "bg": "#ffffff", "panel": "#f6f8fa", "fg": "#1f2328", "muted": "#59636e",
        "grid": "#d1d9e0", "accent": "#0969da", "ok": "#1a7f37", "ok_dim": "#96d8a7",
        "bad": "#cf222e", "warn": "#9a6700", "infra": "#8c959f",
        "banner_bg": "#cf222e", "banner_fg": "#ffffff",
    },
}

# Taxonomy colours, grouped so a viewer reads the FAMILY before the code: green = solved,
# amber = produced nothing to grade, red = graded and wrong, purple = ran out of budget,
# blue = the model itself broke down, yellow = the server, grey = infrastructure.
CODE_COLORS = {
    "OK": "#3fb950",
    "NO_PATCH": "#e3a008", "PATCH_MALFORMED": "#b8860b",
    "TESTS_FAIL": "#f85149", "TESTS_REGRESSION": "#a4232b",
    "BUDGET_ITERATIONS": "#a371f7", "BUDGET_TOKENS": "#8957e5", "BUDGET_WALLCLOCK": "#6e40c9",
    "MODEL_CONTEXT_OVERFLOW": "#58a6ff", "MODEL_MALFORMED_TOOL_CALL": "#388bfd",
    "MODEL_LOOP": "#1f6feb", "MODEL_REFUSAL": "#0d5bd1", "MODEL_EMPTY_RESPONSE": "#0a4aa6",
    "SERVER_ERROR": "#d29922", "SERVER_UNAVAILABLE": "#9e6a03",
    "INFRA_SANDBOX": "#8b949e", "INFRA_GRADER": "#7d8590",
    "INFRA_HOST": "#6e7681", "INFRA_UNKNOWN": "#57606a",
}

MIXED_BANNER = "MIXED-HARNESS AGGREGATE  -  NOT A LIKE-FOR-LIKE COMPARISON"
RANGE_NOTE = ("whiskers = min/max across passes. Decoding is greedy (temperature 0.0), so "
              "passes are not independent samples: this is a range, not a confidence interval.")


class Config:
    """Everything the drawing code needs that is not the report itself."""

    def __init__(self, args):
        self.out_dir = Path(args.out_dir)
        self.c = THEMES[args.theme]
        self.transparent = bool(args.transparent)
        self.dpi = int(args.dpi)
        self.width = int(args.width)
        self.height = int(args.height)
        self.demo = bool(args.demo)
        self.fmt = args.format
        # Point sizes below are tuned for a 1080p frame; scale them so --height 2160 does
        # not render captions that are legible only to the person who exported them.
        self.k = (self.height / float(self.dpi)) / 6.75

    def fs(self, pts: float) -> float:
        return pts * self.k

    def figsize(self):
        return (self.width / float(self.dpi), self.height / float(self.dpi))


# ------------------------------------------------------------------ scaffolding


def new_figure(cfg: Config, title: str, subtitle: str = ""):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=cfg.figsize(), dpi=cfg.dpi)
    if not cfg.transparent:
        fig.patch.set_facecolor(cfg.c["bg"])
    fig.text(0.035, 0.945, title, color=cfg.c["fg"], fontsize=cfg.fs(23),
             fontweight="bold", va="top", ha="left")
    if subtitle:
        fig.text(0.035, 0.895, subtitle, color=cfg.c["muted"], fontsize=cfg.fs(12.5),
                 va="top", ha="left")
    return fig


def style_axes(ax, cfg: Config):
    if not cfg.transparent:
        ax.set_facecolor(cfg.c["bg"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(cfg.c["grid"])
    ax.tick_params(colors=cfg.c["muted"], labelsize=cfg.fs(11.5), length=0)
    ax.grid(axis="x", color=cfg.c["grid"], linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)


def footnote(fig, cfg: Config, text: str):
    if text:
        fig.text(0.035, 0.035, text, color=cfg.c["muted"], fontsize=cfg.fs(10.5),
                 va="bottom", ha="left", wrap=True)


def stamp(fig, cfg: Config, report: dict):
    """The two marks that must survive a crop: mixed-harness, and demo data."""
    if (report.get("comparability") or {}).get("mixed"):
        fig.patches.extend([])
        fig.text(0.5, 0.988, MIXED_BANNER, color=cfg.c["banner_fg"], fontsize=cfg.fs(12),
                 fontweight="bold", ha="center", va="top",
                 bbox=dict(facecolor=cfg.c["banner_bg"], edgecolor="none",
                           boxstyle="square,pad=0.45"))
    if cfg.demo:
        fig.text(0.5, 0.5, "SAMPLE DATA", color=cfg.c["bad"], fontsize=cfg.fs(85),
                 fontweight="bold", ha="center", va="center", rotation=26, alpha=0.13,
                 zorder=1000)


def save(fig, cfg: Config, name: str) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.out_dir / ("%s.%s" % (name, cfg.fmt))
    fig.savefig(str(path), dpi=cfg.dpi, transparent=cfg.transparent,
                facecolor=("none" if cfg.transparent else cfg.c["bg"]))
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def money(x) -> str:
    if x is None:
        return "n/a"
    if x >= 100:
        return "$%.0f" % x
    if x >= 1:
        return "$%.2f" % x
    return "$%.3f" % x


# ------------------------------------------------------------------ figures


def fig_cost(report: dict, cfg: Config):
    """Cost per resolved task, per model. The one number the whole study exists to produce."""
    rows = [r for r in report.get("by_model") or [] if r.get("cost_per_resolved_usd") is not None]
    if not rows:
        return None, "cost: no model has both a cost and a resolved attempt"
    rows.sort(key=lambda r: r["cost_per_resolved_usd"])  # cheapest first == best at the top

    fig = new_figure(cfg, "Cost per resolved task",
                     "same harness, same prompt, same budget, same grader - only the weights differ")
    ax = fig.add_axes([0.26, 0.23, 0.66, 0.58])
    style_axes(ax, cfg)

    ys = list(range(len(rows)))[::-1]
    vals = [r["cost_per_resolved_usd"] for r in rows]
    colors = [cfg.c["ok"] if i == 0 else cfg.c["accent"] for i in range(len(rows))]
    ax.barh(ys, vals, color=colors, height=0.62)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["model"] for r in rows], fontsize=cfg.fs(13.5), color=cfg.c["fg"])
    ax.set_xlabel("USD per resolved task", color=cfg.c["muted"], fontsize=cfg.fs(12),
                  labelpad=cfg.fs(8))
    pad = max(0.0, (4 - len(rows)) / 2.0)
    ax.set_ylim(-0.5 - pad, len(rows) - 0.5 + pad)
    ax.set_xlim(0, max(vals) * 1.22)

    approx = False
    for y, r in zip(ys, rows):
        label = money(r["cost_per_resolved_usd"])
        if r.get("cost_approximate"):
            label += " *"
            approx = True
        ax.text(r["cost_per_resolved_usd"] * 1.02, y, label, va="center", ha="left",
                color=cfg.c["fg"], fontsize=cfg.fs(13), fontweight="bold")

    notes = ["resolved attempts are the denominator; infrastructure failures are excluded "
             "(CONTRACTS.md 4)."]
    if approx:
        notes.append("* cost is approximate: the run is flagged provenance_incomplete "
                     "(included by design, CONTRACTS.md 2.2).")
    if any(r.get("setup_cost_usd") for r in rows):
        notes.append("Setup cost (boot + weight download) is reported separately and is NOT "
                     "amortised into these bars.")
    footnote(fig, cfg, "\n".join(notes))
    stamp(fig, cfg, report)
    return save(fig, cfg, "cost-per-resolved"), None


def fig_resolve(report: dict, cfg: Config):
    """Resolve rate per model x suite, with the honest min/max range over passes."""
    groups = report.get("by_model_suite") or []
    groups = [g for g in groups if g.get("resolve_rate") is not None]
    if not groups:
        return None, "resolve: no group has a resolve rate"

    models = sorted({g["model"] for g in groups})
    suites = [s for s in SUITE_ORDER if any(g["suite"] == s for g in groups)]
    suites += sorted({g["suite"] for g in groups} - set(suites))
    index = {(g["model"], g["suite"]): g for g in groups}

    fig = new_figure(cfg, "Resolution rate", "mean across passes, with the full pass-to-pass range")
    ax = fig.add_axes([0.09, 0.20, 0.86, 0.62])
    style_axes(ax, cfg)
    ax.grid(axis="x", alpha=0)
    ax.grid(axis="y", color=cfg.c["grid"], linewidth=0.8, alpha=0.6)

    span = 0.78
    w = span / max(len(suites), 1)
    palette = [cfg.c["ok"], cfg.c["accent"], cfg.c["warn"], cfg.c["bad"]]
    for si, suite in enumerate(suites):
        xs, ys, lo, hi = [], [], [], []
        for mi, model in enumerate(models):
            g = index.get((model, suite))
            if not g:
                continue
            rate = g["resolve_rate"]
            xs.append(mi - span / 2 + w * (si + 0.5))
            ys.append(rate * 100)
            gmin = g.get("resolve_rate_pass_min")
            gmax = g.get("resolve_rate_pass_max")
            lo.append((rate - gmin) * 100 if gmin is not None else 0.0)
            hi.append((gmax - rate) * 100 if gmax is not None else 0.0)
        if not xs:
            continue
        ax.bar(xs, ys, width=w * 0.86, color=palette[si % len(palette)],
               label=SUITE_SHORT.get(suite, suite),
               yerr=[lo, hi], capsize=cfg.fs(4),
               error_kw=dict(ecolor=cfg.c["fg"], lw=1.4, alpha=0.85))
        for x, y in zip(xs, ys):
            ax.text(x, y + 1.4, "%.0f%%" % y, ha="center", va="bottom",
                    color=cfg.c["fg"], fontsize=cfg.fs(11.5), fontweight="bold")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=cfg.fs(13), color=cfg.c["fg"])
    ax.set_ylabel("% of scored attempts resolved", color=cfg.c["muted"], fontsize=cfg.fs(12))
    ax.set_ylim(0, 100)
    leg = ax.legend(frameon=False, fontsize=cfg.fs(12), loc="upper right", ncol=len(suites))
    for t in leg.get_texts():
        t.set_color(cfg.c["fg"])

    footnote(fig, cfg, RANGE_NOTE)
    stamp(fig, cfg, report)
    return save(fig, cfg, "resolve-rate"), None


def fig_taxonomy(report: dict, cfg: Config):
    """Where the attempts went. The most self-explanatory chart in the set."""
    groups = [g for g in report.get("by_model_suite") or [] if g.get("attempts_total")]
    if not groups:
        return None, "taxonomy: no group has attempts"
    groups.sort(key=lambda g: (SUITE_ORDER.index(g["suite"]) if g["suite"] in SUITE_ORDER else 9,
                               g["model"]))

    present = [c for c in ERROR_CODES if any((g["failure_counts"] or {}).get(c) for g in groups)]
    fig = new_figure(cfg, "Where every attempt ended up",
                     "closed failure taxonomy - CONTRACTS.md 4")
    ax = fig.add_axes([0.26, 0.22, 0.53, 0.60])
    style_axes(ax, cfg)

    ys = list(range(len(groups)))[::-1]
    labels = []
    for y, g in zip(ys, groups):
        total = float(g["attempts_total"])
        left = 0.0
        for code in present:
            n = (g["failure_counts"] or {}).get(code, 0)
            if not n:
                continue
            frac = 100.0 * n / total
            ax.barh(y, frac, left=left, height=0.6,
                    color=CODE_COLORS.get(code, cfg.c["muted"]),
                    hatch=("///" if is_infra(code) else None),
                    edgecolor=(cfg.c["bg"] if not cfg.transparent else "#00000000"),
                    linewidth=0.6)
            if frac >= 6:
                ax.text(left + frac / 2, y, "%.0f%%" % frac, ha="center", va="center",
                        color="#0d1117", fontsize=cfg.fs(10.5), fontweight="bold")
            left += frac
        labels.append("%s  \u00b7  %s  (n=%d)"
                      % (g["model"], SUITE_SHORT.get(g["suite"], g["suite"]),
                         g["attempts_total"]))

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=cfg.fs(11.5), color=cfg.c["fg"])
    ax.set_xlim(0, 100)
    pad = max(0.0, (5 - len(groups)) / 2.0)
    ax.set_ylim(-0.5 - pad, len(groups) - 0.5 + pad)
    ax.set_xlabel("% of all attempts", color=cfg.c["muted"], fontsize=cfg.fs(12))

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(facecolor=CODE_COLORS.get(c, cfg.c["muted"]),
                              hatch=("///" if is_infra(c) else None),
                              label=c.lower().replace("_", " "))
               for c in present]
    leg = fig.legend(handles=handles, frameon=False, fontsize=cfg.fs(10.5),
                     loc="center left", bbox_to_anchor=(0.795, 0.5))
    for t in leg.get_texts():
        t.set_color(cfg.c["fg"])

    footnote(fig, cfg, "Hatched = infrastructure failure: excluded from the resolve-rate "
                       "denominator, because a container that never built is not a model result.")
    stamp(fig, cfg, report)
    return save(fig, cfg, "failure-taxonomy"), None


def fig_contamination(report: dict, cfg: Config):
    """Verified -> Pro -> AgentTask. If a model only shines on the public set, it shows here."""
    rows = []
    for r in report.get("contamination") or []:
        pts = [(s, r.get(k)) for s, k in
               (("swebench-verified", "verified_resolve_rate"),
                ("swebench-pro", "pro_resolve_rate"),
                ("agenttask", "agenttask_resolve_rate"))]
        pts = [(s, v) for s, v in pts if v is not None]
        if len(pts) >= 2:
            rows.append((r, pts))
    if not rows:
        return None, "contamination: fewer than two suites scored for every model"

    fig = new_figure(cfg, "Public benchmark vs. fresh tasks",
                     "the same model, the same harness, three suites of different provenance")
    ax = fig.add_axes([0.11, 0.17, 0.68, 0.65])
    style_axes(ax, cfg)
    ax.grid(axis="x", alpha=0)
    ax.grid(axis="y", color=cfg.c["grid"], linewidth=0.8, alpha=0.5)

    xpos = {s: i for i, s in enumerate(SUITE_ORDER)}
    flagged = False
    tags = []
    for r, pts in rows:
        flag = bool(r.get("contamination_flag"))
        flagged = flagged or flag
        color = cfg.c["bad"] if flag else cfg.c["accent"]
        xs = [xpos[s] for s, _ in pts]
        ys = [v * 100 for _, v in pts]
        ax.plot(xs, ys, marker="o", markersize=cfg.fs(7),
                linewidth=cfg.fs(2.6) if flag else cfg.fs(1.8),
                color=color, alpha=0.95, zorder=3 if flag else 2)
        tags.append({"x": xs[-1], "y": ys[-1],
                     "text": r["model"] + ("  (contamination flag)" if flag else ""),
                     "color": color, "flag": flag})

    # Two models that finish at the same rate would print one label on top of the other.
    # Nudge them apart in y and draw a leader line back to the true endpoint, so the label
    # stays legible without moving the data.
    tags.sort(key=lambda t: t["y"])
    min_sep = 4.6  # percentage points; ~one line of label at this figure size
    for i in range(1, len(tags)):
        if tags[i]["y"] - tags[i - 1]["y"] < min_sep:
            tags[i]["y"] = tags[i - 1]["y"] + min_sep
    for t in tags:
        ax.plot([t["x"] + 0.02, t["x"] + 0.055], [t["y"], t["y"]], color=t["color"],
                linewidth=0.9, alpha=0.55, zorder=2)
        ax.text(t["x"] + 0.07, t["y"], t["text"], va="center", ha="left", color=t["color"],
                fontsize=cfg.fs(12), fontweight="bold" if t["flag"] else "normal", zorder=4)

    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels([SUITE_SHORT[s] for s in SUITE_ORDER], fontsize=cfg.fs(13),
                       color=cfg.c["fg"])
    ax.set_xlim(-0.25, len(SUITE_ORDER) - 0.35)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% resolved", color=cfg.c["muted"], fontsize=cfg.fs(12))

    thr = (report.get("options") or {}).get("contamination_threshold")
    note = ("A model that scores far higher on the public set than on the others is flagged. "
            "The flag is evidence to investigate, not proof of training on the test set.")
    if thr is not None:
        note += "  Threshold: %.0f points of Verified gap vs. the mean of the others." % (thr * 100)
    footnote(fig, cfg, note if flagged else note.replace("is flagged", "would be flagged"))
    stamp(fig, cfg, report)
    return save(fig, cfg, "contamination-slope"), None


def fig_dots(report: dict, cfg: Config, records_by_model: dict):
    """One square per instance. No axis literacy required - the best chart for a thumbnail."""
    if not records_by_model:
        return None, ("dots: no results.jsonl found. Pass --results-root DIR (or run from "
                      "beside the run directories) so the per-instance records can be read.")

    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle

    models = sorted(records_by_model)
    # State per instance, aggregated over passes: solid green only when EVERY pass resolved.
    # "Solved once out of four" is a different claim and gets a different colour.
    states = {}
    for model in models:
        per_instance = {}
        for rec in records_by_model[model]:
            iid = rec.get("instance_id")
            if iid is None:
                continue
            slot = per_instance.setdefault(iid, {"n": 0, "res": 0, "infra": 0})
            slot["n"] += 1
            slot["res"] += 1 if rec.get("resolved") else 0
            slot["infra"] += 1 if is_infra(str(rec.get("error_code") or "")) else 0
        out = []
        for iid in sorted(per_instance):
            s = per_instance[iid]
            if s["res"] == s["n"]:
                out.append((iid, "all"))
            elif s["res"] > 0:
                out.append((iid, "some"))
            elif s["infra"] == s["n"]:
                out.append((iid, "infra"))
            else:
                out.append((iid, "none"))
        states[model] = out

    color_of = {"all": cfg.c["ok"], "some": cfg.c["ok_dim"], "none": cfg.c["bad"],
                "infra": cfg.c["infra"]}

    fig = new_figure(cfg, "Every instance, one square",
                     "green = resolved on every pass, dim = resolved on some, red = not resolved")

    # One slot per model; the caption sits in the top of its own slot, the grid below it.
    left, width = 0.035, 0.93
    top, bottom = 0.855, 0.13
    slot = (top - bottom) / len(models)
    grid_h = slot * 0.70

    # Choose the grid shape from the slot's true pixel aspect, then let matplotlib hold the
    # cells square (set_aspect) rather than stretching them to fill the box. A "dot grid" of
    # wide rectangles reads as a bar chart and defeats the point of the figure.
    n = max(len(v) for v in states.values())
    aspect = (width * cfg.width) / float(grid_h * cfg.height)
    nrows = max(1, int(round(math.sqrt(n / aspect))))
    ncols = max(1, int(math.ceil(n / float(nrows))))
    nrows = max(1, int(math.ceil(n / float(ncols))))

    for mi, model in enumerate(models):
        slot_top = top - slot * mi
        resolved = sum(1 for _, st in states[model] if st == "all")
        fig.text(left, slot_top - slot * 0.06,
                 "%s   %d/%d resolved on every pass" % (model, resolved, len(states[model])),
                 color=cfg.c["fg"], fontsize=cfg.fs(13.5), fontweight="bold",
                 va="top", ha="left")
        ax = fig.add_axes([left, slot_top - slot * 0.18 - grid_h, width, grid_h])
        ax.set_xlim(0, ncols)
        ax.set_ylim(0, nrows)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_anchor("NW")
        ax.axis("off")
        for i, (_iid, st) in enumerate(states[model]):
            r, c = divmod(i, ncols)
            ax.add_patch(Rectangle((c + 0.08, r + 0.08), 0.84, 0.84,
                                   facecolor=color_of[st], edgecolor="none"))

    handles = [mpatches.Patch(facecolor=color_of[k], label=v) for k, v in
               (("all", "resolved, every pass"), ("some", "resolved, some passes"),
                ("none", "not resolved"), ("infra", "infrastructure failure (excluded)"))]
    leg = fig.legend(handles=handles, frameon=False, fontsize=cfg.fs(11),
                     loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4)
    for t in leg.get_texts():
        t.set_color(cfg.c["muted"])
    stamp(fig, cfg, report)
    return save(fig, cfg, "instance-dots"), None


# ------------------------------------------------------------------ inputs


def load_records(report: dict, extra_roots) -> dict:
    """Find each run's results.jsonl and group its records by model.

    The report names its input manifests, so the run directory is usually the manifest's
    own parent. --results-root covers the case where bundles were unpacked elsewhere.
    """
    runs_by_id = {r["run_id"]: r for r in (report.get("runs") or [])}
    candidates = []
    for m in (report.get("inputs") or {}).get("manifests") or []:
        candidates.append(Path(m).resolve().parent)
    for root in extra_roots:
        root = Path(root)
        if root.is_dir():
            candidates.append(root)
            candidates.extend(p for p in sorted(root.iterdir()) if p.is_dir())

    out = {}
    seen = set()
    for d in candidates:
        man, res = d / "run-manifest.json", d / "results.jsonl"
        if not (man.is_file() and res.is_file()):
            continue
        try:
            manifest = json.loads(man.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        run_id = manifest.get("run_id")
        # Only draw runs the aggregator actually included; an excluded run must not sneak
        # back into a figure through the back door.
        if run_id in seen or run_id not in runs_by_id:
            continue
        seen.add(run_id)
        model = runs_by_id[run_id].get("model") or (manifest.get("model") or {}).get("name")
        recs = out.setdefault(model, [])
        try:
            for line in res.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        except (OSError, ValueError) as exc:
            print("==> warning: unreadable %s: %s" % (res, exc), file=sys.stderr)
    return {k: v for k, v in out.items() if v}


def demo_report() -> dict:
    """Synthetic data in the real report's shape, for designing figures before spending."""
    spec = [
        ("model-A", {"swebench-verified": 0.52, "swebench-pro": 0.24, "agenttask": 0.31},
         1.94, False),
        ("model-B", {"swebench-verified": 0.61, "swebench-pro": 0.28, "agenttask": 0.30},
         14.80, False),
        ("model-C", {"swebench-verified": 0.44, "swebench-pro": 0.39, "agenttask": 0.40},
         3.15, True),
    ]
    mixes = {
        "model-A": {"NO_PATCH": 0.09, "TESTS_FAIL": 0.27, "BUDGET_ITERATIONS": 0.06,
                    "MODEL_LOOP": 0.03, "INFRA_SANDBOX": 0.02},
        "model-B": {"NO_PATCH": 0.05, "TESTS_FAIL": 0.24, "PATCH_MALFORMED": 0.03,
                    "MODEL_CONTEXT_OVERFLOW": 0.04, "INFRA_GRADER": 0.01},
        "model-C": {"NO_PATCH": 0.14, "TESTS_FAIL": 0.22, "TESTS_REGRESSION": 0.05,
                    "SERVER_UNAVAILABLE": 0.02, "INFRA_UNKNOWN": 0.02},
    }
    groups, by_model = [], []
    for model, rates, cpr, approx in spec:
        for suite, rate in rates.items():
            total = 100
            counts = {c: 0 for c in ERROR_CODES}
            counts["OK"] = int(round(rate * total))
            rest = total - counts["OK"]
            for code, share in mixes[model].items():
                counts[code] = min(rest, int(round(share * total)))
                rest -= counts[code]
            counts["TESTS_FAIL"] += max(0, rest)
            infra = sum(counts[c] for c in ERROR_CODES if is_infra(c))
            groups.append({
                "model": model, "suite": suite, "attempts_total": total,
                "attempts_scored": total - infra, "resolved_attempts": counts["OK"],
                "resolve_rate": rate, "resolve_rate_pass_min": max(0.0, rate - 0.03),
                "resolve_rate_pass_max": min(1.0, rate + 0.025),
                "failure_counts": counts, "cost_approximate": approx,
                "setup_cost_usd": 6.0,
            })
        by_model.append({"model": model, "cost_per_resolved_usd": cpr,
                         "cost_approximate": approx, "setup_cost_usd": 6.0})
    contamination = []
    for model, rates, _cpr, _a in spec:
        others = [rates["swebench-pro"], rates["agenttask"]]
        gap = rates["swebench-verified"] - sum(others) / len(others)
        contamination.append({
            "model": model, "verified_resolve_rate": rates["swebench-verified"],
            "pro_resolve_rate": rates["swebench-pro"],
            "agenttask_resolve_rate": rates["agenttask"],
            "contamination_flag": gap >= 0.10, "complete_triple": True,
        })
    return {"comparability": {"mixed": False}, "by_model_suite": groups, "by_model": by_model,
            "contamination": contamination, "runs": [], "inputs": {"manifests": []},
            "options": {"contamination_threshold": 0.10}}


def demo_records(report: dict) -> dict:
    """Per-instance records consistent with the demo report's Verified resolve rates."""
    out = {}
    for g in report["by_model_suite"]:
        if g["suite"] != "swebench-verified":
            continue
        recs = []
        n_ok = g["failure_counts"]["OK"]
        n_infra = sum(g["failure_counts"][c] for c in ERROR_CODES if is_infra(c))
        for i in range(g["attempts_total"]):
            if i < n_ok:
                code, res = "OK", True
            elif i < n_ok + n_infra:
                code, res = "INFRA_SANDBOX", False
            else:
                code, res = "TESTS_FAIL", False
            recs.append({"instance_id": "demo__inst-%03d" % i, "pass_idx": 0,
                         "resolved": res, "error_code": code})
        out[g["model"]] = recs
    return out


# ------------------------------------------------------------------ cli


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # pragma: no cover - argparse plumbing
        self.print_usage(sys.stderr)
        self.exit(1, "%s: error: %s\n" % (self.prog, message))


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="plots.py",
                description="Render the figures for 'The Harness Variable' from an "
                            "aggregate.py summary.json.")
    p.add_argument("--summary", default="analysis/tables/summary.json", metavar="PATH",
                   help="report written by aggregate.py (default: %(default)s)")
    p.add_argument("--out-dir", default="analysis/figures", metavar="DIR",
                   help="where the images land (default: %(default)s)")
    p.add_argument("--results-root", nargs="+", default=[], metavar="DIR",
                   help="extra directories holding unpacked run bundles, for the dot grid")
    p.add_argument("--only", nargs="+", choices=FIGURES, metavar="NAME",
                   help="render only these figures (%s)" % ", ".join(FIGURES))
    p.add_argument("--theme", choices=sorted(THEMES), default="dark")
    p.add_argument("--transparent", action="store_true",
                   help="no background fill, for compositing over video footage")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--format", default="png", choices=("png", "svg", "pdf"))
    p.add_argument("--demo", action="store_true",
                   help="render synthetic watermarked figures; --summary is not read")
    return p


def main(argv) -> int:
    args = build_parser().parse_args(argv)
    try:
        import matplotlib
    except ImportError:
        print("error: matplotlib is required for plots.py\n"
              "  fix: python3 -m pip install matplotlib", file=sys.stderr)
        return 1
    matplotlib.use("Agg")

    if args.demo:
        report = demo_report()
        records = demo_records(report)
    else:
        path = Path(args.summary)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            print("error: cannot read %s: %s\n"
                  "  fix: run analysis/aggregate.py first, or pass --summary PATH "
                  "(or --demo to design figures without data)" % (path, exc), file=sys.stderr)
            return 3
        except ValueError as exc:
            print("error: %s is not valid JSON: %s" % (path, exc), file=sys.stderr)
            return 3
        records = load_records(report, args.results_root)

    cfg = Config(args)
    wanted = list(args.only) if args.only else list(FIGURES)
    builders = {
        "cost": lambda: fig_cost(report, cfg),
        "resolve": lambda: fig_resolve(report, cfg),
        "taxonomy": lambda: fig_taxonomy(report, cfg),
        "contamination": lambda: fig_contamination(report, cfg),
        "dots": lambda: fig_dots(report, cfg, records),
    }

    written, skipped = [], []
    for name in wanted:
        try:
            path, why = builders[name]()
        except Exception as exc:  # a bad figure must not lose the good ones
            skipped.append("%s: %s" % (name, exc))
            continue
        if path is None:
            skipped.append(why or ("%s: nothing to draw" % name))
        else:
            written.append(path)

    for s in skipped:
        print("==> skipped %s" % s, file=sys.stderr)
    for w in written:
        print(w)
    if not written:
        print("error: no figure could be rendered from %s"
              % ("demo data" if args.demo else args.summary), file=sys.stderr)
        return 1
    if (report.get("comparability") or {}).get("mixed"):
        print("==> MIXED-HARNESS AGGREGATE: every figure is banner-stamped; do not publish "
              "as a like-for-like comparison", file=sys.stderr)
    print("==> wrote %d figure(s) to %s" % (len(written), cfg.out_dir), file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("==> interrupted", file=sys.stderr)
        sys.exit(130)
