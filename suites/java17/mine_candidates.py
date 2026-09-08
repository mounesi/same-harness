#!/usr/bin/env python3
"""mine_candidates.py — find real Java 8->17 migrations that could become suite tasks.

The premise of the java17 suite (see README.md) is that we do not AUTHOR correctness
tests, we HARVEST them: a merged migration PR carries its own oracle, because the
project's own test suite had to go green under JDK 17 for the PR to land. This script
finds those PRs and filters them down to the ones that could actually become tasks.

    # candidates merged after every evaluated model's training cutoff
    python3 suites/java17/mine_candidates.py --merged-after 2026-01-01 --out /tmp/cand.json

    # widen the net, accept more manual triage
    python3 suites/java17/mine_candidates.py --merged-after 2025-06-01 --limit 200 --min-stars 0

Nothing here decides what enters the suite. It produces a RANKED CANDIDATE LIST for a
human to triage, and it records why each candidate was kept or dropped, because
"which repos did we consider" is part of the methodology and belongs in the audit trail
just as much as the final id list does.

CONTAMINATION IS THE POINT OF --merged-after. If a migration landed before a model's
training cutoff, that model may have memorised the diff, and the task measures recall
rather than capability. The default is deliberately recent. Widening it is a
methodological decision, not a convenience, so the chosen value is written into the
output and belongs in the run's provenance.

Exit codes: 0 ok, 2 usage/config, 3 API/network failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API = "https://api.github.com"
UA = "nvidia-lambda-java17-miner"

# Query set, not a single query. The first cut of this script asked for "java 17" AND
# migrate in the title and found 63 PRs since 2026-01-01 — which reads as "there is not
# enough material for a suite" and is wrong by two orders of magnitude. Real migrations
# are usually named after their VISIBLE consequence (the jakarta namespace flip, the
# Spring Boot 3 bump) rather than after the JDK version, so the narrow query discarded
# ~97% of them. Measured pool sizes, merged:>=2026-01-01, language:Java:
#
#     "java 17" + migrate in:title      63
#     "jdk 17" OR "java 17" in:title  2308
#     jakarta in:title                2815
#     "spring boot 3" in:title        6172
#
# Breadth here costs only triage time; narrowness silently kills the suite.
QUERIES = (
    '"jdk 17" OR "java 17" in:title',
    'jakarta in:title',
    '"spring boot 3" in:title',
)

# Signals that a PR is a genuine runtime migration rather than a passing mention of
# Java 17 in a changelog or a dependabot bump.
TITLE_MUST_MATCH = ("java 17", "jdk 17", "java17", "jdk17", "jakarta", "spring boot 3")
TITLE_HINTS = ("migrat", "upgrad", "bump", "moderniz", "modernis", "port", "to ")

# A repo with no build file cannot be graded by running its tests.
BUILD_FILES = ("pom.xml", "build.gradle", "build.gradle.kts")

# Things that make a repo a poor benchmark task regardless of the PR quality.
NAME_ANTIPATTERNS = ("tutorial", "example", "demo", "sample", "exercise", "playground",
                     "learning", "study", "practice", "test-repo", "hello")


def _get(path: str, token: str | None, params: dict | None = None) -> Any:
    url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": UA,
        **({"Authorization": "Bearer " + token} if token else {}),
    })
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as fh:
                return json.load(fh)
        except urllib.error.HTTPError as exc:
            # 403 here is nearly always the unauthenticated search rate limit (10/min).
            # Backing off is worth it: the alternative is a half-mined list that looks
            # like a real answer.
            if exc.code in (403, 429) and attempt < 3:
                wait = 20 * (attempt + 1)
                print("  rate limited, waiting %ds" % wait, file=sys.stderr)
                time.sleep(wait)
                continue
            if exc.code == 404:
                return None
            raise
        except urllib.error.URLError:
            if attempt < 3:
                time.sleep(5)
                continue
            raise
    return None


def looks_like_migration(title: str) -> bool:
    low = title.lower()
    return any(m in low for m in TITLE_MUST_MATCH) and any(h in low for h in TITLE_HINTS)


def repo_is_plausible(repo: dict, min_stars: int) -> tuple[bool, str]:
    name = (repo.get("full_name") or "").lower()
    if repo.get("archived"):
        return False, "archived"
    if repo.get("fork"):
        return False, "fork"
    if (repo.get("stargazers_count") or 0) < min_stars:
        return False, "stars<%d" % min_stars
    if any(bad in name for bad in NAME_ANTIPATTERNS):
        return False, "name looks like a teaching repo"
    if (repo.get("size") or 0) < 200:  # KB; a real service is not 200 KB
        return False, "too small to be a real codebase"
    return True, "ok"


def has_build_and_tests(full_name: str, sha: str, token: str | None) -> tuple[bool, str]:
    """A task needs a build file to run and a test suite to grade against."""
    tree = _get("/repos/%s/git/trees/%s" % (full_name, sha), token, {"recursive": "1"})
    if not tree or "tree" not in tree:
        return False, "tree unavailable"
    paths = [t["path"] for t in tree["tree"] if t.get("type") == "blob"]
    build = [p for p in paths if p.split("/")[-1] in BUILD_FILES]
    if not build:
        return False, "no maven/gradle build file"
    tests = [p for p in paths if "src/test/java" in p and p.endswith(".java")]
    if len(tests) < 5:
        return False, "only %d test files — too thin to grade" % len(tests)
    return True, "%d test files, %d build file(s)" % (len(tests), len(build))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged-after", default="2026-01-01",
                    help="only PRs merged on/after this date (YYYY-MM-DD). CONTAMINATION "
                         "GUARD: must post-date the training cutoff of every model you "
                         "intend to evaluate. Default 2026-01-01.")
    ap.add_argument("--limit", type=int, default=60, help="PRs to examine (default 60)")
    ap.add_argument("--min-stars", type=int, default=20,
                    help="minimum repo stars (default 20; 0 to disable)")
    ap.add_argument("--out", help="write the candidate list here as JSON")
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"),
                    help="GitHub token ($GITHUB_TOKEN). Without one the search API "
                         "allows only ~10 requests/minute and this will be slow.")
    args = ap.parse_args()

    if not args.token:
        print("note: no --token/$GITHUB_TOKEN — unauthenticated search is rate limited "
              "to ~10 req/min; this will be slow but will work.", file=sys.stderr)

    queries = ["%s type:pr is:merged language:Java merged:>=%s" % (q, args.merged_after)
               for q in QUERIES]
    for q in queries:
        print("query: %s" % q, file=sys.stderr)

    kept: list[dict] = []
    rejected: list[dict] = []
    seen_repos: set[str] = set()
    examined = 0

    for query in queries:
        page = 1
        if examined >= args.limit:
            break
        while examined < args.limit:
            res = _get("/search/issues", args.token,
                       {"q": query, "sort": "created", "order": "desc",
                        "per_page": 30, "page": page})
            items = (res or {}).get("items") or []
            if not items:
                break
            for it in items:
                if examined >= args.limit:
                    break
                examined += 1
                title = it.get("title") or ""
                full_name = "/".join(it["repository_url"].split("/")[-2:])

                if not looks_like_migration(title):
                    rejected.append({"repo": full_name, "pr": it["number"], "title": title,
                                     "why": "title does not read as a migration"})
                    continue
                # One task per repo: several migration PRs in one codebase would share a
                # lineage and correlate, which inflates apparent sample size.
                if full_name in seen_repos:
                    rejected.append({"repo": full_name, "pr": it["number"], "title": title,
                                     "why": "another PR from this repo already kept"})
                    continue

                repo = _get("/repos/%s" % full_name, args.token)
                if not repo:
                    rejected.append({"repo": full_name, "pr": it["number"], "title": title,
                                     "why": "repo unavailable"})
                    continue
                ok, why = repo_is_plausible(repo, args.min_stars)
                if not ok:
                    rejected.append({"repo": full_name, "pr": it["number"], "title": title,
                                     "why": why})
                    continue

                pr = _get("/repos/%s/pulls/%d" % (full_name, it["number"]), args.token)
                if not pr or not pr.get("merge_commit_sha"):
                    rejected.append({"repo": full_name, "pr": it["number"], "title": title,
                                     "why": "no merge commit"})
                    continue

                base_sha = (pr.get("base") or {}).get("sha")
                ok, why = has_build_and_tests(full_name, base_sha, args.token)
                if not ok:
                    rejected.append({"repo": full_name, "pr": it["number"], "title": title,
                                     "why": why})
                    continue

                seen_repos.add(full_name)
                kept.append({
                    "repo": full_name,
                    "pr": it["number"],
                    "title": title,
                    "merged_at": pr.get("merged_at"),
                    "base_sha": base_sha,             # pre-migration: the task snapshot
                    "merge_sha": pr["merge_commit_sha"],  # post-migration: the hidden tests
                    "changed_files": pr.get("changed_files"),
                    "additions": pr.get("additions"),
                    "deletions": pr.get("deletions"),
                    "stars": repo.get("stargazers_count"),
                    "size_kb": repo.get("size"),
                    "license": ((repo.get("license") or {}).get("spdx_id")),
                    "evidence": why,
                })
                print("  KEEP %-45s PR#%-6s %s" % (full_name[:45], it["number"], why),
                      file=sys.stderr)
            page += 1

    # Big diffs do not fit a 40-iteration budget; tiny ones are not a migration.
    kept.sort(key=lambda c: abs((c.get("changed_files") or 999) - 12))

    out = {
        "schema": "java17-candidates/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": query,
        "contamination_guard": {
            "merged_after": args.merged_after,
            "rationale": "tasks must post-date the training cutoff of every evaluated "
                         "model, or the suite measures recall of a public diff rather "
                         "than migration capability",
        },
        "filters": {"min_stars": args.min_stars, "limit": args.limit},
        "kept": kept,
        "rejected": rejected,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print("\nwrote %s — %d kept, %d rejected, %d examined"
              % (args.out, len(kept), len(rejected), examined), file=sys.stderr)
    else:
        json.dump(out, sys.stdout, indent=2, sort_keys=True)
        print()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as exc:
        print("error: GitHub API %s %s" % (exc.code, exc.reason), file=sys.stderr)
        sys.exit(3)
    except KeyboardInterrupt:
        sys.exit(130)
