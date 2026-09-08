#!/usr/bin/env python3
"""_junit.py — read JUnit/Surefire XML reports into per-test-method outcomes.

Shared base module for JVM suites, not an adapter: it is deliberately absent from
`ADAPTERS`, in the same way `_swebench.py` is.

Why this exists rather than parsing build output. Maven Surefire and Gradle both write
structured per-test XML (`target/surefire-reports/TEST-*.xml`,
`build/test-results/test/TEST-*.xml`), which is a strictly better oracle than the stdout
scraping the Python suites are forced into:

  * a test's outcome is an ELEMENT, not a line of prose that a logging change can reword
  * the file exists even when the build's exit code is a lie — Maven exits non-zero on the
    first module failure, so `mvn test`'s return code cannot distinguish "one test failed"
    from "the compiler died", while the reports tell you exactly which methods ran
  * per-method timings and failure messages come along for free

That last point is what makes migration grading tractable at all: a JDK 17 migration that
half-works produces a partial report, and a partial report is a diagnosis. An exit code is
not.

Node id format is `com.example.FooTest#testBar`, matching the `fail_to_pass` /
`pass_to_pass` convention and the form Maven itself accepts for `-Dtest=`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

#: Where Maven and Gradle put their reports, relative to a module or project root.
REPORT_GLOBS = (
    "**/target/surefire-reports/TEST-*.xml",   # maven surefire (unit)
    "**/target/failsafe-reports/TEST-*.xml",   # maven failsafe (integration)
    "**/build/test-results/**/TEST-*.xml",     # gradle
)

#: Outcomes, in the vocabulary the harness already uses for pytest.
PASSED, FAILED, ERROR, SKIPPED = "passed", "failed", "error", "skipped"

_ILLEGAL_XML = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f]"  # control chars a stack trace can embed
)


def node_id(classname: str, name: str) -> str:
    """`com.example.FooTest#testBar` — the id form used in fail_to_pass/pass_to_pass."""
    return "%s#%s" % (classname, name)


def parse_report(path: Path) -> dict[str, str]:
    """One report file -> {node_id: outcome}.

    A malformed report yields {} rather than raising. Surefire writes these files while
    the JVM is still running and a killed fork can leave a truncated one; treating that
    as a hard error would turn "the build was slow" into "the model failed", which is
    exactly the confusion INFRA_GRADER exists to prevent. The caller decides what a
    missing outcome means — it has the planned node ids and this module does not.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    # Stack traces in <failure> bodies routinely contain control characters that are
    # illegal in XML 1.0; Surefire does not escape them and ElementTree refuses the file.
    raw = _ILLEGAL_XML.sub("", raw)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return {}

    out: dict[str, str] = {}
    # A file is either a <testsuite> or a <testsuites> wrapping several.
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    for suite in suites:
        for case in suite.findall("testcase"):
            cls = case.get("classname") or suite.get("name") or ""
            nm = case.get("name") or ""
            if not nm:
                continue
            if case.find("error") is not None:
                outcome = ERROR
            elif case.find("failure") is not None:
                outcome = FAILED
            elif case.find("skipped") is not None:
                outcome = SKIPPED
            else:
                outcome = PASSED
            out[node_id(cls, nm)] = outcome
    return out


def collect(root: Path) -> dict[str, str]:
    """Every report under `root` -> {node_id: outcome}.

    Later files win on collision, which matters for a re-run: Surefire overwrites a
    module's report in place, but a multi-module build can legitimately produce the same
    class name in two modules. That collision is the caller's problem to avoid by scoping
    the task to one module — see the java17 suite's module decomposition.
    """
    merged: dict[str, str] = {}
    for pattern in REPORT_GLOBS:
        for path in sorted(root.glob(pattern)):
            merged.update(parse_report(path))
    return merged


def summarize(outcomes: dict[str, str], planned: list[str]) -> dict:
    """Score `planned` node ids against observed `outcomes`.

    `missing` is kept distinct from `failed` on purpose. A test that did not run is not
    a test that failed: on a migration task the usual cause is that the module never
    compiled, which is a legitimate model failure — but it is also what a grader crash
    looks like, and the two must stay separable in the record.
    """
    passed = [n for n in planned if outcomes.get(n) == PASSED]
    failed = [n for n in planned if outcomes.get(n) in (FAILED, ERROR)]
    skipped = [n for n in planned if outcomes.get(n) == SKIPPED]
    missing = [n for n in planned if n not in outcomes]
    return {
        "passed": len(passed),
        "total": len(planned),
        "failed_ids": failed[:20],
        "skipped_ids": skipped[:20],
        "missing_ids": missing[:20],
        "all_ran": not missing,
    }


# --- self-test (CONTRACTS-style: the module proves itself, no test framework) ----------
#
# Run: python3 -m harness.adapters._junit --self-test
# Wired into ci.yml. Every case here is one this parser met in the wild or will: Surefire
# writes reports while forks are still running, and stack traces carry bytes that are
# illegal in XML 1.0. A parser that raises on those turns an infrastructure hiccup into
# what looks like a model failure.

_SELFTEST_NORMAL = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="com.example.FooTest" tests="4">
  <testcase classname="com.example.FooTest" name="testPasses" time="0.01"/>
  <testcase classname="com.example.FooTest" name="testFails"><failure message="x">t</failure></testcase>
  <testcase classname="com.example.FooTest" name="testErrors"><error message="y">t</error></testcase>
  <testcase classname="com.example.FooTest" name="testSkipped"><skipped/></testcase>
</testsuite>
"""

_SELFTEST_WRAPPED = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="com.example.A"><testcase classname="com.example.A" name="t1"/></testsuite>
  <testsuite name="com.example.B"><testcase classname="com.example.B" name="t2"/></testsuite>
</testsuites>
"""

_SELFTEST_CTRL = ('<?xml version="1.0"?><testsuite name="c.Bar">'
                  '<testcase classname="c.Bar" name="t"><failure message="m">a\x07b</failure>'
                  '</testcase></testsuite>')

_SELFTEST_TRUNCATED = '<?xml version="1.0"?><testsuite name="c.T"><testcase classname="c.T'


def _self_test() -> int:
    import tempfile
    fails = []

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    root = Path(tempfile.mkdtemp(prefix="junit-selftest-"))
    rep = root / "target" / "surefire-reports"
    rep.mkdir(parents=True)
    (rep / "TEST-normal.xml").write_text(_SELFTEST_NORMAL, encoding="utf-8")
    (rep / "TEST-wrapped.xml").write_text(_SELFTEST_WRAPPED, encoding="utf-8")
    (rep / "TEST-ctrl.xml").write_text(_SELFTEST_CTRL, encoding="utf-8")
    (rep / "TEST-trunc.xml").write_text(_SELFTEST_TRUNCATED, encoding="utf-8")

    got = collect(root)
    check(got.get("com.example.FooTest#testPasses") == PASSED, "passing case -> passed")
    check(got.get("com.example.FooTest#testFails") == FAILED, "<failure> -> failed")
    check(got.get("com.example.FooTest#testErrors") == ERROR, "<error> -> error (kept distinct)")
    check(got.get("com.example.FooTest#testSkipped") == SKIPPED, "<skipped> -> skipped")
    check(got.get("com.example.A#t1") == PASSED and got.get("com.example.B#t2") == PASSED,
          "<testsuites> wrapper is walked, not just bare <testsuite>")
    check(got.get("c.Bar#t") == FAILED,
          "illegal XML control byte in a stack trace does not lose the case")
    check(not any(k.startswith("c.T#") for k in got),
          "truncated report yields nothing instead of raising")

    s = summarize(got, ["com.example.FooTest#testPasses", "com.example.FooTest#testFails",
                        "com.example.Gone#never"])
    check(s["passed"] == 1 and s["total"] == 3, "summarize counts against PLANNED ids")
    check(s["missing_ids"] == ["com.example.Gone#never"] and not s["all_ran"],
          "a test that never ran is 'missing', never silently 'failed'")

    print("\n%s" % ("SELF-TEST PASSED" if not fails else "SELF-TEST FAILED: %d" % len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    import sys as _sys
    if "--self-test" in _sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
