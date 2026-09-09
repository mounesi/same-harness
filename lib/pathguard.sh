#!/usr/bin/env bash
# pathguard.sh — one guard for the one bug this project keeps paying for.
#
# Four separate live-GPU failures, one root cause: the program was INSTALLED AND PRESENT,
# and invisible only because its directory was not on PATH for non-interactive execution.
# `ssh host "cmd"` does not source ~/.profile, so the ~/.local/bin and venv bin dirs an
# interactive login picks up silently are simply absent — which is why every one of these
# was found on a rented GPU and never on a laptop:
#
#   AI-3155  the agenttask grader shelled out to `python`; preflight probed sys.executable.
#            Preflight passed, then every attempt died rc=127 at GRADE time — after the
#            agent loop had already run.
#   AI-3157  the `hf` CLI: in ~/.local/bin, off PATH, and renamed between hub versions
#            (huggingface-cli -> hf). modelctl died "hf: command not found" after
#            provisioning.
#   AI-3157  the `vllm` binary itself: "nohup: failed to run command 'vllm'", with
#            /home/ubuntu/.local/bin/vllm present the whole time.
#   AI-3159  `ninja`: vLLM's FlashInfer JIT-compiles kernels during ENGINE STARTUP and
#            execs `ninja`. It was in the venv, installed by pip as a vLLM dependency. It
#            failed AFTER the weights were resolved — the most expensive possible moment.
#
# Each was fixed as a one-off point fix, and a fifth is likely, because the pattern is not
# in any one program: it is that we hand a program an absolute path and then assume
# everything IT reaches for by name is visible too. So, two functions:
#
#   pathguard_adopt <bin-dir|program-path>
#       A console script sets sys.executable and NOTHING else: running $VENV/bin/vllm does
#       not put $VENV/bin on PATH, so whatever that binary later execs BY NAME is invisible
#       even though it sits right beside it. Given a program we invoke by absolute path,
#       put its bin dir on PATH — and say so on stderr, because a repair nobody can see is
#       drift nobody notices.
#
#   pathguard_require <what needs them> <program>...
#       Assert every program resolves NOW, before anything expensive happens, and refuse
#       with the list of the ones that do not. For each miss it searches the bin dirs we
#       know about and reports "installed at X, but that dir is not on PATH" separately
#       from "not installed anywhere we looked" — that distinction is the entire diagnosis,
#       and it is exactly what `command not found` withholds.
#
# The division of labour matters. adopt REPAIRS the dirs we already know about; that is the
# behaviour that makes modelctl work today and it stays. require is the ASSERTION that the
# result is actually sufficient — it is what fires for the fifth program, the one in a
# directory nobody thought to adopt, and it fires while nothing is billing. require never
# repairs PATH: silent repair would hide exactly the drift we want to see.
#
# The python-side twin of require is `manifest.py grading-preflight`, which checks the
# binaries and modules grade() will reach for before the first model call. Between them,
# every program the harness execs by name is asserted before the expensive step that needs
# it: serving (here, via modelctl) and grading (there).
#
# Self-test:  bash lib/pathguard.sh --self-test        (run by smoke/run-smoke.sh, so CI
#                                                       covers it on every push)

# Bin dirs adopted by this process, ':'-joined. Kept because they are also the first place
# to look when something is missing: if we were pointed at a venv, that venv is where the
# rest of its console scripts live.
PATHGUARD_ADOPTED="${PATHGUARD_ADOPTED:-}"
# Extra dirs to SEARCH on a miss (never added to PATH). ':'-joined; callers may extend it.
PATHGUARD_SEARCH_DIRS="${PATHGUARD_SEARCH_DIRS:-}"

# pathguard_adopt <bin-dir> | <program invoked by absolute path>
# Prepend the bin dir to PATH, idempotently, and record it. Silent no-op when the argument
# is empty, relative or does not exist: a bare name ($VLLM_BIN defaults to `vllm`) is
# already a PATH lookup, and a phantom PATH entry is not a repair, it is noise.
pathguard_adopt() {
  local target="${1:-}" dir=""
  [[ "$target" == /* ]] || return 0
  if [[ -d "$target" ]]; then
    dir="$(cd "$target" && pwd)" || return 0
  elif [[ -x "$target" ]]; then
    dir="${target%/*}"                      # not dirname: see pathguard_require
    dir="$(cd "${dir:-/}" && pwd)" || return 0
  else
    return 0
  fi
  case ":$PATHGUARD_ADOPTED:" in
    *":$dir:"*) ;;
    *) PATHGUARD_ADOPTED="${PATHGUARD_ADOPTED:+$PATHGUARD_ADOPTED:}$dir" ;;
  esac
  case ":$PATH:" in
    *":$dir:"*) return 0 ;;
  esac
  PATH="$dir:$PATH"; export PATH
  printf 'pathguard: PATH += %s (bin dir of %s)\n' "$dir" "$target" >&2
}

# Candidate dirs to search when a program is missing, one per line. Diagnosis only —
# nothing here is added to PATH.
_pathguard_search_dirs() {
  local d dirs
  dirs="$PATHGUARD_ADOPTED:$PATHGUARD_SEARCH_DIRS:$HOME/.local/bin:/usr/local/bin:/opt/homebrew/bin:/snap/bin"
  # Any venv directly under $HOME: gpuctl builds ~/harness-venv, and that is precisely
  # where `ninja` sat for the whole of AI-3159.
  for d in "$HOME"/*/bin; do
    [[ -d "$d" ]] && dirs="$dirs:$d"
  done
  (
    IFS=:
    for d in $dirs; do
      if [[ -n "$d" && -d "$d" ]]; then printf '%s\n' "$d"; fi
    done
  )
  return 0
}

# pathguard_where <program> — first place we can find it that PATH could not. Empty if none.
pathguard_where() {
  local prog="${1:-}" d
  [[ -n "$prog" ]] || return 0
  while IFS= read -r d; do
    if [[ -x "$d/$prog" ]]; then printf '%s\n' "$d/$prog"; return 0; fi
  done < <(_pathguard_search_dirs)
  return 0
}

# pathguard_require <what needs them> <program>...
# 0 if every program resolves; otherwise prints the diagnosis on stderr and returns 1.
# An argument containing a slash is an explicit path, not a PATH lookup, and is tested as
# a file — that is how $VLLM_BIN and $HARNESS_PYTHON arrive here.
pathguard_require() {
  local what="${1:-programs}"; shift || true
  local prog resolved found missing=() ok_line=""
  for prog in "$@"; do
    [[ -n "$prog" ]] || continue
    if [[ "$prog" == */* ]]; then
      if [[ -x "$prog" ]]; then ok_line="$ok_line $prog"; continue; fi
    else
      if resolved="$(command -v "$prog" 2>/dev/null)"; then ok_line="$ok_line $resolved"; continue; fi
    fi
    missing+=("$prog")
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    printf 'pathguard: %s — ok:%s\n' "$what" "$ok_line" >&2
    return 0
  fi

  {
    printf '\n  !!  REQUIRED PROGRAM NOT FOUND — REFUSING TO CONTINUE  !!\n'
    printf '      needed by : %s\n' "$what"
    for prog in "${missing[@]}"; do
      # ${x##*/} and ${x%/*}, not basename/dirname: a report about a broken PATH must not
      # itself depend on finding programs on PATH.
      found="$(pathguard_where "${prog##*/}")"
      if [[ -n "$found" ]]; then
        printf '      %-12s INSTALLED at %s — but that directory is not on PATH.\n' "$prog" "$found"
        printf '      %-12s A non-interactive `ssh host "cmd"` never sources ~/.profile, so this\n' ""
        printf '      %-12s reads as "not installed" and never is. fix: export PATH=%s:$PATH\n' "" "${found%/*}"
      elif [[ "$prog" == */* ]]; then
        printf '      %-12s no such executable (explicit path)\n' "$prog"
      else
        printf '      %-12s not found on PATH, and not in any bin dir we know of\n' "$prog"
        printf '      %-12s fix: install it into the environment that will run this\n' ""
      fi
    done
    printf '      PATH      : %s\n' "$PATH"
    printf '  Nothing expensive has happened yet — this is the cheap place to find out.\n\n'
  } >&2
  return 1
}

# --------------------------------------------------------------------- self-test
# Sourced, this file defines functions and does nothing else. Run directly with
# --self-test it proves the two behaviours the guard exists for: adopting a bin dir makes
# its programs resolve, and a missing program is REFUSED with the "installed over there"
# distinction rather than repaired.
if [[ "${BASH_SOURCE[0]}" == "${0}" && "${1:-}" == "--self-test" ]]; then
  set -euo pipefail
  _t="$(mktemp -d "${TMPDIR:-/tmp}/pathguard-XXXXXX")"
  trap 'rm -rf "$_t"' EXIT
  _fail() { printf 'pathguard self-test FAILED: %s\n' "$*" >&2; exit 1; }

  mkdir -p "$_t/bin"
  printf '#!/bin/sh\nexit 0\n' >"$_t/bin/pathguard-fake"; chmod +x "$_t/bin/pathguard-fake"

  # 1. a program that exists but is not on PATH is missing, and is diagnosed as installed.
  PATHGUARD_SEARCH_DIRS="$_t/bin"
  _out="$(pathguard_require "self-test" pathguard-fake 2>&1)" && _fail "require passed for a program off PATH"
  case "$_out" in *"INSTALLED at $_t/bin/pathguard-fake"*) ;; *) _fail "no 'INSTALLED at' diagnosis: $_out" ;; esac

  # 2. adopting the dir makes it resolve — and adopting twice does not double the entry.
  pathguard_adopt "$_t/bin" 2>/dev/null
  _before="$PATH"
  pathguard_adopt "$_t/bin" 2>/dev/null
  [[ "$PATH" == "$_before" ]] || _fail "pathguard_adopt is not idempotent"
  pathguard_require "self-test" pathguard-fake >/dev/null 2>&1 || _fail "require failed after adopt"

  # 3. adopting a PROGRAM path adopts its bin dir (the $VLLM_BIN / $HARNESS_PYTHON case).
  mkdir -p "$_t/venv/bin"
  printf '#!/bin/sh\nexit 0\n' >"$_t/venv/bin/pathguard-sibling"; chmod +x "$_t/venv/bin/pathguard-sibling"
  printf '#!/bin/sh\nexit 0\n' >"$_t/venv/bin/pathguard-main"; chmod +x "$_t/venv/bin/pathguard-main"
  pathguard_adopt "$_t/venv/bin/pathguard-main" 2>/dev/null
  pathguard_require "self-test" pathguard-sibling >/dev/null 2>&1 \
    || _fail "adopting a program path did not expose its siblings"

  # 4. a genuinely absent program is refused, and says so without claiming it is installed.
  _out="$(pathguard_require "self-test" pathguard-nowhere-at-all 2>&1)" && _fail "require passed for an absent program"
  case "$_out" in *"not in any bin dir we know of"*) ;; *) _fail "wrong diagnosis for an absent program: $_out" ;; esac

  # 5. an explicit path is tested as a file, not looked up on PATH.
  pathguard_require "self-test" "$_t/bin/pathguard-fake" >/dev/null 2>&1 || _fail "explicit path rejected"
  pathguard_require "self-test" "$_t/bin/nope" >/dev/null 2>&1 && _fail "nonexistent explicit path accepted"

  printf 'pathguard self-test ok\n'
fi
