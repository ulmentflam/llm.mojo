#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# llm.mojo box-wide gate lock.
#
# Serializes FULL test suites across worktrees. MAX compiles are already
# multi-threaded, so concurrent suites oversubscribe the cores rather than
# pipeline: four teams each run ~4x slower, and every duration measured on a
# loaded box is meaningless. One suite at a time is strictly faster for
# everyone AND is the only way timings mean anything.
#
#   make gate                               # the full merge gate, serialized
#   scripts/llm-mojo-gate.sh run make test  # wait for the lock, then run
#   scripts/llm-mojo-gate.sh status         # who holds it right now
#   scripts/llm-mojo-gate.sh run make format lint check test
#
# Zero-install equivalent (identical locking, no holder info):
#   flock /tmp/llm-mojo-gate.lock make test
#
# DO NOT lock short targeted runs -- `make smoke`, a single test file, a GEMM
# sweep. If the lock is inconvenient nobody uses it. Full suites only.
#
# STALE / ORPHANED LOCKS -- read this, it is the failure that motivated the
# whole convention. flock holds the lock on an open file descriptor, and
# CHILD PROCESSES INHERIT THAT DESCRIPTOR. So killing the runner script does
# NOT release the lock while any descendant is still alive: an orphaned
# `make test` keeps holding it. That is the correct semantics -- the work
# really is still running and really should still exclude others -- but it
# means "the holder died so the lock is free" is FALSE in exactly the case
# you care about. `status` therefore reports the live processes actually
# holding the descriptor, not the recorded PID, so an orphan is visible by
# name instead of looking like a wedged lock. Verified by killing a holder
# and watching its child keep the lock for a further 54 seconds.
# ---------------------------------------------------------------------------
set -uo pipefail

LOCK="${LLM_MOJO_GATE_LOCK:-/tmp/llm-mojo-gate.lock}"
INFO="${LOCK}.holder"

usage() { sed -n '2,33p' "$0" | sed 's/^# \?//'; exit 2; }

# Real processes holding the lock file open, found by scanning /proc rather
# than trusting the recorded PID (which an orphan invalidates).
real_holders() {
  local target pid fd
  target=$(readlink -f "$LOCK" 2>/dev/null) || return 0
  for fd in /proc/[0-9]*/fd/*; do
    [ -e "$fd" ] || continue
    if [ "$(readlink -f "$fd" 2>/dev/null)" = "$target" ]; then
      pid=${fd#/proc/}; pid=${pid%%/*}
      echo "$pid"
    fi
  done 2>/dev/null | sort -un
}

describe_pid() {
  local p=$1 cwd cmd
  cwd=$(readlink "/proc/$p/cwd" 2>/dev/null || echo "?")
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-90)
  printf '    pid=%-8s cwd=%s\n            cmd=%s\n' "$p" "$cwd" "$cmd"
}

cmd_status() {
  if [ ! -e "$LOCK" ]; then echo "gate lock: FREE (never used)"; return 0; fi

  if flock -n "$LOCK" true 2>/dev/null; then
    echo "gate lock: FREE"
    if [ -f "$INFO" ]; then
      echo "  (clearing stale holder record -- nothing holds the lock)"
      sed 's/^/    stale: /' "$INFO"
      rm -f "$INFO"
    fi
    return 0
  fi

  echo "gate lock: HELD"
  if [ -f "$INFO" ]; then
    sed 's/^/  recorded /' "$INFO"
    local hpid
    hpid=$(sed -n 's/^pid=//p' "$INFO" | head -1)
    if [ -n "${hpid:-}" ] && [ ! -d "/proc/$hpid" ]; then
      echo "  !! the RECORDED holder ($hpid) is DEAD, but the lock is still held."
      echo "     An orphaned descendant inherited the lock fd. Real holders:"
    fi
  else
    echo "  (no holder record -- probably a bare 'flock' one-liner)"
  fi

  local holders; holders=$(real_holders)
  if [ -n "$holders" ]; then
    echo "  live processes holding the lock:"
    local p; for p in $holders; do describe_pid "$p"; done
    echo "  To reclaim from an orphan, stop it GRACEFULLY (never SIGKILL a"
    echo "  multi-rank run -- it hangs GPU firmware): kill -INT <pid>, wait,"
    echo "  then -TERM. The lock frees when the last holder exits."
  else
    echo "  (could not identify holders -- may be owned by another user)"
  fi
}

cmd_run() {
  [ "$#" -ge 1 ] || usage
  if ! flock -n "$LOCK" true 2>/dev/null; then
    echo "gate lock is HELD; waiting..." >&2
    cmd_status >&2
  fi
  exec 200>"$LOCK"
  flock 200
  {
    echo "pid=$$"
    echo "worktree=$(pwd -P)"
    echo "started=$(date -Is)"
    echo "cmd=$*"
  } > "$INFO"
  trap 'rm -f "$INFO"' EXIT INT TERM
  echo "gate lock ACQUIRED by pid=$$ ($(pwd -P))" >&2
  "$@"
  local rc=$?
  echo "gate lock RELEASED (exit $rc)" >&2
  return $rc
}

case "${1:-}" in
  run)    shift; cmd_run "$@" ;;
  status) cmd_status ;;
  *)      usage ;;
esac
