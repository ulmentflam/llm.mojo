# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, or any other tool)
working in this repository. Read this before making changes.

For what this project *is* and how to build/train/test it, see
[`README.md`](README.md) — this file is about how an agent should work here,
not what the code does.

## 1. Disclose what you worked on

Every agentic contribution to this repo must be legible after the fact —
to the person who asked for it, and to the next agent (possibly you, in a
future session) who picks the repo back up.

- **Commit messages** explain what broke/changed and why, not just what.
  Look at recent history (`git log`) for the standard this repo holds to
  before writing one — generic messages ("update code", "fix bug") are not
  acceptable.
- **Final reports/summaries** to the user state concretely what was done,
  what was verified (and how), and what wasn't finished or is uncertain.
  Don't claim something works because you wrote it; state that you ran it
  and what you observed.
- **Every `docs/ai/*.md` file ends with an "AI use statement"** — a short
  closing section naming that the work was done with AI assistance, the
  tool, and who directed it. Every existing file in `docs/ai/` follows this;
  match it exactly rather than inventing a new disclosure format.

## 2. `docs/ai/` is where agentic implementation work and its documentation live

If an agent designs, implements, debugs, or benchmarks something non-trivial
in this repo — a root-caused bug, a porting campaign, a benchmark
methodology, an optimization pass — the writeup belongs in `docs/ai/`, not
scattered across commit messages, PR descriptions, or nowhere. Commit
messages should be a pointer to the doc for anything with real depth, not a
substitute for it.

Existing files to match in tone, depth, and structure (read at least one
before writing a new one):
- `docs/ai/ai_assisted_optimizations_and_benchmarks.md` — a running
  benchmark/optimization log.
- `docs/ai/metal_port_gotchas_and_optimizations.md` — a porting-campaign
  writeup (root causes, fixes, gotchas catalog).
- `docs/ai/gpt2_124m_fineweb_training_run.md` — a full run report (timeline,
  incidents, hyperparameters, replication steps).
- `docs/ai/bf16_generation_misaligned_address_bug.md` — a single-bug deep
  dive (symptom, root cause, fix, verification).

Pick the closest-fitting shape for what you're documenting rather than
inventing a fifth structure. Every doc ends with the AI use statement (§1).

If you're not sure whether something rises to "belongs in `docs/ai/`":
err toward writing it down. A short, real doc beats a long commit message
nobody will read six months from now.

## 3. Operational conventions learned the hard way in this repo

These aren't style preferences — each one is here because skipping it caused
a real incident during this project's development. Follow them.

- **Verify the binary is fresh before trusting a run.** `mojo build` outputs
  are not automatically rebuilt by every entry point; check the binary's
  mtime against the source you just changed before drawing conclusions from
  its behavior.
- **Prefer a `make <target>` over a bare `pixi run python scripts/...` or raw
  binary invocation** when documenting how to run something (in READMEs,
  docstrings, or your own final report). This repo's convention is that
  reproducing any result is one `make` command; keep it that way.
- **Parallel agents must not share one mutable working tree.** Two agents
  each running `git add`/`git commit` against the same checkout will race —
  one's commit can silently sweep up the other's in-progress staged changes.
  Any fan-out of multiple simultaneous agents against this repo must give
  each its own `git worktree`; a coordinator reviews and merges each one
  into `main` afterward. Agents working in an isolated worktree should
  commit freely there but not push/merge to `main` themselves.
- **`make lint` (ruff + `mojo format --check` + pyrefly) gates every commit**
  via a pre-commit hook that runs across the *whole* tree, not just staged
  files. An unrelated unformatted file anywhere will block your commit —
  run `make lint` yourself before committing to catch this early, and don't
  work around it with `--no-verify`.
- **A checkpoint/build claim needs an independent check, not just the
  author's word.** Before merging another agent's work (especially from an
  isolated worktree, where you can't see it happen), rebuild and re-run the
  specific thing it claims to have verified yourself if the claim is load-
  bearing (training resumes correctly, inference produces the same output
  across code paths, a kernel fix doesn't regress the hot path, etc.).

## AI use statement

This file was written with AI assistance via Claude Code, under the
direction of Evan Owen, to formalize disclosure and documentation
conventions that emerged organically over the course of AI-assisted
development on this project.
