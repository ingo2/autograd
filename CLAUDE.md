# CLAUDE.md

The shared coding rules for this repository live in `AGENTS.md` so that other tools pick them up
too. They are imported here and apply in full:

@AGENTS.md

Everything below is additional context that only Claude Code loads.

## Project overview

A hand-written autograd engine in Go (micrograd-style) and the MLPs built on top of it, applied to
the m,n,k game (generalized tic-tac-toe), starting with 3x3. The goal is to understand how the
methods work, not just to ship a strong player.

Data artifacts: generated board datasets live in `datasets/` (git-ignored, regenerate with
`cmd/datagen/generate_boards_*.sh`), trained models in `models/` (committed, retrain with the
scripts in `cmd/train/`).

## Engineering approach

- Despite the experimental purpose of the project, we aim to adhere to best practices: write
  well-structured code and keep the project organized.
- We generally improve things iteratively rather than building everything up front, but still
  avoid foundational choices that are likely to cause a big mess or technical debt later on.
- Avoid unnecessary complexity and over-design. Prefer the simplest solution that does the job
  well.
- Do not implicitly assume a request implies implementing something. Only start changing code
  when explicitly instructed; when in doubt, ask back first.

## Key conventions

- Board values: 0 = empty, 1 = side to move, 2 = opponent (perspective-canonical).
- Value target: discounted minimax value in [-1, 1], gamma = 0.9 by default.
- Move selection: argmax over the model's per-move scores on legal cells (no minimax playing).
- Dataset: reachable decision states only (no terminals), NDJSON with per-move scores.
