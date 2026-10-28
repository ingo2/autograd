# AGENTS.md

Rules for any coding agent working in this repository. Keep this file short: it is loaded by
tools with a small context window.

## Layout

Pure Go module `autograd`:

- `nn/`: the micrograd-style autograd engine (`Value`) and the MLP built on it.
- `tictactoe/`: the m,n,k game engine, minimax, and board generation/annotation.
- `scoremodel/`: the per-move score model, its losses, training, evaluation, and player.
- `cmd/`: the binaries `datagen`, `train`, `tictactoe`, and `xor`.
- `models/`: trained models (committed). `datasets/`: generated boards (git-ignored).

## Formatting

- Line breaks at 100 characters for Go, shell scripts, markdown and text. Use the full width, do
  not wrap at 80 characters. Only break earlier when it genuinely aids readability.
- More than 100 characters are ok for function declarations in Go or when line breaks would be
  very confusing.
- ASCII characters only: no Unicode, no fancy quotes, no special symbols. This applies to all
  source, markdown, text, and data files in the project.
- Never use an em dash or a double hyphen (`--`) as punctuation. Please resist. Use a single hyphen
  (-), a comma, a colon, parentheses, or a separate sentence instead. This covers prose, comments,
  docstrings, markdown and commit messages. A `--` that is part of a command line is of course
  fine.
- Sentences should end with a period (.), even if there is no verb in the sentence. Does not apply
  to short inline comments.
- Keep comments short and precise. Avoid overly complex sentences. Write only what a reader will
  still need later, not the context of the conversation the code was written in.

## Go

- Code must be `gofmt`-formatted and pass `go vet ./...`.
- When compiling a binary to verify a build (e.g. `go build -o ...`), always output it to
  `tmp/` in the repo root. That directory is git-ignored, so binaries never end up in commits.
- When adding, removing, or renaming CLI flags on a binary that has an entry in
  `.vscode/launch.json`, update that entry in the same change so the debug config stays runnable.
  The same applies to the shell scripts under `cmd/` that call that binary.

## Shell scripts

- Always double-quote variable expansions: `"${MYVAR}"` not `$MYVAR` or `${MYVAR}`.

## Testing

- Run all tests with `go test ./...`.
- When adding a new function or method, also add a test for it if a test file already exists in
  that package.
- Prefer table-driven tests for multiple cases.
