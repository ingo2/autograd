#!/usr/bin/env bash
set -euo pipefail

# Annotates all reachable 3x3 positions using a perfect minimax oracle (unlimited search).

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../.." && pwd)"
BIN="${DIR}/bin/datagen"

echo "==> Compiling..."
mkdir -p "${DIR}/bin" "${REPO_ROOT}/datasets"
go build -o "${BIN}" "${DIR}"
echo

echo "==> Annotating all 3x3 boards..."
"${BIN}" \
    -gamma 0.9 \
    -overwrite \
    -out "${REPO_ROOT}/datasets/boards_3x3_perfect.ndjson"
echo

echo "Done."
