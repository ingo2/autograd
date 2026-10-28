#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../.." && pwd)"
BIN="${DIR}/bin/train"
DATA="${REPO_ROOT}/datasets/boards_3x3_perfect.ndjson"
OUT="${REPO_ROOT}/models/scoremodel_3x3_mse_18-32-32-9.json"

echo "==> Compiling..."
mkdir -p "${DIR}/bin"
go build -o "${BIN}" "${DIR}"
echo

echo "==> Training score regression MLP..."
"${BIN}" \
    -data          "${DATA}" \
    -out           "${OUT}" \
    -hidden        "32,32" \
    -epochs             400 \
    -matchup-interval   32 \
    -matchup-patience   3 \
    -matchup-games      100 \
    -matchup-opponent   minimax \
    -learning-rate      0.01 \
    -weight-decay       1e-4 \
    -activation-hidden  relu \
    -activation-out     tanh \
    -loss               mse \
    -test-fraction      0.1 \
    -outcome-classes
echo

echo "Done."
