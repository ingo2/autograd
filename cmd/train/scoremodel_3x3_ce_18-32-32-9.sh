#!/usr/bin/env bash
set -euo pipefail

# The cross-entropy loss floor for this training setup.
#
# FYI: Unlike MSE or hinge loss, cross-entropy against a uniform target over best_pos has an 
# irreducible loss floor. This follows from the exact decomposition of the cross-entropy:
#
#   H(p, q)  =  H(p)            +  KL(p || q)
#   total       floor:             reducible:
#   loss        uncertainty of     penalty for the
#               the target         model deviating
#               itself             from the target
#
# [KL = Kullback-Leibler-Divergence]
#
# Training can only drive the KL term to zero (reached when q = p); what remains is H(p), the
# entropy of the target itself. On a board with k equally optimal moves the target is uniform over
# them, so the per-board floor is H(p) = ln(k); it is zero only when the best move is unique.

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../.." && pwd)"
BIN="${DIR}/bin/train"
DATA="${REPO_ROOT}/datasets/boards_3x3_perfect.ndjson"
OUT="${REPO_ROOT}/models/scoremodel_3x3_ce_18-32-32-9.json"

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
    -matchup-interval   8 \
    -matchup-patience   3 \
    -matchup-games      100 \
    -matchup-opponent   minimax \
    -learning-rate      0.01 \
    -weight-decay       1e-4 \
    -activation-hidden  relu \
    -activation-out     linear \
    -loss               crossentropy \
    -test-fraction      0.1 \
    -outcome-classes
echo

echo "Done."
