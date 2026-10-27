#!/usr/bin/env bash
set -euo pipefail

# The cross-entropy loss floor for this training setup.
#
# Unlike MSE or hinge loss, cross-entropy against a uniform target over
# best_pos has an irreducible loss floor. This follows from the exact
# decomposition of the cross-entropy:
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
#
# The dataset floor is the mean of ln(|best_pos|) over all training boards. For the full 3x3 dataset
# (4298 non-terminal boards with more than one legal move, discounted solver with gamma = 0.9) this
# is approximately 0.3222: 2920 boards have a unique best move (floor 0), the remaining 1378 have
# 2-9 tied optimal moves (floor ln(2)..ln(9)). A perfect model converges to ~0.32 not to 0.
#
# Note that loss values are not comparable across loss functions (MSE and hinge have a floor of
# zero) and the loss is only a proxy anyway: the real signal is the play-strength metric
# (critical-position accuracy on held-out boards). A loss sitting just above the floor is the
# signature of a healthy run; a loss significantly below the floor of the training set indicates a
# bug in masking or targets.

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
    -epochs             300 \
    -learning-rate      0.01 \
    -patience           10 \
    -test-patience      10 \
    -weight-decay       1e-4 \
    -activation-hidden  relu \
    -activation-out     linear \
    -loss               crossentropy \
    -test-fraction       0.1 \
    -validation-fraction 0.0 \
    -outcome-classes     true
echo

echo "Done."
