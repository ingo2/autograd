#!/usr/bin/env bash
set -euo pipefail

# Trains all score-regression models defined in this directory.
DIR="$(cd "$(dirname "$0")" && pwd)"

"${DIR}/scoremodel_3x3_ce_18-32-32-9.sh"
"${DIR}/scoremodel_3x3_mse_18-32-32-9.sh"
"${DIR}/scoremodel_3x3_hinge_18-32-32-9.sh"
