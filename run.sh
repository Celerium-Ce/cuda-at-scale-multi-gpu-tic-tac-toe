#!/usr/bin/env bash
# Builds the project (if needed) and plays one game of CUDA Multi-GPU
# Tic-Tac-Toe, writing a timestamped proof-of-execution log to logs/.
#
# Usage: ./run.sh [extra gpu_tictactoe args, e.g. --first O]
set -euo pipefail

cd "$(dirname "$0")"

make all

mkdir -p logs
log_file="logs/game_$(date +%Y%m%d_%H%M%S).log"

./bin/gpu_tictactoe --log "$log_file" "$@"

echo "Done. Log written to $log_file"
