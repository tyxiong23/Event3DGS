#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if (( $# < 2 )); then
    echo "Usage: $0 DATASET_SCENE OUTPUT_DIR [train_event.py options...]" >&2
    exit 2
fi
scene_path="$1"
output_path="$2"
shift 2
"$PYTHON" train_event.py -s "$scene_path" -m "$output_path" -r 1 --data_device cpu --eval "$@"
