#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if (( $# < 1 )); then
    echo "Usage: $0 MODEL_DIR [render_event.py options...]" >&2
    exit 2
fi
model_dir="$1"
shift
"$PYTHON" render_event.py -m "$model_dir" --skip_train "$@"
