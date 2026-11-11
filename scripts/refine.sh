#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if (( $# < 3 )); then
    echo "Usage: $0 EVENT_MODEL_DIR BLUR_MANIFEST NEW_OUTPUT_DIR [refine_appearance.py options...]" >&2
    exit 2
fi
event_model="$1"
manifest="$2"
output="$3"
shift 3
"$PYTHON" refine_appearance.py --event_model "$event_model" --blur_manifest "$manifest" -m "$output" "$@"
