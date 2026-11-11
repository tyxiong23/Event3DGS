#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if (( $# < 1 )); then
    echo "Usage: $0 RENDER_SET_DIR [evaluate.py options...]" >&2
    echo "Example RENDER_SET_DIR: output/lego/round_2/test/ours_30000" >&2
    exit 2
fi
render_set="$1"
shift
"$PYTHON" evaluate.py --render_dir "$render_set/renders" --gt_dir "$render_set/gt" "$@"
