#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if (( $# < 2 )); then
    echo "Usage: $0 DATASET_SCENE OUTPUT_ROOT [train_event.py options...]" >&2
    echo "Environment: ROUNDS=2 ITERATIONS=30000 OPACITY_THRESHOLD=0.9 PYTHON=python" >&2
    exit 2
fi
scene_path="$1"
output_root="$2"
shift 2
rounds="${ROUNDS:-2}"
iterations="${ITERATIONS:-30000}"
threshold="${OPACITY_THRESHOLD:-0.9}"
if ! [[ "$rounds" =~ ^[1-9][0-9]*$ && "$iterations" =~ ^[1-9][0-9]*$ ]]; then
    echo "ROUNDS and ITERATIONS must be positive integers." >&2
    exit 2
fi
# The runner owns these options, so its round/checkpoint paths cannot diverge.
for option in "$@"; do
    case "$option" in
        -m*|--model_path|--model_path=*|-s*|--source_path|--source_path=*|--iterations|--iterations=*|--ply_path|--ply_path=*|--ply_thres|--ply_thres=*|--start_checkpoint|--start_checkpoint=*)
            echo "Pass dataset/output positionally and use ITERATIONS/OPACITY_THRESHOLD; reserved option: $option" >&2
            exit 2 ;;
    esac
done
previous_ply=""
for (( round=1; round<=rounds; round++ )); do
    round_dir="$output_root/round_$round"
    if [[ -e "$round_dir/config.json" || -e "$round_dir/point_cloud" ]]; then
        echo "Refusing to overwrite an existing round: $round_dir. Use a new output root." >&2
        exit 2
    fi
done
for (( round=1; round<=rounds; round++ )); do
    round_dir="$output_root/round_$round"
    extra=()
    if (( round > 1 )); then
        [[ -f "$previous_ply" ]] || { echo "Missing preceding PLY: $previous_ply" >&2; exit 1; }
        extra=(--ply_path "$previous_ply" --ply_thres "$threshold")
    fi
    echo "Progressive round $round/$rounds: $round_dir"
    "$PYTHON" train_event.py -s "$scene_path" -m "$round_dir" -r 1 --data_device cpu --eval \
        "$@" --iterations "$iterations" "${extra[@]}"
    previous_ply="$round_dir/point_cloud/iteration_$iterations/point_cloud.ply"
done
printf 'Final model: %s\n' "$output_root/round_$rounds"
