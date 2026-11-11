#!/usr/bin/env bash
# Shared path handling; source this file from a runner.
set -euo pipefail
EVENT3DGS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$EVENT3DGS_ROOT"
