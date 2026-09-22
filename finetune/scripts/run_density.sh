#!/usr/bin/env bash
set -euo pipefail
python -m finetune.tasks.density_prediction "$@"
