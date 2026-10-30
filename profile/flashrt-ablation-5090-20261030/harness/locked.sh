#!/usr/bin/env bash
set -euo pipefail
trap 'nvidia-smi -rgc; nvidia-smi -rmc' EXIT
nvidia-smi -lgc 2400,2400
nvidia-smi -lmc 14001,14001
"$@"
