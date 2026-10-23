#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${NC_POINTDATA:?Set NC_POINTDATA to the extracted pointdata directory}"
: "${NC_HISTORY:?Set NC_HISTORY to the compiled native-backend directory}"
export NC_DEVICE="${NC_DEVICE:-cuda}"
export NC_THREADS="${NC_THREADS:-1}"
export NC_STUDY="${NC_STUDY:-modality3_reproduction}"
python modality3_study.py prepare
python -u modality3_study.py run --parallel 3
python modality3_study.py aggregate
