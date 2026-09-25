#!/usr/bin/env bash
# Stage inputs from /tmp/amz (login node) into a per-job folder in home and submit.
# The job moves the folder to node-local scratch and deletes it from home at start.
#
#   cd ~/Sumit/AMZ && bash slurm/submit.sh slurm/test_p100.slurm
set -euo pipefail
JOB_SCRIPT=${1:?usage: submit.sh <slurm script>}
SRC=${SRC:-/tmp/amz}
HERE=$(cd "$(dirname "$0")/.." && pwd)
STAGE="$HERE/stage-$(date +%Y%m%d-%H%M%S)"

for f in py312.sif venv.tar.gz; do [ -f "$SRC/$f" ] || { echo "missing $SRC/$f"; exit 1; }; done
mkdir -p "$STAGE/AMZ"
cp "$SRC/py312.sif" "$SRC/venv.tar.gz" "$STAGE/"
cp -r "$HERE/scripts" "$HERE/utils" "$HERE/requirements.txt" "$STAGE/AMZ/"
cp -r "$SRC/artifacts" "$SRC/dataset" "$SRC/hf_cache" "$SRC/validator" "$STAGE/AMZ/"
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
echo "staged $(du -sh "$STAGE" | cut -f1) in $STAGE"
cd "$HERE"
sbatch --export=ALL,STAGE="$STAGE" "$JOB_SCRIPT"
