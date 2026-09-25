#!/bin/bash
#SBATCH --job-name=amz_er_test_a100
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-A100
#SBATCH --qos=phd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Test pipeline for the entity-resolution challenge on a a100 node (generated from test_h100.slurm):
# test query embeddings -> GPU TF-IDF retrieval (4 parts) -> XGBoost scoring
# -> output/matching_results.tsv + output/candidate_pairs.tsv -> official validator.
#
# Submit with slurm/submit.sh (not sbatch directly):  cd ~/Sumit/AMZ && bash slurm/submit.sh slurm/test_a100.slurm
#
# Compute nodes have no internet, and home storage is limited, so:
#   - inputs + container image live in /tmp/amz on the login node; submit.sh copies
#     them to a per-job staging folder in home ($STAGE), which this job moves to the
#     node-local SSD ($TMPDIR) and deletes from home as its first step;
#   - packages come as a prebuilt venv (venv.tar.gz, built on the login node for the
#     python:3.12 image py312.sif); it is unpacked on $TMPDIR and mounted at /opt/venv;
#   - all intermediates (~30 GB) stay on $TMPDIR; gzipped outputs + logs go to
#     $TMPDIR/results, which cluster policy copies to ~/job_results/$SLURM_JOB_ID.
set -euo pipefail
: "${STAGE:?submit with slurm/submit.sh, which sets STAGE}"
cd "$TMPDIR" || exit 1
echo "Job $SLURM_JOB_ID running on $(hostname) at $(date)"
echo "TMPDIR: $TMPDIR"
df -h "$TMPDIR" | tail -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# --- stage to scratch, then free the home copy
cp -r "$STAGE"/. ./
rm -rf "$STAGE"
echo "$(date +%T) staged to $TMPDIR, removed $STAGE from home"
mkdir -p results opt
tar -xzf venv.tar.gz -C opt && rm venv.tar.gz
RUN="apptainer exec --nv --bind $TMPDIR --bind $TMPDIR/opt:/opt --env HF_HOME=$TMPDIR/AMZ/hf_cache --env HF_HUB_OFFLINE=1 $TMPDIR/py312.sif"
PYTHON=/opt/venv/bin/python

# --- fail fast if torch cannot use this GPU; XGBoost falls back to CPU where its
# build lacks the GPU architecture (e.g. P100 / SM 60): prediction only, 24+ cores
$RUN $PYTHON - <<'PYCHECK'
import torch
print("torch", torch.__version__, "| gpu", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
S = torch.rand(2000, 500, device="cuda"); S[S < 0.99] = 0
D = torch.rand(500, 64, device="cuda")
assert torch.allclose(torch.sparse.mm(S.to_sparse_csr(), D), S @ D, atol=1e-4), "sparse matmul mismatch"
print("torch gpu ok")
PYCHECK
XGB_DEVICE=cuda
$RUN $PYTHON -c "
import numpy as np, xgboost as xgb
x = np.random.rand(1000, 5).astype(np.float32); y = (x[:, 0] > 0.5).astype(int)
xgb.train({'device': 'cuda', 'tree_method': 'hist', 'objective': 'binary:logistic'}, xgb.DMatrix(x, label=y), 5)
" > /dev/null 2>&1 || XGB_DEVICE=cpu
echo "xgboost device: $XGB_DEVICE"

# --- pipeline
echo "$(date +%T) running the test pipeline"
cd AMZ
status=0
$RUN env PY=$PYTHON XGB_DEVICE=$XGB_DEVICE VALIDATOR=validator/validate_submission.py \
    bash scripts/run_test.sh local xgb-train30-full || status=$?

# --- results (copied back to ~/job_results/$SLURM_JOB_ID)
cp -r logs "$TMPDIR/results/" 2>/dev/null || true
for f in output/*.tsv; do [ -f "$f" ] && gzip -c "$f" > "$TMPDIR/results/$(basename "$f").gz"; done
ls -la "$TMPDIR/results"
echo "Job finished at $(date) with status $status"
exit $status
