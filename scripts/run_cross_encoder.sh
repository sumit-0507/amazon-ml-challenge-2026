#!/usr/bin/env bash
# Cross-encoder pipeline, cross-fitted on the whole training split (the eval split is never trained on):
#   GPU retrieval over the full training split (4 parts)
#   -> fine-tune <name>-A on fold A and <name>-B on fold B
#   -> score eval-30 (mean of both models) and train-30 (out-of-fold), the XGBoost rows.
#
#   bash scripts/run_cross_encoder.sh <slurm job id> [name] [train args...]   attach each step to that job
#   bash scripts/run_cross_encoder.sh local [name] [train args...]            run steps directly (batch job)
#   name default: ce-minilm; train args go to train_cross_encoder.py (e.g. --base xlm-roberta-base)
#   env: PY, STEP_OPTS (srun resources per step), FOLDS (default "A B")
#
# Progress: logs/run_cross_encoder-<name>.log; per-step logs in logs/.
# Resumable: finished retrieval parts and models are skipped, training restarts from its
# last checkpoint, scoring skips finished chunks.
set -euo pipefail
JOB=${1:?usage: run_cross_encoder.sh <slurm job id|local> [name] [train args...]}
NAME=${2:-ce-minilm}
shift $(( $# < 2 ? $# : 2 ))
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PY:-/scratch/ckarfa/venvs/sumit/bin/python}
STEP_OPTS=${STEP_OPTS:---cpus-per-task=6 --gres=gpu:1g.24gb:1}
if [ "$JOB" = local ]; then STEP=""; else STEP="srun --jobid=$JOB --overlap --ntasks=1 $STEP_OPTS"; fi
mkdir -p logs
log() { echo "$(date '+%F %T')  $*" | tee -a "logs/run_cross_encoder-$NAME.log"; }
trap 'log "FAILED (line $LINENO); see the step logs in logs/"' ERR

log "start: cross-encoder $NAME, job $JOB"
for k in 0 1 2 3; do
    RUN=gpu-tfidf-train-part${k}of4
    if grep -q "fused candidate rows" "logs/$RUN.log" 2>/dev/null; then
        log "retrieval $RUN: already done, skipped"
    else
        $STEP $PY scripts/gpu_retrieve.py --queries train --parts 4 --part "$k" --no-tsv > "logs/$RUN.stdout" 2>&1
        log "retrieval $RUN done: $(grep -oE '[0-9,]+ fused candidate rows' "logs/$RUN.log")"
    fi
done

MODELS=()
for FOLD in ${FOLDS:-A B}; do
    MODELS+=("$NAME-$FOLD")
    if [ -f "artifacts/cross_encoders/$NAME-$FOLD/meta.json" ]; then
        log "training $NAME-$FOLD: already done, skipped"
    else
        $STEP $PY scripts/train_cross_encoder.py --name "$NAME-$FOLD" --fold "$FOLD" "$@" \
            > "logs/ce-train-$NAME-$FOLD.stdout" 2>&1
        log "training $NAME-$FOLD done: $(grep 'valid:' "logs/ce-train-$NAME-$FOLD.log" | tail -1 | sed 's/^.*valid: //')"
    fi
done

for RUN in gpu-tfidf-eval-30 gpu-tfidf-train-30; do
    $STEP $PY scripts/score_cross_encoder.py --models "${MODELS[@]}" --name "$NAME" --run "$RUN" \
        > "logs/ce-score-$NAME-$RUN.stdout" 2>&1
    log "scored $RUN"
done
log "finished"
