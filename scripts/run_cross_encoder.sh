#!/usr/bin/env bash
# Cross-encoder pipeline, cross-fitted on the training split (the eval split is never trained on):
#   GPU retrieval over the full training split (4 parts)
#   -> fine-tune <name>-A on fold A and <name>-B on fold B
#   -> score eval-30 (mean of both models) and train-30 (out-of-fold), the XGBoost rows.
#
#   bash scripts/run_cross_encoder.sh new [name] [train args...]      each step in its own allocation;
#                                                                     parts, folds and runs in parallel
#   bash scripts/run_cross_encoder.sh <slurm job id> [name] [...]     steps one after another in that job
#   bash scripts/run_cross_encoder.sh local [name] [...]              steps directly, one after another
#   name default: ce-minilm; train args go to train_cross_encoder.py (e.g. --train-queries 400000)
#   env: PY, STEP_OPTS (srun resources per step inside a job), ALLOC_OPTS (per allocation with "new"),
#        FOLDS (default "A B")
#
# Progress: logs/run_cross_encoder-<name>.log; per-step logs in logs/.
# Resumable: finished retrieval parts and models are skipped (a part still running from an
# earlier launch is waited for), training restarts from its last checkpoint, scoring skips
# finished chunks.
set -euo pipefail
JOB=${1:?usage: run_cross_encoder.sh <new|slurm job id|local> [name] [train args...]}
NAME=${2:-ce-minilm}
shift $(( $# < 2 ? $# : 2 ))
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PY:-/scratch/ckarfa/venvs/sumit/bin/python}
STEP_OPTS=${STEP_OPTS:---cpus-per-task=6 --gres=gpu:1g.24gb:1}
ALLOC_OPTS=${ALLOC_OPTS:--p gpu_small --gres=gpu:1g.24gb:1 --cpus-per-task=6 --mem=45G}
mkdir -p logs
log() { echo "$(date '+%F %T')  $*" | tee -a "logs/run_cross_encoder-$NAME.log"; }
trap 'log "FAILED (line $LINENO); see the step logs in logs/"' ERR

# step <time limit> <command...>: one pipeline step on the chosen resources
step() {
    local limit=$1; shift
    case "$JOB" in
        local) "$@" ;;
        new) srun $ALLOC_OPTS -t "$limit" "$@" ;;
        *) srun --jobid="$JOB" --overlap --ntasks=1 $STEP_OPTS "$@" ;;
    esac
}
# run <stdout file> <time limit> <command...>: in the background with "new", else in the foreground
PIDS=()
run() {
    local out=$1; shift
    if [ "$JOB" = new ]; then step "$@" > "$out" 2>&1 & PIDS+=($!); else step "$@" > "$out" 2>&1; fi
}
wait_all() { for p in "${PIDS[@]}"; do wait "$p"; done; PIDS=(); }

log "start: cross-encoder $NAME on $JOB, train args: ${*:-(defaults)}"
RUNNING=()
for k in 0 1 2 3; do
    RUN=gpu-tfidf-train-part${k}of4
    if grep -q "fused candidate rows" "logs/$RUN.log" 2>/dev/null; then
        log "retrieval $RUN: already done, skipped"
    elif pgrep -f "gpu_retrieve.py --queries train --parts 4 --part $k " > /dev/null; then
        log "retrieval $RUN: still running from an earlier launch, waiting for it"
        if [ "$JOB" = new ]; then
            RUNNING+=("$k")
        else  # one step at a time: finish it before the next part starts
            while pgrep -f "gpu_retrieve.py --queries train --parts 4 --part $k " > /dev/null; do sleep 60; done
        fi
    else
        run "logs/$RUN.stdout" 2:00:00 $PY scripts/gpu_retrieve.py --queries train --parts 4 --part "$k" --no-tsv
    fi
done
for k in "${RUNNING[@]}"; do
    while pgrep -f "gpu_retrieve.py --queries train --parts 4 --part $k " > /dev/null; do sleep 60; done
done
wait_all
for k in 0 1 2 3; do
    grep -q "fused candidate rows" "logs/gpu-tfidf-train-part${k}of4.log" || { log "retrieval part $k incomplete"; exit 1; }
done
log "retrieval done: 4 parts"

MODELS=()
for FOLD in ${FOLDS:-A B}; do
    MODELS+=("$NAME-$FOLD")
    if [ -f "artifacts/cross_encoders/$NAME-$FOLD/meta.json" ]; then
        log "training $NAME-$FOLD: already done, skipped"
    else
        run "logs/ce-train-$NAME-$FOLD.stdout" 12:00:00 \
            $PY scripts/train_cross_encoder.py --name "$NAME-$FOLD" --fold "$FOLD" "$@"
    fi
done
wait_all
for M in "${MODELS[@]}"; do
    [ -f "artifacts/cross_encoders/$M/meta.json" ] || { log "training $M incomplete"; exit 1; }
    log "trained $M: $(grep 'valid:' "logs/ce-train-$M.log" | tail -1 | sed 's/^.*valid: //')"
done

for RUN in gpu-tfidf-eval-30 gpu-tfidf-train-30; do
    run "logs/ce-score-$NAME-$RUN.stdout" 4:00:00 \
        $PY scripts/score_cross_encoder.py --models "${MODELS[@]}" --name "$NAME" --run "$RUN"
done
wait_all
log "scored gpu-tfidf-eval-30 and gpu-tfidf-train-30 -> artifacts/ce_scores/$NAME/"
log "finished"
