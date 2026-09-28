#!/usr/bin/env bash
# Test pipeline, end to end:
#   test query embeddings -> GPU retrieval in parts (S1 test TF-IDF is built by part 0)
#   -> each part is scored by the model while the next part is retrieved
#   -> both submission files -> official validator.
#
#   bash scripts/run_test.sh <slurm job id> [model]   attach each step to that job (srun --overlap)
#   bash scripts/run_test.sh local [model]            run steps directly (e.g. inside a batch job /
#                                                     container); set PY to the python to use
#   model default: xgb-train30-full; env: PARTS (4), PY, VALIDATOR,
#   STEP_OPTS (srun resources per step), SEQUENTIAL=1 (score each part before retrieving
#   the next: halves peak memory, for small allocations)
#
# Progress: logs/run_test.log; per-step logs in logs/. Outputs: output/matching_results.tsv,
# output/candidate_pairs.tsv.
# Resumable: finished steps are skipped on a re-run (embeddings file present, retrieval log
# complete, scoring marker in artifacts/predictions/<model>-test/). RESTART=1 redoes everything.
set -euo pipefail
JOB=${1:?usage: run_test.sh <slurm job id> [model]}
MODEL=${2:-xgb-train30-full}
PARTS=${PARTS:-4}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PY:-python3}
VALIDATOR=${VALIDATOR:-6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py}
STEP_OPTS=${STEP_OPTS:---cpus-per-task=12 --gres=gpu:2g.48gb:1}
if [ "$JOB" = local ]; then STEP=""; else STEP="srun --jobid=$JOB --overlap --ntasks=1 $STEP_OPTS"; fi
mkdir -p logs output
log() { echo "$(date '+%F %T')  $*" | tee -a logs/run_test.log; }
trap 'log "FAILED (line $LINENO); see the step logs in logs/"' ERR

WORK=artifacts/predictions/$MODEL-test
if [ "${RESTART:-0}" = 1 ]; then rm -f artifacts/embeddings/s23-test.npy "$WORK"/scored-*.done logs/gpu-tfidf-test-part*of"$PARTS".log; fi
log "start: model $MODEL, $PARTS retrieval parts, job $JOB"
if [ -f artifacts/embeddings/s23-test.npy ]; then
    log "test query embeddings: already done, skipped"
else
    $STEP $PY scripts/embed_names.py s23-test > logs/embed-s23-test.stdout 2>&1
    log "test query embeddings done"
fi

SCORE_PID=""
for k in $(seq 0 $((PARTS - 1))); do
    RUN=gpu-tfidf-test-part${k}of${PARTS}
    if grep -q "fused candidate rows" "logs/$RUN.log" 2>/dev/null; then
        log "retrieval $RUN: already done, skipped"
    else
        $STEP $PY scripts/gpu_retrieve.py --queries test --parts "$PARTS" --part "$k" --no-tsv > "logs/$RUN.stdout" 2>&1
        log "retrieval $RUN done: $(grep -oE '[0-9,]+ fused candidate rows' "logs/$RUN.log")"
    fi
    if [ -n "$SCORE_PID" ]; then wait "$SCORE_PID"; log "scoring of the previous part done"; SCORE_PID=""; fi
    if [ -f "$WORK/scored-$RUN.done" ]; then
        log "scoring $RUN: already done, skipped"
    else
        ( $STEP $PY scripts/predict.py --runs "$RUN" --model "$MODEL" --stage score > "logs/predict-$RUN.stdout" 2>&1 \
          && touch "$WORK/scored-$RUN.done" ) &
        SCORE_PID=$!
        if [ "${SEQUENTIAL:-0}" = 1 ]; then wait "$SCORE_PID"; log "scoring $RUN done"; SCORE_PID=""; fi
    fi
done
if [ -n "$SCORE_PID" ]; then wait "$SCORE_PID"; log "scoring of the last part done"; fi

$STEP $PY scripts/predict.py --model "$MODEL" --stage write > logs/predict-write.stdout 2>&1
log "submission files written: $(ls -la output/*.tsv | awk '{print $NF, $5}' | tr '\n' ' ')"

$PY "$VALIDATOR" \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test > logs/validate_submission.log 2>&1 && log "validator: PASS" \
    || log "validator: FAILED, see logs/validate_submission.log"
log "finished"
