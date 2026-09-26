#!/usr/bin/env bash
# Round 3 (revised): three cross-encoders feeding one XGBoost, scoring the candidates within
# GAP of some search's #1 score (score_cross_encoder.py --gap) instead of the fused top 5:
# eval-30 recall of the scored set 99.25% with 3.36 pairs per record (fused top 5: 98.41% with 5).
#   ce-minilm-A   name | address   (round 2 model, re-scored on the new set)
#   cen-minilm-A  name only        (training on the big GPU, job 14832, started by run_r3.sh)
#   cea-minilm-A  address only     (training on the small GPU, job 14824, started by run_r3.sh)
#   -> score sets <model>-g02 for eval-30, train-30 and the 4 test parts
#   -> XGBoost xgb-train30-ce3g-minilm-A (fold B, same params as round 2)
#   -> gate: eval-30 macro F0.5 must beat round 2 (xgb-train30-ce-minilm-A)
#   -> predict -> output-r3/ (on /scratch) -> low-memory validator
# Both GPUs take scoring jobs from one list (a lock per job, so no job runs twice). The small job
# ends ~18:57; what it has not finished (training resumes from its checkpoint, scoring from its
# finished chunks) is picked up by the big job. output/ and output-ce/ are never touched.
cd /home/ckarfa/sumit/AMZ
PY=/scratch/ckarfa/venvs/sumit/bin/python
BIG="srun --jobid=14832 --overlap --ntasks=1 --cpus-per-task=6 --mem=40G --gres=gpu:2g.48gb:1"
BIG_XL="srun --jobid=14832 --overlap --ntasks=1 --cpus-per-task=10 --mem=44G --gres=gpu:2g.48gb:1"
SMALL="srun --jobid=14824 --overlap --ntasks=1 --cpus-per-task=6 --gres=gpu:1g.24gb:1"
CE=ce-minilm-A; CEN=cen-minilm-A; CEA=cea-minilm-A
GAP=0.2; SFX=g02
XGB=xgb-train30-ce3g-minilm-A; PREV=xgb-train30-ce-minilm-A
TRAIN_ARGS="--fold A --train-queries 400000"
XY="gpu-tfidf-eval-30 gpu-tfidf-train-30"
TEST="gpu-tfidf-test-part0of4 gpu-tfidf-test-part1of4 gpu-tfidf-test-part2of4 gpu-tfidf-test-part3of4"
OUT=output-r3
LOCKS=/scratch/ckarfa/amz/locks-r3b
mkdir -p "$LOCKS" logs
log() { echo "$(date '+%F %T')  $*" | tee -a logs/run_r3.log; }
fail() { log "FAILED: $*"; exit 1; }
trained() { [ -f "artifacts/cross_encoders/$1/meta.json" ]; }
scored() { [ -f "artifacts/ce_scores/$1-$SFX/$2/_done" ]; }
training() { pgrep -f "^(srun|$PY) .*train_cross_encoder\.py --name $1 " > /dev/null; }
jobs_of() { local runs=$1; shift; for m in "$@"; do for r in $runs; do echo "$m:$r"; done; done; }
all_done() { for j in "$@"; do scored "${j%%:*}" "${j#*:}" || return 1; done; }

# train_ce <srun prefix...> -- <name> <text mode>: wait for a running training, else (re)start it
train_ce() {
    local step=() ; while [ "$1" != -- ]; do step+=("$1"); shift; done; shift
    local name=$1 mode=$2
    while training "$name"; do sleep 60; done
    trained "$name" && return 0
    log "training $name ($mode): starting / resuming"
    "${step[@]}" $PY scripts/train_cross_encoder.py --name "$name" --text "$mode" $TRAIN_ARGS \
        >> "logs/ce-train-$name.stdout" 2>&1 || return 1
    trained "$name" || return 1
}

# score_once <srun prefix...> -- <model> <run> <where>: 0 scored, 1 failed, 2 taken / model not ready
score_once() {
    local step=() ; while [ "$1" != -- ]; do step+=("$1"); shift; done; shift
    local m=$1 run=$2 where=$3
    scored "$m" "$run" && return 0
    trained "$m" || return 2
    mkdir "$LOCKS/$m.$run" 2>/dev/null || return 2
    if "${step[@]}" $PY scripts/score_cross_encoder.py --models "$m" --name "$m-$SFX" --run "$run" --gap $GAP \
            >> "logs/ce-score-$m-$SFX-$run.stdout" 2>&1; then
        rmdir "$LOCKS/$m.$run"; log "CE $m scored $run ($where)"; return 0
    fi
    rmdir "$LOCKS/$m.$run"; log "CE $m scoring $run stopped ($where), see logs/ce-score-$m-$SFX-$run.stdout"; return 1
}

# worker <srun prefix...> -- <where> <job...>: one pass over the jobs; stops at the first failure
worker() {
    local step=() ; while [ "$1" != -- ]; do step+=("$1"); shift; done; shift
    local where=$1; shift
    for j in "$@"; do
        score_once "${step[@]}" -- "${j%%:*}" "${j#*:}" "$where"
        [ $? = 1 ] && return 1
    done
    return 0
}

log "==== round 3 revised: CE scoring set = candidates within $GAP of a search's #1 (was fused top 5); run_r3.sh stopped, trainings continue ===="

# ---- small GPU (14824): address model, then scoring jobs (address model first), in the background
(
    train_ce $SMALL -- $CEA address || { log "small GPU: training $CEA stopped (job 14824 ended?); the big GPU will resume it"; exit 1; }
    log "trained $CEA (address): $(grep 'valid:' logs/ce-train-$CEA.log | tail -1 | sed 's/^.*valid: //')"
    worker $SMALL -- small-gpu $(jobs_of "$XY" $CEA $CE $CEN) $(jobs_of "$TEST" $CEA $CE $CEN) \
        || { log "small GPU: stopping"; exit 1; }
    log "small GPU: no jobs left"
) &
SMALL_PID=$!

# ---- big GPU (14832): name model, then eval/train scoring jobs
train_ce $BIG -- $CEN name || fail "training $CEN, see logs/ce-train-$CEN.log"
log "trained $CEN (name): $(grep 'valid:' logs/ce-train-$CEN.log | tail -1 | sed 's/^.*valid: //')"
XY_JOBS=$(jobs_of "$XY" $CEN $CE $CEA)
while ! all_done $XY_JOBS; do
    worker $BIG -- big-gpu $XY_JOBS || fail "eval/train scoring on the big GPU"
    all_done $XY_JOBS && break
    if kill -0 $SMALL_PID 2>/dev/null; then
        sleep 60
    else
        train_ce $BIG -- $CEA address || fail "training $CEA on the big GPU, see logs/ce-train-$CEA.log"
    fi
done

# ---- XGBoost with all three cross-encoders (retry with fewer fit queries if memory runs out)
if [ ! -f "artifacts/models/$XGB.meta.json" ]; then
    for NQ in 400000 250000; do
        $BIG_XL $PY scripts/train_xgb.py --train gpu-tfidf-train-30 --eval gpu-tfidf-eval-30 --name $XGB \
            --train-queries $NQ --all-negatives --ce $CE-$SFX $CEN-$SFX $CEA-$SFX --fold B > logs/train-$XGB.stdout 2>&1 && break
        log "XGBoost with --train-queries $NQ failed (see logs/train-$XGB.stdout)"
        [ $NQ = 250000 ] && fail "XGBoost training"
    done
fi
F05=$($PY -c "import json; print(json.load(open('artifacts/models/$XGB.meta.json'))['eval']['f05'])")
BASE=$($PY -c "import json; print(json.load(open('artifacts/models/$PREV.meta.json'))['eval']['f05'])")
log "EVAL gpu-tfidf-eval-30: $XGB macro F0.5 $F05 vs round 2 ($PREV) $BASE"
$PY -c "
import json; m = json.load(open('artifacts/models/$XGB.meta.json')); e = m['eval']
print('  precision', round(e['pair_precision'], 4), 'recall', round(e['pair_recall'], 4), 'threshold', m['threshold'],
      'trees', m['best_iteration'] + 1, 'features', len(m['features']))" | tee -a logs/run_r3.log
$BIG $PY scripts/analyze_missed.py --model $XGB > /dev/null 2>&1 \
    && grep -E "accepted|rejected \(|wrong top|not retrieved" "logs/analyze-missed-gpu-tfidf-eval-30-$XGB.log" | head -4 | sed 's/^[0-9: -]*/  /' | tee -a logs/run_r3.log
$PY -c "import sys; sys.exit(0 if $F05 > $BASE else 1)" \
    || { log "gate: no gain over round 2, test not run; output-ce/ stays the best submission"; log "finished"; exit 0; }
log "gate: passed, scoring test"

# ---- test scoring: big GPU takes the list in the other order; leftovers after the small GPU stops
TEST_JOBS=$(jobs_of "$TEST" $CEN $CE $CEA | tac)
for pass in 1 2 3; do
    worker $BIG -- big-gpu $TEST_JOBS || fail "test scoring on the big GPU"
    all_done $TEST_JOBS && break
    log "waiting for the small GPU to finish or stop"; wait $SMALL_PID
done
all_done $TEST_JOBS || fail "test scores incomplete"
wait $SMALL_PID 2>/dev/null

# ---- predict test -> output-r3/, then validate
$BIG_XL $PY scripts/predict.py --model $XGB --runs $TEST --out $OUT \
    > logs/predict-$XGB-test.stdout 2>&1 || fail "predict test, see logs/predict-$XGB-test.stdout"
log "submission files written: $(ls -laL $OUT/*.tsv | awk '{print $NF, $5}' | tr '\n' ' ')"
log "md5 matching_results.tsv: $(md5sum $OUT/matching_results.tsv | cut -c1-32)"
srun --jobid=14832 --overlap --ntasks=1 --cpus-per-task=1 --mem=8G $PY scripts/validate_lowmem.py \
    --matching $OUT/matching_results.tsv --candidate $OUT/candidate_pairs.tsv --test-dir dataset/test \
    > logs/validate_lowmem-r3.log 2>&1 && log "validator: $(tail -1 logs/validate_lowmem-r3.log)" \
    || log "validator: FAILED, see logs/validate_lowmem-r3.log"
log "finished"
