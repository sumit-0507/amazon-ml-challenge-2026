#!/usr/bin/env bash
# End-to-end pipeline of the final submission (round 4): raw data -> blocking -> matching ->
# output/matching_results.tsv + output/candidate_pairs.tsv. Run from anywhere:
#
#   bash run_pipeline.sh
#
# Needs dataset/train/{train_source1,2,3,train_ground_truth}.tsv and
# dataset/test/test_source{1,2,3}.tsv next to this script, one CUDA GPU (>= 24 GB) and ~48 GB RAM.
# env: PY (python, default python3), STEP (prefix for every step, e.g. "srun --jobid=123 --overlap"),
#      VALIDATOR (path to the official validate_submission.py; the check is skipped when unset
#      and the challenge resource folder is not next to this script).
# Every step skips work that is already finished, so an interrupted run is resumed by re-running.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
STEP=${STEP:-}
mkdir -p logs output
log() { echo "$(date '+%F %T')  $*" | tee -a logs/run_pipeline.log; }
trap 'log "FAILED at line $LINENO; see the step logs in logs/"' ERR
run() { local out=$1; shift; $STEP "$@" > "logs/$out.stdout" 2>&1; }

TEST="gpu-tfidf-test-part0of4 gpu-tfidf-test-part1of4 gpu-tfidf-test-part2of4 gpu-tfidf-test-part3of4"
TRAIN_PARTS="gpu-tfidf-train-part0of4 gpu-tfidf-train-part1of4 gpu-tfidf-train-part2of4 gpu-tfidf-train-part3of4"
XY="gpu-tfidf-eval-30 gpu-tfidf-train-30"
MINILM=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
GTE=Alibaba-NLP/gte-multilingual-reranker-base
XGB=xgb-train30-ce4-A
GAP=0.2
done_run() { grep -q "fused candidate rows" "logs/$1.log" 2>/dev/null; }

log "==== pipeline start ===="

# 0. pretrained models (MIT / Apache-2.0) into the Hugging Face cache; later steps run offline
HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
for m in ['$MINILM', '$GTE', 'Alibaba-NLP/new-impl']:
    snapshot_download(m)"
log "0. pretrained models cached"

# 1. train / eval split by Source 1 entity (95 / 5, seed 42)
[ -f dataset/splits/ground_truth_eval.tsv ] || run split $PY scripts/split_train_eval.py
log "1. split: dataset/splits/"

# 2. native-script -> English word map learned from training matches (name standardization)
[ -f artifacts/native_word_map.json ] || run word-map $PY utils/name_standardization.py --build-word-map
log "2. word map: artifacts/native_word_map.json"

# 3. name embeddings (feature embedding_cosine)
[ -f artifacts/embeddings/s23-test.npy ] || run embed $PY scripts/embed_names.py s1-train s1-test s23-eval s23-train s23-test
log "3. embeddings: artifacts/embeddings/"

# 4. blocking: TF-IDF cosine (bi, tri, words, phonetic) within country + state, RRF fusion
done_run gpu-tfidf-eval-30 || run retrieve-eval-30 $PY scripts/gpu_retrieve.py --queries eval --frac 0.3 --name gpu-tfidf-eval-30
done_run gpu-tfidf-train-30 || run retrieve-train-30 $PY scripts/gpu_retrieve.py --queries train --frac 0.3 --name gpu-tfidf-train-30
for k in 0 1 2 3; do
    done_run gpu-tfidf-train-part${k}of4 || run retrieve-train-$k $PY scripts/gpu_retrieve.py --queries train --parts 4 --part $k --no-tsv
    done_run gpu-tfidf-test-part${k}of4 || run retrieve-test-$k $PY scripts/gpu_retrieve.py --queries test --parts 4 --part $k --no-tsv
done
log "4. candidates: artifacts/candidates/ (eval-30, train-30, 4 train parts, 4 test parts)"

# 5. cross-encoders, fine-tuned on fold A of the training split (fold B trains XGBoost)
train_ce() {  # name, text mode, extra args
    local name=$1 mode=$2; shift 2
    [ -f artifacts/cross_encoders/$name/meta.json ] || \
        run ce-train-$name $PY scripts/train_cross_encoder.py --name $name --fold A --text $mode "$@"
}
train_ce ce-minilm-A both --train-queries 400000
train_ce cen-minilm-A name --train-queries 400000
train_ce cea-minilm-A address --train-queries 400000
train_ce ceg-gte-A both --base $GTE --train-queries 250000 --lr 2e-5
log "5. cross-encoders: artifacts/cross_encoders/"

# 6. cross-encoder scores for the candidates within $GAP of some search's #1
for M in ce-minilm-A cen-minilm-A cea-minilm-A ceg-gte-A; do
    for RUN in $XY $TEST; do
        run ce-score-$M-$RUN $PY scripts/score_cross_encoder.py --models $M --name $M-g02 --run $RUN --gap $GAP
    done
done
log "6. cross-encoder scores: artifacts/ce_scores/"

# 7. XGBoost on fold B of train-30, threshold chosen by macro F0.5 on its validation part
[ -f artifacts/models/$XGB.meta.json ] || run train-$XGB $PY scripts/train_xgb.py \
    --train gpu-tfidf-train-30 --eval gpu-tfidf-eval-30 --name $XGB --train-queries 400000 --all-negatives \
    --ce ce-minilm-A-g02 cen-minilm-A-g02 cea-minilm-A-g02 ceg-gte-A-g02 --fold B
log "7. model: artifacts/models/$XGB (eval-30 macro F0.5 $($PY -c "import json; print(round(json.load(open('artifacts/models/$XGB.meta.json'))['eval']['f05'], 4))"))"

# 8. test prediction -> output/
run predict $PY scripts/predict.py --model $XGB --runs $TEST --out output
log "8. output/matching_results.tsv, output/candidate_pairs.tsv"

# 9. submission check with the official validator
if [ -n "${VALIDATOR:-}" ] || [ -d 6ab10eb3b23ba_student_resource ]; then
    run validate $PY scripts/validate_lowmem.py --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir dataset/test
    log "9. validator: $(tail -1 logs/validate.stdout)"
fi
log "==== pipeline finished ===="
