# Business Entity Resolution — Amazon ML Challenge 2026

**Team:** Minds&Machines

| Role | Name |
|---|---|
| Team leader | Sumit Agrawal |
| Member | Gobinda Panda |

For every business in **Source 1**, find all records in **Source 2** and **Source 3** that
describe the same business, across noisy names, messy addresses, Indian scripts and three
countries (US, India, France). Scored by **macro F0.5** per Source 1 entity (precision weighted
twice as much as recall).

**Result:** public leaderboard **0.980**, held-out eval **0.9841** (pair precision 0.9984,
recall 0.9663).

| Round | Change | Eval F0.5 | Leaderboard |
|---|---|---|---|
| 1 | TF-IDF blocking + XGBoost | 0.9572 | 0.967 |
| 2 | + MiniLM cross-encoder | 0.9801 | 0.975 |
| 3 | + name-only / address-only cross-encoders, adaptive candidate set | 0.9826 | 0.978 |
| 4 | + gte reranker cross-encoder (**final**) | **0.9841** | **0.980** |

The full round-by-round write-up (metrics, error analysis, what did and didn't work) is in
[APPROACH.md](APPROACH.md).

## How it works

```
raw TSVs ─► standardize names & addresses ─► blocking: 4 GPU TF-IDF searches (bigrams, trigrams,
            (native scripts → English,          words, Double Metaphone) within country + state,
             legal forms, states → codes)       RRF fusion, ~40 candidates per record, 99.37% recall
                                                                    │
                  4 fine-tuned multilingual cross-encoders ◄────────┘
                  (name|address, name, address: MiniLM; name|address: gte reranker)
                  score the candidates within 0.2 of a search's best score
                                                                    │
                  XGBoost on 54 features ─► best candidate per record if p ≥ 0.90
                                                                    │
                  output/matching_results.tsv + output/candidate_pairs.tsv
```

Cross-encoders are trained on one half of the training entities (fold A) and XGBoost on the
other (fold B), so XGBoost only sees out-of-fold scores, as on test. All models are MIT /
Apache-2.0; no external data or lookups are used.

## Repository layout

| Path | Contents |
|---|---|
| `run_pipeline.sh` | **Entry point**: raw data → final submission files, end to end |
| `utils/name_standardization.py` | Business-name normalization (aliases, websites, transliteration, legal forms) |
| `utils/address_standardization.py` | Address normalization, state codes for US / India / France |
| `scripts/split_train_eval.py` | 95/5 train/eval split by Source 1 entity |
| `scripts/embed_names.py` | Multilingual name embeddings (one feature) |
| `scripts/gpu_retrieve.py` | Blocking: TF-IDF cosine searches on the GPU + RRF fusion |
| `scripts/retrieve_candidates.py` | Shared helpers (query sets, fusion, output schema, logging) |
| `scripts/train_cross_encoder.py` | Fine-tunes a cross-encoder (`--text both / name / address`) |
| `scripts/score_cross_encoder.py` | Scores candidate pairs (`--gap` adaptive set) |
| `scripts/train_xgb.py` | Features, XGBoost training, threshold selection, eval scoring |
| `scripts/predict.py` | Test prediction → `matching_results.tsv`, `candidate_pairs.tsv` |
| `scripts/validate_lowmem.py` | Official validator + streamed checks of the 5 GB candidate file |
| `scripts/analyze_missed.py`, `s1_decision.py`, `build_word_lookup.py` | Error analysis and experiments |
| `scripts/run_round3.sh`, `run_cross_encoder.sh`, `run_test.sh`, `slurm/` | Cluster run scripts used during the challenge |
| `scripts/index_opensearch.py`, `opensearch_server.sh` | Early OpenSearch-based blocking (replaced by `gpu_retrieve.py`) |
| `requirements.txt` | Pinned environment |
| `APPROACH.md` | Detailed approach and results, round 1 to round 4 |

## How to run

### 1. Environment

Python 3.12, one CUDA GPU with at least 24 GB, about 48 GB RAM and 150 GB of free disk.

```bash
python3.12 -m venv venv && . venv/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130   # CUDA build for your GPU
pip install -r requirements.txt
```

### 2. Data

Place (or symlink) the challenge data in `dataset/` at the repository root:

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

### 3. Run

```bash
PY=python VALIDATOR=/path/to/validate_submission.py bash run_pipeline.sh
```

Outputs: `output/matching_results.tsv` (upload to the leaderboard) and
`output/candidate_pairs.tsv`. Progress goes to `logs/run_pipeline.log`, per-step logs to `logs/`.

- `VALIDATOR` points to the challenge's `validate_submission.py`; without it the final check is skipped.
- `STEP` prefixes every step, e.g. `STEP="srun --jobid=123 --overlap --gres=gpu:1"` on Slurm.
- Every step skips finished work (cross-encoder training resumes from its checkpoint), so an
  interrupted run continues when started again.
- Step 0 downloads the pretrained models to the Hugging Face cache (`HF_HOME` to choose where);
  later steps run offline.

| Step | What | Time (one GPU slice) |
|---|---|---|
| 0–2 | models, split, native-script word map | minutes |
| 3 | name embeddings | ~1 h |
| 4 | blocking (eval, train, test) | ~4 h |
| 5 | 4 cross-encoders | ~9 h |
| 6 | cross-encoder scoring | ~8 h |
| 7 | XGBoost | ~15 min |
| 8 | test prediction | ~3 h |
| 9 | validation | minutes |

### Individual steps

```bash
python scripts/split_train_eval.py                                   # dataset/splits/
python utils/name_standardization.py --build-word-map                # artifacts/native_word_map.json
python scripts/gpu_retrieve.py --queries eval --frac 0.3 --name gpu-tfidf-eval-30
python scripts/train_cross_encoder.py --name cen-minilm-A --fold A --text name --train-queries 400000
python scripts/score_cross_encoder.py --models cen-minilm-A --name cen-minilm-A-g02 --run gpu-tfidf-eval-30 --gap 0.2
python scripts/analyze_missed.py --model xgb-train30-ce4-A           # error breakdown on eval
```

See `run_pipeline.sh` for the exact sequence and arguments of the final submission.

## Models

| Model | License | Size | Use |
|---|---|---|---|
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Apache-2.0 | 118M | name embeddings; base of 3 cross-encoders |
| Alibaba-NLP/gte-multilingual-reranker-base | Apache-2.0 | 306M | base of the 4th cross-encoder |
| XGBoost | Apache-2.0 | — | final classifier |
