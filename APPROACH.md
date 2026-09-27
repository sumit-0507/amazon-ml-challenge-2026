# Business Entity Resolution — Approach, Round 1 to Round 4

**Team:** Minds&Machines (Sumit Agrawal, Gobinda Panda) · Amazon ML Challenge 2026

Final submission: **public leaderboard macro F0.5 0.980**, held-out eval **0.9841**
(pair precision 0.9984, recall 0.9663).

| Round | What changed | Eval F0.5 | Eval precision | Eval recall | **Leaderboard (test)** |
|---|---|---|---|---|---|
| 1 | TF-IDF blocking + XGBoost on 42 string / retrieval features | 0.9572 | 0.9926 | 0.9395 | **0.967** |
| 2 | + fine-tuned MiniLM cross-encoder (name \| address) | 0.9801 | 0.9980 | 0.9588 | **0.975** |
| 3 | + name-only and address-only cross-encoders, adaptive cross-encoder candidate set | 0.9826 | 0.9983 | 0.9630 | **0.978** |
| 4 | + gte-multilingual-reranker cross-encoder | **0.9841** | **0.9984** | **0.9663** | **0.980** |

Test precision / recall are not available: the test set has no labels and the leaderboard only
reports macro F0.5 on its public subset. Eval precision / recall are **pair level** (accepted
S2/S3 → S1 links); F0.5 is the **macro average per S1 entity**, exactly as the leaderboard
computes it.

---

## Contents

1. [Task and metric](#1-task-and-metric)
2. [Data facts that shaped the design](#2-data-facts-that-shaped-the-design)
3. [Evaluation setup](#3-evaluation-setup)
4. [Shared foundation: standardization and blocking](#4-shared-foundation-standardization-and-blocking)
5. [Round 1 — TF-IDF blocking + XGBoost](#5-round-1--tf-idf-blocking--xgboost)
6. [Round 2 — first cross-encoder](#6-round-2--first-cross-encoder)
7. [Round 3 — three cross-encoders and an adaptive scoring set](#7-round-3--three-cross-encoders-and-an-adaptive-scoring-set)
8. [Round 4 — a stronger reranker](#8-round-4--a-stronger-reranker)
9. [Round-by-round comparison](#9-round-by-round-comparison)
10. [Diagnostics and ideas that did not help](#10-diagnostics-and-ideas-that-did-not-help)
11. [Final submission and reproduction](#11-final-submission-and-reproduction)
12. [What we would do next](#12-what-we-would-do-next)

---

## 1. Task and metric

For every **Source 1 (S1)** business, list the records in **Source 2 (S2)** and **Source 3 (S3)**
that describe the same business.

- **Metric:** F0.5 = 1.25·P·R / (0.25·P + R), computed **per S1 entity** and averaged over all
  S1 entities (macro). Precision counts twice as much as recall.
- **Singletons count:** an S1 entity with no true match scores 1.0 if we predict nothing and
  0.0 if we predict anything. A false merge on a singleton costs its whole score.
- Test size: 1,732,544 S1 entities; 9,969,589 S2 + S3 records.

## 2. Data facts that shaped the design

| Fact (training data) | Consequence |
|---|---|
| S1 is deduplicated; **no S2/S3 record belongs to more than one S1 entity** | Query S1 with each S2/S3 record; keep at most one S1 per record |
| **26%** of S2/S3 records match no S1 entity | A probability threshold is needed, not "take the top hit" |
| A matched S1 entity has **3.46** S2/S3 records on average (max 11) | Group accepted records per S1 at the end |
| Country agrees in **100%** of true pairs; state in **99.35%** (when both known) | Country and state are safe blocking filters |
| ~**20%** of Indian S2/S3 names are in Indian scripts | Transliteration in name standardization |
| Heavy noise: aliases before `d/b/a`, look-alike digits (`T0tal`), honorifics, legal-form typos, website-style names, mojibake, reordered addresses | Rule-based standardization before anything else |
| **3.3%** of records have an empty address | Hardest records; the source of most remaining errors |
| Test has **France** (15% of S1), absent from training | No country-specific learned features; French rules in standardization |
| Train and test businesses **do not overlap** (0.00% of test S1 share a standardized name + address with train S1) | Nothing can be looked up from training; only patterns transfer |

## 3. Evaluation setup

- **Train / eval split by S1 entity** (`scripts/split_train_eval.py`, 95/5, seed 42): an entity
  and all its records land on the same side, so eval is scored per entity like the leaderboard.
  The S1 search index still holds all training S1 entities, so eval queries face every wrong
  candidate.
- **eval-30:** 30% of the eval entities: **153,951** S2/S3 records, **113,716** of them with a true
  match, ~33,100 S1 entities in the F0.5 average. All eval numbers in this document are on eval-30.
- **train-30:** 30% of the training entities, used to train XGBoost (fit + validation part for
  early stopping and threshold selection).
- **Cross-fitting (rounds 2–4):** training entities are split into folds **A** and **B** by a
  crc32 hash of the S1 entity id (unmatched records by their own id). Cross-encoders are trained
  on fold A only; XGBoost is trained and tuned on fold B only, so it only ever sees
  **out-of-fold** cross-encoder scores, the same situation as on test.
- **Gate:** each new round had to beat the previous round's eval F0.5 before any test GPU time
  was spent on it.

## 4. Shared foundation: standardization and blocking

Used unchanged in all four rounds.

### 4.1 Standardization (`utils/`)

**Names** (`name_standardization.py`):

| Step | Example |
|---|---|
| Drop the injected alias before `d/b/a`, `t/a`, `a/k/a` | `Zetajax t/a Ps Wisdom Pvt Ltd` → `ps wisdom pvt ltd` |
| Websites | `keetontrading.com` → `keetontrading` |
| Native script → English: word map learned from 551k matched training name pairs (1,347 words, 96.6% word precision), `anyascii` fallback | `श्री इंफोटेक प्रा. लि.` → `shree infotech pvt ltd` |
| Look-alike digits inside words | `T0tal` → `total` |
| Honorifics dropped | `M/s`, `Mr`, `Smt`, `Dr`, `Sri`, `Shri` |
| Legal forms (incl. edit-distance-1 typos) normalized and moved to the end | `Limted`, `Lmtd` → `ltd`; `S.A.R.L.` → `sarl` |

Effect on matched training pairs: identical names went from 2.7% → 51.8% (India) and
5.9% → 38.7% (US); native-script name similarity from 13.7 → 99.9.

**Addresses** (`address_standardization.py`): mojibake repair, placeholders (`null`, `N/A`),
street-type abbreviations (`Street`/`St` → `st`, `R.` → `rue`), ordinals (`10th`/`Tenth` → `10`),
Indian city aliases, house-number markers dropped, and **states in any form or script → one
code** (all 36 Indian states/UTs incl. local-language names, US states, French regions and
departments). Example: `KANSAS CITY, MO, 630 45ND TERRACE, null` and
`Missouri, 630 45th Terrace, Kansas City` → `630 45 ter, kansas, mo`.

### 4.2 Blocking (`scripts/gpu_retrieve.py`)

- Candidates must share the **country**, and the **state group** when the record has a
  recognized state (the state filter loses only 0.018% of true matches).
- Four **TF-IDF cosine searches** over S1, computed as sparse matrix products on the GPU; each
  scores `2 × name similarity + address similarity` and keeps its top 16:

| Search | Tokens | Recall alone (eval-30) |
|---|---|---|
| `words` | words | 98.86% |
| `tri` | character trigrams | 98.19% |
| `bi` | character bigrams | 97.77% |
| `phonetic` | Double Metaphone codes | 96.53% |

- **Reciprocal rank fusion** (k = 60): RRF(c) = Σ 1/(60 + rank in each search). All fused
  candidates are kept: **~40.5 per record** (max 64).

| True match within the fused top k | 1 | 2 | 3 | 5 | 10 | 20 | all |
|---|---|---|---|---|---|---|---|
| Recall (eval-30) | 96.27% | 97.43% | 97.95% | 98.41% | 98.87% | 99.21% | **99.37%** |

- **Test candidate set:** **405,956,744 pairs**; every S1 entity has candidates
  (`candidate_pairs.tsv`, identical in all rounds).

---

## 5. Round 1 — TF-IDF blocking + XGBoost

**Idea:** a gradient-boosted classifier over cheap retrieval and string-similarity features.

**Features (42):**
- Retrieval: per search score, rank, name score, address score, gap to the record's best; RRF
  score and gap; fused rank; number of searches that found the candidate; candidates per record.
- Names: RapidFuzz ratio, token-set, token-sort, partial ratio, Jaro-Winkler; the same on the
  **core name** (legal forms removed) plus exact core match; legal-form agreement; length difference;
  multilingual MiniLM embedding cosine.
- Addresses: ratio and token-set; Jaccard of the numbers; first number equal; empty address flags.

**Model:** XGBoost (`xgb-train30-full`), trained on 400k train-30 records with **all** their
candidates as negatives; early stopping on log-loss → **1,585 trees**. The threshold is swept on the
validation part for macro F0.5 → **0.81**. Each record keeps its single best candidate if p ≥ 0.81.

**Eval-30:**

| Metric | Value |
|---|---|
| Macro F0.5 | **0.9572** (baseline "always take the top fused candidate": 0.4255) |
| Pair precision / recall | 0.9926 / 0.9395 |
| Records accepted | 107,641 of 153,951 |

| Records with a true match (113,716) | Share |
|---|---|
| Correct and accepted | 93.95% |
| Right candidate, but p below the threshold | 4.08% |
| Wrong candidate on top | 1.34% |
| True match not retrieved | 0.63% |

Top features by gain: `words_gap`, `fused_rank`, `number_jaccard`, `rrf_gap`, `words_rank`,
`address_token_set`.

**Test:** 5,731,789 records matched (57.5%); 1,632,478 S1 entities with matches.
**Leaderboard: 0.967.**

**Lesson:** retrieval was already near its ceiling (99.37%); the losses were in the decision
(4.08% right-but-rejected). String similarities could not separate true matches from close
wrong ones confidently enough.

## 6. Round 2 — first cross-encoder

**Idea:** a transformer that reads both records together ("name | address" of the S2/S3 record
and of the S1 candidate) and outputs a match score, used as XGBoost features.

**Cross-encoder `ce-minilm-A`:**
- Base: `paraphrase-multilingual-MiniLM-L12-v2` (118M parameters, Apache-2.0, 50+ languages).
- Fine-tuned on **fold A**: for each record the fused top 5 candidates + 2 random lower-ranked
  ones; binary cross-entropy; 1 epoch; **11.2M pairs**; 2h46m on a 24 GB MIG slice.
- Validation: log-loss 0.0121, AUC-PR 0.9990, **top-1 accuracy 99.23%** (the fused retrieval
  ranking alone: 97.61%).
- Scored the **fused top 5** candidates of every record (eval, train fold B, test).

**XGBoost** `xgb-train30-ce-minilm-A`: 42 + 3 cross-encoder features (`ce_logit`, gap to the
record's best logit, rank) = **45 features**; fold B only; **545 trees**; threshold **0.91**.

**Eval-30:**

| Metric | Round 1 | **Round 2** |
|---|---|---|
| Macro F0.5 | 0.9572 | **0.9801** |
| Pair precision | 0.9926 | 0.9980 |
| Pair recall | 0.9395 | 0.9588 |
| Records accepted | 107,641 | 109,252 |
| Correct and accepted | 93.95% | 95.88% |
| Right but rejected | 4.08% | 2.23% |
| Wrong candidate on top | 1.34% | 1.27% |
| Not retrieved | 0.63% | 0.63% |

The three cross-encoder features took the top 3 places by gain (`ce_logit` 10,577,
`ce_rank` 9,099, `ce_gap` 750; the next feature 144).

**Test:** 5,805,624 records matched (58.2%); 1,633,619 S1 entities with matches. The file differs
from round 1 for 257,931 S1 entities (15%). **Leaderboard: 0.975.**

**Lesson:** the eval gain (+2.29 points, errors −53%) transferred only partly to the leaderboard
(+0.8, errors −24%). France (never seen in training) changed the most (5.9% of French records vs
2.5–2.7% for India / US), and eval-30 contains no French records.

## 7. Round 3 — three cross-encoders and an adaptive scoring set

**Ideas:**

1. **Two more cross-encoders** with separate views, so XGBoost gets separate "names match" and
   "addresses match" signals (useful for empty addresses and for same-address/different-name
   records):
   - `cen-minilm-A`: **name only** (valid log-loss 0.1157, AUC-PR 0.8984, top-1 95.23%)
   - `cea-minilm-A`: **address only**, records without an address skipped (log-loss 0.0464,
     AUC-PR 0.9660, top-1 98.94%)
   - both MiniLM-L12, fold A, same settings as round 2.
2. **Adaptive cross-encoder candidate set.** Instead of the fused top 5, score every candidate
   whose score in **some search is within δ = 0.2 of that search's own #1**. Easy records (all
   searches agree on one clear winner) get one pair; hard ones (identical-name siblings) get all
   close contenders.

| Cross-encoder candidate set (eval-30) | Recall of true matches | Pairs per record |
|---|---|---|
| Fused top 5 (round 2) | 98.41% | 5.00 |
| Fused top 10 | 98.87% | 10.00 |
| Each search's top 5 | 99.01% | 12.01 |
| **Within δ = 0.2 of a search's #1 (round 3)** | **99.25%** | **3.36** |
| All fused candidates | 99.37% | 40.5 |

   Higher recall with fewer pairs than the fixed top 5. On test the rule selects 3.9 pairs per
   record, and **France gets the most attention** (5.5–5.8 pairs per record vs 2.6 for the US).

All three cross-encoders (round 2's re-scored) score the δ = 0.2 set. **XGBoost**
`xgb-train30-ce3g-minilm-A`: 42 + 3 × 3 = **51 features**; **488 trees**; threshold **0.91**.

**Eval-30:**

| Metric | Round 2 | **Round 3** |
|---|---|---|
| Macro F0.5 | 0.9801 | **0.9826** |
| Pair precision | 0.9980 | 0.9983 |
| Pair recall | 0.9588 | 0.9630 |
| Records accepted | 109,252 | 109,701 |
| Correct and accepted | 95.88% | 96.30% |
| Right but rejected | 2.23% | 1.88% |
| Wrong candidate on top | 1.27% | 1.19% |
| Not retrieved | 0.63% | 0.63% |

**Test:** 5,802,183 records matched (58.2%); 1,633,047 S1 entities with matches.
**Leaderboard: 0.978.** This time the eval gain (+0.25) transferred fully (+0.3), plausibly because
the adaptive set helped France, which eval cannot measure.

## 8. Round 4 — a stronger reranker

**Idea:** replace capacity, not just add views: a larger multilingual model pre-trained for
reranking.

**Cross-encoder `ceg-gte-A`:**
- Base: `Alibaba-NLP/gte-multilingual-reranker-base` (306M parameters, 2.6× MiniLM, Apache-2.0),
  "name | address", fold A, **7.0M pairs** (250k records per training part), learning rate 2e-5,
  1 epoch, 2h32m on a 48 GB MIG slice.
- Engineering fixes: the model ships custom code and an **fp16** checkpoint. With
  transformers 5, (a) its rotary and position buffers were left uninitialized (fixed by rebuilding
  them after loading) and (b) fp16 weights overflowed to **NaN** in the first training steps
  (fixed by loading in fp32 and training under bf16 autocast).
- Validation: log-loss **0.0114**, AUC-PR **0.9992**, top-1 99.22%. At step 10,000 it was already
  ahead of MiniLM at the same point (log-loss 0.0164 vs 0.0238).
- Scores the same δ = 0.2 set as the MiniLM models.

**XGBoost** `xgb-train30-ce4-A`: 42 + 4 × 3 = **54 features**; **339 trees**; threshold **0.90**.
The new model dominates the decision:

| Top features by gain (round 4) | Gain |
|---|---|
| `ceg_logit` (gte) | 19,146 |
| `ceg_rank` | 2,667 |
| `ce_logit` (MiniLM name \| address) | 1,732 |
| `ce_rank` | 332 |
| `cen_logit` (name only) | 128 |
| `cea_logit` (address only) | 32 |
| `fused_rank` | 27 |

**Eval-30:**

| Metric | Round 3 | **Round 4** |
|---|---|---|
| Macro F0.5 | 0.9826 | **0.9841** |
| Pair precision | 0.9983 | **0.9984** |
| Pair recall | 0.9630 | **0.9663** |
| Records accepted | 109,701 | 110,062 |
| Correct and accepted | 96.30% | **96.63%** |
| Right but rejected | 1.88% | **1.55%** |
| Wrong candidate on top | 1.19% | 1.19% |
| Not retrieved | 0.63% | 0.63% |

**Test:** 5,787,102 records matched (58.0%); 1,632,188 S1 entities with matches.
**Leaderboard: 0.980** (eval +0.15 → leaderboard +0.2, transferred fully again).

---

## 9. Round-by-round comparison

### 9.1 Eval-30 (held out, labelled)

| | Round 1 | Round 2 | Round 3 | **Round 4** |
|---|---|---|---|---|
| Macro F0.5 | 0.9572 | 0.9801 | 0.9826 | **0.9841** |
| Error (1 − F0.5) | 4.28% | 1.99% | 1.74% | **1.59%** |
| Pair precision | 0.9926 | 0.9980 | 0.9983 | **0.9984** |
| Pair recall | 0.9395 | 0.9588 | 0.9630 | **0.9663** |
| Validation F0.5 (train-30, fold B) | 0.9578 | 0.9815 | 0.9839 | 0.9849 |
| Records accepted (of 153,951) | 107,641 | 109,252 | 109,701 | 110,062 |
| Correct and accepted | 93.95% | 95.88% | 96.30% | **96.63%** |
| Right but rejected | 4.08% | 2.23% | 1.88% | **1.55%** |
| Wrong candidate on top | 1.34% | 1.27% | 1.19% | 1.19% |
| Not retrieved | 0.63% | 0.63% | 0.63% | 0.63% |

### 9.2 Model

| | Round 1 | Round 2 | Round 3 | **Round 4** |
|---|---|---|---|---|
| Cross-encoders | — | MiniLM (name \| address) | + MiniLM name, MiniLM address | + gte reranker |
| Cross-encoder candidate set | — | fused top 5 | within 0.2 of a search's #1 | within 0.2 |
| Features | 42 | 45 | 51 | 54 |
| XGBoost trees | 1,585 | 545 | 488 | 339 |
| Threshold | 0.81 | 0.91 | 0.91 | 0.90 |

### 9.3 Test (unlabelled) and leaderboard

| | Round 1 | Round 2 | Round 3 | **Round 4** |
|---|---|---|---|---|
| **Leaderboard macro F0.5** | 0.967 | 0.975 | 0.978 | **0.980** |
| Leaderboard error | 3.3% | 2.5% | 2.2% | **2.0%** |
| Eval gain → leaderboard gain | — | +2.29 → +0.8 | +0.25 → +0.3 | +0.15 → +0.2 |
| Records matched (of 9,969,388) | 5,731,789 | 5,805,624 | 5,802,183 | 5,787,102 |
| S1 entities with matches (of 1,732,544) | 1,632,478 | 1,633,619 | 1,633,047 | 1,632,188 |
| Candidate pairs | 405,956,744 | 405,956,744 | 405,956,744 | 405,956,744 |
| `matching_results.tsv` md5 | `3d39866b…` | `96979393…` | `1294a2ef…` | **`d5a2c19e…`** |

---

## 10. Diagnostics and ideas that did not help

Every idea was measured before spending test GPU time.

| Idea | Result | Decision |
|---|---|---|
| **France diagnostic submission** (round 3 with all French rows emptied) | Leaderboard 0.848 → French F0.5 ≈ 0.95–0.97, US + India ≈ 0.98 | France is not the main gap; no French-specific work |
| **Phonetic variants**: full-length Double Metaphone, Indian phonetic key, both | 71 → 71 / 74 / 71 true matches not found (of 14,773) | Kept 4-character Double Metaphone |
| **Skip-bigram search** | +4 unique true matches of 14,773 (+0.03% recall) | Not worth a full retrieval re-run |
| **All 7 searches together** | 71 → 64 not found (+0.05%) | Same |
| **Perfect search, estimated** | ≤ +0.24 eval points (73–81% of search misses are empty-address, garbled-name records) | The decision stage is where the error is |
| **Each search's top 5 as the cross-encoder set** | 99.01% recall at 12 pairs / record vs 99.25% at 3.36 for δ = 0.2 | δ = 0.2 |
| **Per-S1-entity expected-F0.5 decision** (instead of one threshold) | Held-out half: 0.9808 vs 0.9832 (−0.0024) | Kept the single threshold |
| **Separate thresholds** for records with / without an address | Held-out half: 0.9831 vs 0.9832 (−0.0001) | Kept the single threshold |
| **LightGBM instead of XGBoost** | Comparison stopped before a result (both are gradient-boosted trees; the cross-encoder features carry the decision) | Kept XGBoost |
| **Alias lookup from training data** | 0.00% of test businesses appear in training (by standardized name + address) | Not applicable |

**Where the remaining eval errors are (round 2 analysis, same pattern later):**
- **Missed matches:** 59% have an **empty address**. 56% of those have an S1 candidate with
  **exactly the same name** as the true match (unresolvable without an address; rejecting is right
  under F0.5); the rest have heavily garbled names (`STERLING FUGRCRERET` vs `Sterling Futurecrest`).
  Next: records whose name is completely different from S1 while the address is identical
  (`Novimira` vs `Tss Foods Pvt Ltd`).
- **False merges** (precision 0.998, so rare): ~40% are empty-address records merged onto one of
  several identical-name S1 entities; single-word names are the next group (~16%).
- **Single-word / website-style names** (6.8% of records) are missed 4× as often as others
  (5.9% vs 1.5%). In training, 2.85% of true pairs have very different names, and 96.5% of those
  share the address (`Universal Logistics Private Limited` ↔ `ulprivate.com`).

## 11. Final submission and reproduction

- Leaderboard file: round 4 `matching_results.tsv` (md5 `d5a2c19ebf62ae524874031d32e0571e`),
  passed the official validator.
- Package `Minds&Machines_submission.zip`: `output/` (both files), `code/business_entity_resolution/`
  (`src/run_pipeline.sh`, sources, `README.md`, pinned `requirements.txt`), methodology document.
- Reproduce end to end: `bash run_pipeline.sh` (repository root) runs split → word map →
  embeddings → blocking → 4 cross-encoders → scoring → XGBoost → prediction → validator; every step
  resumes from finished work. A fresh copy reproduced the split and word map byte for byte.
- Models: paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0), gte-multilingual-reranker-base
  (Apache-2.0), XGBoost (Apache-2.0). No external data, APIs or lookups.

Hardware: NVIDIA RTX PRO 6000 Blackwell MIG slices (24 GB and 48 GB). Approximate one-slice
times: blocking ~4 h, cross-encoder training ~9 h (all four), scoring ~8 h, XGBoost ~15 min,
test prediction ~3 h.

## 12. What we would do next

In order of expected gain for the remaining ~2% leaderboard error:

1. **A fine-tuned LLM matcher** (≤ 8B, Apache-2.0, e.g. Qwen2.5-7B with LoRA) scoring only the
   uncertain records (~10% of test). Estimated ~10 h on one H100.
2. **Collective (sibling) features:** each business has ~3.7 S2/S3 records; use a record's
   similarity to the records already confidently matched to the same S1. Targets "right but
   rejected" (1.55%).
3. **Acronym / website-name features** (`ulprivate.com` ↔ `Universal Logistics Private`) for the
   single-word names that are missed 4× as often.
4. **Full training data** for the cross-encoders (we used about a third) and XGBoost, and fold-B
   cross-encoders averaged with fold A on test.
