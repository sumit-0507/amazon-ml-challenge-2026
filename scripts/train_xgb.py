"""Train the XGBoost matcher on retrieved candidates and score it with the leaderboard metric.

Input: candidate runs from gpu_retrieve.py (or retrieve_candidates.py) in
artifacts/candidates/: <name>.parquet (fused candidates with per-search scores)
and <name>.queries.parquet.

Features per (S2/S3 record, S1 candidate):
    retrieval   per search (bi, tri, words, phonetic): combined / name / address
                score and rank; RRF, fused rank, number of searches that found it;
                gap to the query's best score per search and for RRF; candidates
                in the query
    strings     rapidfuzz on standardized text: name ratio / token_set / token_sort
                / partial / Jaro-Winkler, core-name ratio and token_set, address
                ratio and token_set; address-number Jaccard and first-number match;
                same legal form; length differences; empty-address flags
    embedding   cosine of the name embeddings (artifacts/embeddings), if present
No country feature: test has a country (France) that training does not.

Training rows: all positives + the hardest --hard-neg negatives (by fused rank)
+ --rand-neg random negatives per query, from --train-queries queries; with
--all-negatives, every candidate of each fit query instead (same distribution as
validation and inference, so early stopping on log-loss is meaningful). 10% of the
training S1 entities (and of the unmatched records) are held out as validation
(at most --valid-queries queries, all their candidates): early stopping on AUC-PR,
and the decision threshold is chosen there. Only the used queries' candidates are
read from disk.

Decision: each S2/S3 record takes its most probable candidate if p >= threshold
(a record matches at most one S1 entity). Score: the README's macro F0.5 over S1
entities, singletons included (1.0 for an empty prediction, 0.0 otherwise).

    srun --jobid=<id> --overlap --gres=gpu:2g.48gb:1 --cpus-per-task=12 \\
        python scripts/train_xgb.py --train gpu-tfidf-train-30 --eval gpu-tfidf-eval-30
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "scripts"))
import retrieve_candidates as rc  # noqa: E402  (logging, paths)
from address_standardization import standardize_addresses  # noqa: E402
from name_standardization import LEGAL_FORMS, standardize_names  # noqa: E402

CAND = ROOT / "artifacts" / "candidates"
EMB = ROOT / "artifacts" / "embeddings"
MODELS = ROOT / "artifacts" / "models"
FEATURES = ROOT / "artifacts" / "features"
DATA = ROOT / "dataset"
SEARCHES = ["bi", "tri", "words", "phonetic"]
GROUND_TRUTH = {"train": DATA / "splits" / "ground_truth_train.tsv", "eval": DATA / "splits" / "ground_truth_eval.tsv"}
log = rc.log
_NUMBER = re.compile(r"\d+")

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_queries(name):
    return pd.read_parquet(CAND / f"{name}.queries.parquet")


CANDIDATE_COLUMNS = ["query_row", "s1_id", "rrf", "fused_rank", "label"] + [
    f"{m}_{c}" for m in SEARCHES for c in ("score", "rank", "name_score", "address_score")]
FEATURE_CHUNK = 1_000_000  # rows per chunk for string/embedding features (bounds peak memory)


def load_candidates(name, query_rows=None):
    """Fused candidates of a run (needed columns only); only the given query rows if provided."""
    filters = None if query_rows is None else [("query_row", "in", np.asarray(query_rows, dtype=np.int32))]
    return pd.read_parquet(CAND / f"{name}.parquet", columns=CANDIDATE_COLUMNS, filters=filters
                           ).reset_index(drop=True)


def query_split(run_name):
    return "eval" if "eval" in run_name else "train"


class Texts:
    """Standardized name / core name / address for S1 records and query records."""

    def __init__(self, n_jobs, s1_split="train"):
        self.n_jobs = n_jobs
        s1 = pd.read_csv(DATA / s1_split / f"{s1_split}_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
        self.s1 = self._standardize(s1).set_index(s1.entity_id)
        self.s1_row = pd.Series(np.arange(len(s1)), index=s1.entity_id)

    def _standardize(self, df):
        return pd.DataFrame({
            "name": standardize_names(df.business_name, n_jobs=self.n_jobs),
            "core": standardize_names(df.business_name, drop_legal=True, n_jobs=self.n_jobs),
            "address": standardize_addresses(df.business_address, df.country, n_jobs=self.n_jobs),
        })

    def queries(self, split, file_rows):
        files, _ = rc.QUERY_SETS[split]
        df = pd.concat([pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False) for f in files],
                       ignore_index=True).iloc[file_rows]
        return self._standardize(df.reset_index(drop=True))

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def _numbers(texts):
    return [set(_NUMBER.findall(t)) for t in texts]


def _legal(texts):
    return [frozenset(w for w in t.split() if w in LEGAL_FORMS) for t in texts]


def build_features(cand, qtext, s1text, q_emb, s1_emb, s1_row):
    """cand: candidate rows (query_row, s1_id, scores...); qtext: per query_row; s1text: indexed by s1_id.
    Retrieval features are computed on the whole frame; string and embedding features in chunks."""
    f = _retrieval_features(cand)
    parts = [_pair_features(cand.iloc[i:i + FEATURE_CHUNK], qtext, s1text, q_emb, s1_emb, s1_row)
             for i in range(0, len(cand), FEATURE_CHUNK)]
    pair = pd.concat(parts)
    del parts
    gc.collect()
    return pd.concat([f, pair], axis=1)


def _retrieval_features(cand):
    f = pd.DataFrame(index=cand.index)
    for m in SEARCHES:
        for col in (f"{m}_score", f"{m}_rank", f"{m}_name_score", f"{m}_address_score"):
            f[col] = cand[col].astype(np.float32)
        best = cand.groupby("query_row")[f"{m}_score"].transform("max")
        f[f"{m}_gap"] = (cand[f"{m}_score"] - best).astype(np.float32)
    f["rrf"] = cand.rrf.astype(np.float32)
    f["rrf_gap"] = (cand.rrf - cand.groupby("query_row").rrf.transform("max")).astype(np.float32)
    f["fused_rank"] = cand.fused_rank.astype(np.float32)
    f["n_searches"] = cand[[f"{m}_rank" for m in SEARCHES]].notna().sum(axis=1).astype(np.float32)
    f["n_candidates"] = cand.groupby("query_row").s1_id.transform("size").astype(np.float32)
    return f


def _pair_features(cand, qtext, s1text, q_emb, s1_emb, s1_row):
    f = pd.DataFrame(index=cand.index)
    q = qtext.iloc[cand.query_row.to_numpy()]
    s = s1text.loc[cand.s1_id.to_numpy()]
    qn, sn = q.name.tolist(), s.name.tolist()
    qc, sc = q.core.tolist(), s.core.tolist()
    qa, sa = q.address.tolist(), s.address.tolist()

    def pair(a, b, scorer):
        return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)

    f["name_ratio"] = pair(qn, sn, fuzz.ratio)
    f["name_token_set"] = pair(qn, sn, fuzz.token_set_ratio)
    f["name_token_sort"] = pair(qn, sn, fuzz.token_sort_ratio)
    f["name_partial"] = pair(qn, sn, fuzz.partial_ratio)
    f["name_jaro_winkler"] = pair(qn, sn, JaroWinkler.normalized_similarity)
    f["core_ratio"] = pair(qc, sc, fuzz.ratio)
    f["core_token_set"] = pair(qc, sc, fuzz.token_set_ratio)
    f["core_exact"] = (np.array(qc, dtype=object) == np.array(sc, dtype=object)).astype(np.float32)
    f["address_ratio"] = pair(qa, sa, fuzz.ratio)
    f["address_token_set"] = pair(qa, sa, fuzz.token_set_ratio)
    qnum, snum = _numbers(qa), _numbers(sa)
    f["number_jaccard"] = np.array([len(x & y) / len(x | y) if (x or y) else np.nan for x, y in zip(qnum, snum)],
                                   dtype=np.float32)
    qfirst = [_NUMBER.search(t) for t in qa]
    sfirst = [_NUMBER.search(t) for t in sa]
    f["first_number_equal"] = np.array([np.nan if (x is None or y is None) else float(x.group() == y.group())
                                        for x, y in zip(qfirst, sfirst)], dtype=np.float32)
    f["legal_equal"] = np.array([float(x == y) for x, y in zip(_legal(qn), _legal(sn))], dtype=np.float32)
    f["name_len_diff"] = np.abs(np.array([len(x) for x in qn]) - np.array([len(x) for x in sn])).astype(np.float32)
    f["query_address_empty"] = np.array([not x for x in qa], dtype=np.float32)
    f["s1_address_empty"] = np.array([not x for x in sa], dtype=np.float32)
    if q_emb is not None:
        qe = q_emb[cand.query_row.to_numpy()]
        se = s1_emb[s1_row.loc[cand.s1_id.to_numpy()].to_numpy()]
        f["embedding_cosine"] = np.einsum("ij,ij->i", qe.astype(np.float32), se.astype(np.float32))
    return f


def cached_features(key, cand, build):
    """Features for cand from artifacts/features/<key>.parquet, built and saved if missing."""
    path = FEATURES / f"{key}.parquet"
    if path.exists():
        X = pd.read_parquet(path)
        if len(X) == len(cand):
            log.info(f"  features loaded from {path}")
            X.index = cand.index
            return X
    X = build()
    FEATURES.mkdir(parents=True, exist_ok=True)
    X.reset_index(drop=True).to_parquet(path, index=False)
    log.info(f"  features saved to {path}")
    return X


CE_SCORES = ROOT / "artifacts" / "ce_scores"


def ce_scores(ce, run, row_range=None):
    """Cross-encoder logits of a run (score_cross_encoder.py): query_row, s1_id, ce_logit."""
    path = CE_SCORES / ce / run
    if not (path / "_done").exists():
        sys.exit(f"cross-encoder scores missing or incomplete: {path}")
    filters = None if row_range is None else [("query_row", ">=", int(row_range[0])),
                                              ("query_row", "<=", int(row_range[1]))]
    return pd.read_parquet(path, columns=["query_row", "s1_id", "ce_logit"], filters=filters)


def add_ce_features(X, cand, scores, prefix="ce"):
    """Cross-encoder logit, gap to the query's best logit and rank among the query's scored
    candidates (NaN for candidates below the cross-encoder's fused-rank cut or not scored)."""
    logit = cand[["query_row", "s1_id"]].merge(scores, on=["query_row", "s1_id"], how="left").ce_logit.to_numpy(np.float32)
    by_query = pd.Series(logit).groupby(cand.query_row.to_numpy())
    X[f"{prefix}_logit"] = logit
    X[f"{prefix}_gap"] = (logit - by_query.transform("max").to_numpy()).astype(np.float32)
    X[f"{prefix}_rank"] = by_query.rank(ascending=False, method="first").to_numpy(np.float32)
    return X


CE_PREFIX = {"both": "ce", "name": "cen", "address": "cea"}


def ce_names(value):
    """Cross-encoder score sets of a model: meta 'ce' is None, one name (older models) or a list."""
    return [] if not value else [value] if isinstance(value, str) else list(value)


def ce_prefix(ce):
    """Feature prefix of a score set, by the text its cross-encoder reads (name | address -> ce,
    name -> cen, address -> cea). The model is the one named in the set's _done files, else the
    model folder of the set's own name; neither found: name | address."""
    model = ce
    for done in sorted((CE_SCORES / ce).glob("*/_done")):
        model = (json.loads(done.read_text()).get("models") or [ce])[0]
        break
    meta = ROOT / "artifacts" / "cross_encoders" / model / "meta.json"
    mode = json.loads(meta.read_text()).get("text_mode", "both") if meta.exists() else "both"
    return CE_PREFIX[mode]


def add_all_ce_features(X, cand, ces, run, row_range=None):
    """Features of every cross-encoder score set in `ces` (see ce_names) for candidates of `run`."""
    prefixes = [ce_prefix(ce) for ce in ce_names(ces)]
    if len(set(prefixes)) < len(prefixes):
        sys.exit(f"cross-encoder score sets {ce_names(ces)} share a feature prefix ({prefixes})")
    for ce, prefix in zip(ce_names(ces), prefixes):
        X = add_ce_features(X, cand, ce_scores(ce, run, row_range), prefix)
    return X


def query_embeddings(split, queries):
    path = EMB / f"s23-{split}.npy"
    if not path.exists():
        return None
    ids = np.load(EMB / f"s23-{split}.ids.npy")
    if not np.array_equal(ids[queries.file_row.to_numpy()], queries.entity_id.to_numpy(dtype=str)):
        log.info(f"  {path.name} does not line up with the queries; embedding feature skipped")
        return None
    return np.load(path, mmap_mode="r")[queries.file_row.to_numpy()]

# --------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------


def macro_f05(queries, best_s1, accept, gt_singletons, rng):
    """README metric on the sampled queries: per S1 entity F0.5, averaged over the S1 entities that are
    true targets of sampled queries or receive a prediction, plus a proportional sample of singletons."""
    truth = queries.source1_entity_id.to_numpy()
    pred = np.where(accept, best_s1, "")
    true_sets, pred_sets = {}, {}
    for q, (t, p) in enumerate(zip(truth, pred)):
        if t:
            true_sets.setdefault(t, set()).add(q)
        if p:
            pred_sets.setdefault(p, set()).add(q)
    n_single = int(round(len(true_sets) * gt_singletons["ratio"]))
    singles = rng.choice(gt_singletons["ids"], size=min(n_single, len(gt_singletons["ids"])), replace=False)
    entities = set(true_sets) | set(pred_sets) | set(singles)
    scores = []
    for e in entities:
        T, P = true_sets.get(e, set()), pred_sets.get(e, set())
        if not T:
            scores.append(1.0 if not P else 0.0)
        elif not P:
            scores.append(0.0)
        else:
            tp = len(T & P)
            prec, rec = tp / len(P), tp / len(T)
            scores.append(0.0 if tp == 0 else 1.25 * prec * rec / (0.25 * prec + rec))
    return float(np.mean(scores)), len(entities)


def singleton_info(split):
    gt = pd.read_csv(GROUND_TRUTH[split], sep="\t", dtype=str, keep_default_na=False)
    empty = gt.matched_entity_ids == ""
    return {"ids": gt.source1_entity_id[empty].to_numpy(), "ratio": empty.sum() / max((~empty).sum(), 1)}


def best_candidates(cand, prob):
    """Most probable candidate per query_row -> (query_rows, s1_ids, probs)."""
    order = np.lexsort((-prob, cand.query_row.to_numpy()))
    rows = cand.query_row.to_numpy()[order]
    first = np.r_[True, rows[1:] != rows[:-1]]
    return rows[first], cand.s1_id.to_numpy()[order][first], prob[order][first]


def evaluate(cand, queries, prob, threshold, gt_singletons, seed=0):
    rows, s1, p = best_candidates(cand, prob)
    best_s1 = np.full(len(queries), "", dtype=object)
    best_p = np.zeros(len(queries))
    best_s1[rows], best_p[rows] = s1, p
    f05, n_entities = macro_f05(queries, best_s1, best_p >= threshold, gt_singletons, np.random.default_rng(seed))
    truth = queries.source1_entity_id.to_numpy()
    accepted = best_p >= threshold
    correct = accepted & (best_s1 == truth) & (truth != "")
    return {"f05": f05, "entities": n_entities,
            "pair_precision": float(correct.sum() / max(accepted.sum(), 1)),
            "pair_recall": float(correct.sum() / max((truth != "").sum(), 1)),
            "accepted": int(accepted.sum()), "queries": len(queries)}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="candidate run used for training (and validation)")
    ap.add_argument("--eval", required=True, help="candidate run used for the final score")
    ap.add_argument("--name", help="model name (default: xgb-<train run>)")
    ap.add_argument("--train-queries", type=int, default=1_000_000, help="training queries used for fitting")
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--valid-queries", type=int, default=200_000, help="cap on validation queries")
    ap.add_argument("--all-negatives", action="store_true", help="train on full candidate lists (no sampling)")
    ap.add_argument("--f05-every", type=int, default=100, help="log validation macro F0.5 every N rounds")
    ap.add_argument("--hard-neg", type=int, default=10)
    ap.add_argument("--rand-neg", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--early-stop", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--min-child-weight", type=float, default=5)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ce", nargs="+", help="cross-encoder score sets in artifacts/ce_scores/ (each adds "
                    "<prefix>_logit, _gap, _rank; prefix ce / cen / cea for name | address / name / address)")
    ap.add_argument("--fold", choices=["A", "B"], help="fit and validate on this cross-encoder fold only (the "
                    "fold the cross-encoder was not trained on)")
    args = ap.parse_args()
    name = args.name or f"xgb-{args.train}"
    rc.setup_logging(f"train-{name}")
    n_jobs = len(os.sched_getaffinity(0))
    rng = np.random.default_rng(args.seed)
    log.info(f"train {name}: train run {args.train}, eval run {args.eval}, "
             f"params {json.dumps({k: v for k, v in vars(args).items() if k not in ('train', 'eval', 'name')})}")

    t0 = time.time()
    texts = Texts(n_jobs)
    s1_emb = np.load(EMB / "s1-train.npy", mmap_mode="r") if (EMB / "s1-train.npy").exists() else None
    log.info(f"S1 texts standardized in {time.time() - t0:.0f}s")

    # ---- training run: split queries into fit / valid by S1 entity
    queries = load_queries(args.train)
    truth = queries.source1_entity_id.to_numpy()
    entities = pd.unique(truth[truth != ""])
    valid_entities = set(entities[rng.random(len(entities)) < args.valid_frac])
    is_valid = np.where(truth != "", pd.Series(truth).isin(valid_entities), rng.random(len(queries)) < args.valid_frac)
    in_fold = np.ones(len(queries), dtype=bool)
    if args.fold:  # only records the cross-encoder did not train on
        import train_cross_encoder as C
        in_fold = C.folds(queries) == args.fold
        log.info(f"fold {args.fold}: {in_fold.sum():,} of {len(queries):,} queries")
    fit_rows = np.flatnonzero(~is_valid & in_fold)
    if len(fit_rows) > args.train_queries:
        fit_rows = np.sort(rng.choice(fit_rows, args.train_queries, replace=False))
    valid_rows = np.flatnonzero(is_valid & in_fold)
    if len(valid_rows) > args.valid_queries:
        valid_rows = np.sort(rng.choice(valid_rows, args.valid_queries, replace=False))
    cand = load_candidates(args.train, np.concatenate([fit_rows, valid_rows]))
    log.info(f"{args.train}: {len(queries):,} queries; fit {len(fit_rows):,} queries, valid {len(valid_rows):,} "
             f"queries; {len(cand):,} candidates read")

    # negatives: hardest by fused rank + random (chosen on 3 small columns, then rows selected once)
    is_fit = cand.query_row.isin(fit_rows).to_numpy()
    if args.all_negatives:
        fit_cand, valid_cand = cand[is_fit], cand[~is_fit]
        del cand
    else:
        fit_cand, valid_cand = _sample_negatives(cand, is_fit, args, rng)
    gc.collect()
    log.info(f"fit rows {len(fit_cand):,} ({fit_cand.label.mean():.2%} positive), valid rows {len(valid_cand):,}")
    _train_and_score(args, name, texts, s1_emb, queries, fit_cand, valid_cand, valid_rows)


def _sample_negatives(cand, is_fit, args, rng):
    key = cand.loc[is_fit, ["query_row", "fused_rank", "label"]]
    neg = key[key.label == 0]
    min_rank = neg.groupby("query_row").fused_rank.transform("min")
    hard_idx = neg.index[(neg.fused_rank <= min_rank + args.hard_neg - 1).to_numpy()]
    rest = neg.drop(hard_idx)
    order = np.lexsort((rng.random(len(rest)), rest.query_row.to_numpy()))
    rest = rest.iloc[order]
    rand_idx = rest.index[rest.groupby("query_row").cumcount().to_numpy() < args.rand_neg]
    pos_idx = key.index[key.label.to_numpy() == 1]
    return cand.loc[np.sort(np.concatenate([pos_idx, hard_idx, rand_idx]))], cand[~is_fit]


class F05Monitor(xgb.callback.TrainingCallback):
    """Logs the best validation macro F0.5 (over a coarse threshold grid) every `every` rounds."""

    def __init__(self, dvalid, vc, vq, singletons, every):
        self.dvalid, self.vc, self.vq, self.singletons, self.every = dvalid, vc, vq, singletons, every

    def after_iteration(self, model, epoch, evals_log):
        if self.every and epoch % self.every == 0:
            p = model.predict(self.dvalid, iteration_range=(0, epoch + 1))
            grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            best = max((evaluate(self.vc, self.vq, p, t, self.singletons)["f05"], t) for t in grid)
            log.info(f"  round {epoch}: valid macro F0.5 {best[0]:.4f} (threshold {best[1]:.3f})")
        return False


def _train_and_score(args, name, texts, s1_emb, queries, fit_cand, valid_cand, valid_rows):

    split = query_split(args.train)
    qtext = texts.queries(split, queries.file_row.to_numpy())
    q_emb = query_embeddings(split, queries)
    t0 = time.time()
    negatives = "all" if args.all_negatives else f"h{args.hard_neg}-r{args.rand_neg}"
    sample_key = (f"{args.train}-fit{args.train_queries}-valid{args.valid_queries}-{negatives}-s{args.seed}"
                  + (f"-fold{args.fold}" if args.fold else ""))
    X_fit = cached_features(f"{sample_key}-fit", fit_cand,
                            lambda: build_features(fit_cand, qtext, texts.s1, q_emb, s1_emb, texts.s1_row))
    log.info(f"  fit features done ({time.time() - t0:.0f}s)")
    X_valid = cached_features(f"{sample_key}-valid", valid_cand,
                              lambda: build_features(valid_cand, qtext, texts.s1, q_emb, s1_emb, texts.s1_row))
    if args.ce:
        X_fit = add_all_ce_features(X_fit, fit_cand, args.ce, args.train)
        X_valid = add_all_ce_features(X_valid, valid_cand, args.ce, args.train)
        for ce in args.ce:
            log.info(f"  cross-encoder features from {ce} ({ce_prefix(ce)}_*): "
                     f"{X_fit[ce_prefix(ce) + '_logit'].notna().mean():.1%} of fit candidates scored")
    log.info(f"features: {X_fit.shape[1]} columns, built in {time.time() - t0:.0f}s"
             f"{'' if q_emb is not None else ' (no embedding feature: s23-' + split + ' embeddings missing)'}")

    # ---- train
    feature_names = list(X_fit.columns)
    dfit = xgb.QuantileDMatrix(X_fit, label=fit_cand.label.to_numpy(), missing=np.nan)
    dvalid = xgb.QuantileDMatrix(X_valid, label=valid_cand.label.to_numpy(), missing=np.nan, ref=dfit)
    del X_fit, X_valid
    gc.collect()
    params = {"objective": "binary:logistic", "eval_metric": ["aucpr", "logloss"], "tree_method": "hist",
              "device": "cuda", "learning_rate": args.learning_rate, "max_depth": args.max_depth,
              "min_child_weight": args.min_child_weight, "subsample": args.subsample,
              "colsample_bytree": args.colsample, "seed": args.seed}
    vq = queries.iloc[valid_rows].reset_index(drop=True)
    remap = pd.Series(np.arange(len(valid_rows)), index=valid_rows)
    vc = valid_cand.assign(query_row=remap.loc[valid_cand.query_row.to_numpy()].to_numpy())
    train_singletons = singleton_info("train")
    t0 = time.time()
    model = xgb.train(params, dfit, num_boost_round=args.rounds, evals=[(dvalid, "valid")],
                      early_stopping_rounds=args.early_stop, verbose_eval=False,
                      callbacks=[xgb.callback.EvaluationMonitor(period=100, logger=log.info),
                                 F05Monitor(dvalid, vc, vq, train_singletons, args.f05_every)])
    log.info(f"trained in {time.time() - t0:.0f}s, best iteration {model.best_iteration}, "
             f"valid logloss {model.best_score:.5f} (early stopping on logloss: global AUC-PR across queries "
             f"is not what the per-query decision needs)")

    # ---- threshold on valid (README macro F0.5)
    p_valid = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
    sweep = []
    for t in np.round(np.r_[np.arange(0.05, 0.80, 0.05), np.arange(0.80, 0.995, 0.01)], 2):
        sweep.append({"threshold": float(t), **evaluate(vc, vq, p_valid, t, train_singletons)})
    best = max(sweep, key=lambda r: r["f05"])
    log.info("valid threshold sweep (macro F0.5 | pair precision | pair recall):")
    for r in sweep:
        log.info(f"  t={r['threshold']:.2f}  F0.5 {r['f05']:.4f}  P {r['pair_precision']:.4f}  R {r['pair_recall']:.4f}"
                 + ("   <- best" if r is best else ""))

    # ---- final score on the eval run
    equeries = load_queries(args.eval)
    ecand = load_candidates(args.eval)
    esplit = query_split(args.eval)
    etext = texts.queries(esplit, equeries.file_row.to_numpy())
    X_eval = cached_features(f"{args.eval}-all", ecand, lambda: build_features(
        ecand, etext, texts.s1, query_embeddings(esplit, equeries), s1_emb, texts.s1_row))
    if args.ce:
        X_eval = add_all_ce_features(X_eval, ecand, args.ce, args.eval)
    p_eval = model.predict(xgb.DMatrix(X_eval[feature_names], missing=np.nan),
                           iteration_range=(0, model.best_iteration + 1))
    eval_singletons = singleton_info(esplit)
    result = evaluate(ecand, equeries, p_eval, best["threshold"], eval_singletons)
    top1 = evaluate(ecand, equeries, -ecand.fused_rank.to_numpy().astype(float), -1.0, eval_singletons)
    log.info(f"EVAL {args.eval} at threshold {best['threshold']:.2f}: macro F0.5 {result['f05']:.4f} over "
             f"{result['entities']:,} S1 entities | pair precision {result['pair_precision']:.4f}, "
             f"recall {result['pair_recall']:.4f}, accepted {result['accepted']:,}/{result['queries']:,} queries")
    log.info(f"baseline (always take the top fused candidate): macro F0.5 {top1['f05']:.4f}, "
             f"precision {top1['pair_precision']:.4f}, recall {top1['pair_recall']:.4f}")

    gain = model.get_score(importance_type="gain")
    log.info("top features by gain: " + ", ".join(f"{k} {v:.1f}" for k, v in
                                                   sorted(gain.items(), key=lambda kv: -kv[1])[:20]))
    MODELS.mkdir(parents=True, exist_ok=True)
    model.save_model(MODELS / f"{name}.json")
    (MODELS / f"{name}.meta.json").write_text(json.dumps({
        "features": feature_names, "ce": args.ce, "threshold": best["threshold"], "best_iteration": model.best_iteration,
        "valid_sweep": sweep, "eval": result, "eval_baseline_top1": top1, "args": vars(args)}, indent=2))
    log.info(f"model -> {MODELS / name}.json (+ .meta.json)")


if __name__ == "__main__":
    main()
