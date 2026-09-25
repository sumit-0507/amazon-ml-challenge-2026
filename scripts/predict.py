"""Predict matches for the test set and write the two submission files.

Input: test candidate runs from gpu_retrieve.py (one or more parts) and a model
trained by train_xgb.py. For every S2/S3 record: build the same features as in
training (in chunks of --chunk-queries records), score its candidates, and keep
the most probable one if p >= the model's threshold (a record matches at most one
S1 entity). Then write, in --out:

    matching_results.tsv   source1_entity_id, matched_entity_ids   (the leaderboard file)
    candidate_pairs.tsv    source1_entity_id, candidate_entity_ids (every candidate the model scored)

Every test S1 entity gets exactly one row in each (empty list when nothing matched /
no candidate). Per-record predictions (best S1, probability) are also saved in
artifacts/predictions/<name>/ for analysis.

    srun --jobid=<id> --overlap --gres=gpu:2g.48gb:1 --cpus-per-task=12 python scripts/predict.py \\
        --runs gpu-tfidf-test-part0of4 gpu-tfidf-test-part1of4 gpu-tfidf-test-part2of4 gpu-tfidf-test-part3of4 \\
        --model xgb-train30-full

Stages (to overlap scoring with retrieval of the next part):
    --stage score   score the given runs only (per-record results kept in the work folder)
    --stage write   write the two files from everything scored so far in the work folder
    --stage all     both (default)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "scripts"))
import retrieve_candidates as rc  # noqa: E402
import train_xgb as T  # noqa: E402

log = rc.log
PRED = ROOT / "artifacts" / "predictions"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=[], help="candidate runs (parts) to score")
    ap.add_argument("--model", required=True, help="model name in artifacts/models/")
    ap.add_argument("--split", default="test", choices=["test", "eval", "train"], help="query set of the runs")
    ap.add_argument("--threshold", type=float, help="override the model's threshold")
    ap.add_argument("--chunk-queries", type=int, default=250_000)
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--name", help="prediction name (default: <model>-<split>)")
    ap.add_argument("--stage", choices=["all", "score", "write"], default="all")
    args = ap.parse_args()
    name = args.name or f"{args.model}-{args.split}"
    rc.setup_logging(f"predict-{name}-{args.stage}")
    n_jobs = len(os.sched_getaffinity(0))
    s1_split = "test" if args.split == "test" else "train"

    meta = json.loads((T.MODELS / f"{args.model}.meta.json").read_text())
    model = xgb.Booster()
    model.load_model(T.MODELS / f"{args.model}.json")
    # XGB_DEVICE=cpu for GPUs the xgboost build does not support (e.g. P100 / SM 60)
    model.set_param({"device": os.environ.get("XGB_DEVICE", "cuda"), "nthread": n_jobs})
    features, n_trees = meta["features"], meta["best_iteration"] + 1
    threshold = args.threshold if args.threshold is not None else meta["threshold"]
    log.info(f"predict {name}: model {args.model} ({n_trees} trees, threshold {threshold}), runs {args.runs}")

    t0 = time.time()
    texts = T.Texts(n_jobs, s1_split=s1_split)
    s1_emb_path = T.EMB / f"s1-{s1_split}.npy"
    s1_emb = np.load(s1_emb_path, mmap_mode="r") if s1_emb_path.exists() else None
    log.info(f"S1 {s1_split}: {len(texts.s1):,} records standardized in {time.time() - t0:.0f}s")

    work = PRED / name
    (work / "candidate_pairs").mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "score"):
        score_runs(args, texts, s1_emb, model, features, n_trees, threshold, work)
    if args.stage in ("all", "write"):
        write_outputs(args, texts, threshold, work, n_jobs)
    log.info(f"done in {time.time() - t0:,.0f}s")


def score_runs(args, texts, s1_emb, model, features, n_trees, threshold, work):
    n_queries, n_pairs, n_matched, t1 = 0, 0, 0, time.time()
    for run in args.runs:
        for old in list(work.glob(f"best-{run}-*.parquet")) + list((work / "candidate_pairs").glob(f"{run}-*.parquet")):
            old.unlink()
        queries = T.load_queries(run)
        qtext = texts.queries(args.split, queries.file_row.to_numpy())
        q_emb = T.query_embeddings(args.split, queries)
        if "embedding_cosine" in features and (q_emb is None or s1_emb is None):
            sys.exit(f"model uses embedding_cosine but embeddings for {args.split} queries / S1 are missing")
        qids = queries.entity_id.to_numpy()
        for a in range(0, len(queries), args.chunk_queries):
            b = min(a + args.chunk_queries, len(queries))
            cand = pd.read_parquet(T.CAND / f"{run}.parquet", columns=T.CANDIDATE_COLUMNS[:4] + T.CANDIDATE_COLUMNS[5:],
                                   filters=[("query_row", ">=", a), ("query_row", "<", b)]).reset_index(drop=True)
            X = T.build_features(cand, qtext, texts.s1, q_emb, s1_emb, texts.s1_row)[features]
            p = model.predict(xgb.DMatrix(X, missing=np.nan), iteration_range=(0, n_trees))
            rows, s1, bp = T.best_candidates(cand, p)
            chunk_pred = pd.DataFrame({"query_entity_id": qids[rows], "best_s1": s1, "p": bp})
            chunk_pred.to_parquet(work / f"best-{run}-{a}.parquet", index=False)
            n_matched += int((chunk_pred.p >= threshold).sum())
            pd.DataFrame({"s1": cand.s1_id.to_numpy(), "q": qids[cand.query_row.to_numpy()]}).to_parquet(
                work / "candidate_pairs" / f"{run}-{a}.parquet", index=False)
            n_queries += b - a
            n_pairs += len(cand)
            rate = n_queries / (time.time() - t1)
            log.info(f"  {run} queries {b:,}/{len(queries):,}: {len(cand):,} pairs scored, "
                     f"{(chunk_pred.p >= threshold).sum():,} matches  ({rate:,.0f} q/s overall)")
    log.info(f"scored {n_pairs:,} candidate pairs for {n_queries:,} records in {time.time() - t1:,.0f}s; "
             f"{n_matched:,} records matched ({n_matched / max(n_queries, 1):.1%})")


def write_outputs(args, texts, threshold, work, n_jobs):
    """Both submission files from all scored runs in the work folder; one row per S1 entity, in Source 1 order."""
    best = pd.concat([pd.read_parquet(f) for f in sorted(work.glob("best-*.parquet"))], ignore_index=True)
    if best.query_entity_id.duplicated().any():
        sys.exit("a record was scored twice in the work folder; re-run --stage score for a clean set of runs")
    matches = best[best.p >= threshold]
    log.info(f"{len(best):,} scored records in {work}; {len(matches):,} matched at threshold {threshold}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s1_ids = pd.Series(texts.s1.index, name="source1_entity_id")
    grouped = matches.groupby("best_s1").query_entity_id.agg(lambda ids: ",".join(sorted(set(ids))))
    result = s1_ids.to_frame().assign(matched_entity_ids=s1_ids.map(grouped).fillna("").to_numpy())
    result.to_csv(out / "matching_results.tsv", sep="\t", index=False)
    log.info(f"matching_results.tsv: {len(result):,} S1 rows, {(result.matched_entity_ids != '').sum():,} with matches "
             f"-> {out / 'matching_results.tsv'}")

    import duckdb
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{work / 'duckdb_tmp'}'; SET memory_limit='12GB'; SET threads={n_jobs}")
    con.register("s1_ids", s1_ids.to_frame())
    con.execute(f"""
        COPY (
            SELECT s.source1_entity_id, c.ids AS candidate_entity_ids  -- NULL is written as an empty field
            FROM s1_ids s
            LEFT JOIN (SELECT s1, string_agg(DISTINCT q, ',' ORDER BY q) AS ids
                       FROM read_parquet('{work / 'candidate_pairs' / '*.parquet'}') GROUP BY s1) c
              ON c.s1 = s.source1_entity_id
        ) TO '{out / 'candidate_pairs.tsv'}' (DELIMITER '\t', HEADER)
    """)
    log.info(f"candidate_pairs.tsv: candidate pairs grouped by S1 -> {out / 'candidate_pairs.tsv'}")


if __name__ == "__main__":
    main()
