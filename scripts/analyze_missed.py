"""Where do the missed matches sit? Compare, on a labelled candidate run, the records whose true
S1 match is the model's top pick but falls below the threshold ("right but rejected") with the
ones that are accepted, by how many searches found the true candidate, its RRF / fused rank /
per-search ranks, and the model probability.

    python scripts/analyze_missed.py --run gpu-tfidf-eval-30 --model xgb-train30-full
"""

import argparse
import json
import sys
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="gpu-tfidf-eval-30")
    ap.add_argument("--model", default="xgb-train30-full")
    args = ap.parse_args()
    rc.setup_logging(f"analyze-missed-{args.run}-{args.model}")

    meta = json.loads((T.MODELS / f"{args.model}.meta.json").read_text())
    model = xgb.Booster()
    model.load_model(T.MODELS / f"{args.model}.json")
    model.set_param({"device": "cpu"})
    threshold = meta["threshold"]
    queries = T.load_queries(args.run)
    cand = T.load_candidates(args.run)
    X = pd.read_parquet(T.FEATURES / f"{args.run}-all.parquet")
    if meta.get("ce"):
        X = T.add_all_ce_features(X, cand, meta["ce"], args.run)
    X = X[meta["features"]]
    cand["p"] = model.predict(xgb.DMatrix(X, missing=np.nan), iteration_range=(0, meta["best_iteration"] + 1))
    cand["n_searches"] = X["n_searches"].to_numpy()
    del X

    top = cand.sort_values(["query_row", "p"], ascending=[True, False]).groupby("query_row").head(1)
    true = cand[cand.label == 1].set_index("query_row")
    matched = queries.index[queries.source1_entity_id != ""]
    top_true = top[top.label == 1].set_index("query_row")
    groups = {
        "accepted (right, p >= thr)": top_true[top_true.p >= threshold].index,
        "right but rejected (p < thr)": top_true[top_true.p < threshold].index,
        "wrong top pick": pd.Index(matched).difference(top_true.index).intersection(true.index),
        "not retrieved": pd.Index(matched).difference(true.index),
    }
    log.info(f"{args.run}: {len(matched):,} matched queries, threshold {threshold}")
    for name, idx in groups.items():
        log.info(f"  {name:32} {len(idx):>8,} ({len(idx) / len(matched):.2%})")

    log.info("true candidate, by group (rows: group; values: share or median):")
    searches = [m for m in T.SEARCHES]
    header = f"{'':32}{'n=4':>7}{'n=3':>7}{'n=2':>7}{'n=1':>7}{'rrf':>9}{'fused':>7}" + \
             "".join(f"{m + '_rk':>10}" for m in searches) + f"{'p med':>8}"
    log.info(header)
    for name, idx in list(groups.items())[:3]:
        t = true.loc[true.index.intersection(idx)]
        n = t.n_searches.value_counts(normalize=True)
        ranks = "".join(f"{t[f'{m}_rank'].median():>10.0f}" if t[f'{m}_rank'].notna().any() else f"{'-':>10}"
                        for m in searches)
        log.info(f"{name:32}" + "".join(f"{n.get(k, 0):>7.1%}" for k in (4, 3, 2, 1))
                 + f"{t.rrf.median():>9.4f}{t.fused_rank.median():>7.0f}{ranks}{t.p.median():>8.3f}")
    rej = true.loc[true.index.intersection(groups["right but rejected (p < thr)"])]
    missing = {m: rej[f"{m}_rank"].isna().mean() for m in searches}
    log.info("right but rejected: share where each search did NOT find the true candidate: "
             + ", ".join(f"{m} {v:.1%}" for m, v in missing.items()))
    q = rej.p.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    log.info("right but rejected: p quantiles 10/25/50/75/90%: " + " ".join(f"{v:.3f}" for v in q))


if __name__ == "__main__":
    main()
