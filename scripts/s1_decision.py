"""Per-S1-entity match decisions that maximize expected macro F0.5.

The leaderboard averages F0.5 over Source 1 entities, singletons included. A single
global threshold on each record's best candidate ignores that: an extra correct
record is worth more for an entity with one true match than for one with five, and
one false merge onto a singleton costs its whole 1.0. Here every record still points
to its most probable S1 candidate; then, per S1 entity, the records pointing to it
are sorted by probability and the number k to accept (0..n) is the one with the
highest expected F0.5 under independent probabilities q_i = p_i ** gamma:

    E[F | top k] ~= 1.25 * sum_{i<=k} q_i / (0.25 * (sum_i q_i + m) + k)      k >= 1
    E[F | none]  ~= prod_i (1 - q_i)   (probability the entity really has no match)

m = expected true matches that point elsewhere (misses), gamma calibrates the
probabilities. Both are tuned on half of the eval S1 entities; the other half
reports the score, next to the global-threshold baseline on the same half.

    python scripts/s1_decision.py --run gpu-tfidf-eval-30 --model xgb-train30-full
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


def best_per_query(cand, prob, n_queries):
    rows, s1, p = T.best_candidates(cand, prob)
    best_s1 = np.full(n_queries, "", dtype=object)
    best_p = np.zeros(n_queries)
    best_s1[rows], best_p[rows] = s1, p
    return best_s1, best_p


def decide(best_s1, best_p, gamma, m, floor=0.02):
    """Accept mask per query from per-S1 expected-F0.5 decisions."""
    df = pd.DataFrame({"s1": best_s1, "q": np.clip(best_p, 0, 1) ** gamma, "row": np.arange(len(best_s1))})
    df = df[(df.s1 != "") & (df.q >= floor)].sort_values(["s1", "q"], ascending=[True, False])
    accept = np.zeros(len(best_s1), dtype=bool)
    for _, g in df.groupby("s1", sort=False):
        q = g.q.to_numpy()
        cum = np.cumsum(q)
        k = np.arange(1, len(q) + 1)
        e_some = 1.25 * cum / (0.25 * (cum[-1] + m) + k)
        e_none = np.prod(1 - q)
        best_k = int(np.argmax(e_some)) + 1
        if e_some[best_k - 1] > e_none:
            accept[g.row.to_numpy()[:best_k]] = True
    return accept


def score(queries, best_s1, accept, singletons, seed=0):
    f05, n = T.macro_f05(queries, best_s1, accept, singletons, np.random.default_rng(seed))
    truth = queries.source1_entity_id.to_numpy()
    correct = accept & (best_s1 == truth) & (truth != "")
    return {"f05": f05, "entities": n, "precision": float(correct.sum() / max(accept.sum(), 1)),
            "recall": float(correct.sum() / max((truth != "").sum(), 1)), "accepted": int(accept.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="gpu-tfidf-eval-30")
    ap.add_argument("--model", default="xgb-train30-full")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rc.setup_logging(f"s1-decision-{args.model}-{args.run}")

    meta = json.loads((T.MODELS / f"{args.model}.meta.json").read_text())
    model = xgb.Booster()
    model.load_model(T.MODELS / f"{args.model}.json")
    queries = T.load_queries(args.run)
    cand = T.load_candidates(args.run)
    X = pd.read_parquet(T.FEATURES / f"{args.run}-all.parquet")[meta["features"]]
    prob = model.predict(xgb.DMatrix(X, missing=np.nan), iteration_range=(0, meta["best_iteration"] + 1))
    del X
    best_s1, best_p = best_per_query(cand, prob, len(queries))
    log.info(f"{args.run}: {len(queries):,} queries scored with {args.model}")

    # split the S1 entities (and unmatched queries) into a tuning half and a report half
    rng = np.random.default_rng(args.seed)
    truth = queries.source1_entity_id.to_numpy()
    ents = pd.unique(np.concatenate([truth[truth != ""], best_s1[best_s1 != ""]]))
    tune_ents = set(ents[rng.random(len(ents)) < 0.5])
    key = np.where(truth != "", truth, best_s1)
    in_tune = np.array([k in tune_ents if k else r < 0.5 for k, r in zip(key, rng.random(len(key)))])
    singles = T.singleton_info("eval" if "eval" in args.run else "train")
    halves = {}
    for name, mask in (("tune", in_tune), ("report", ~in_tune)):
        sid = rng.permutation(singles["ids"])
        part = {"ids": sid[: len(sid) // 2] if name == "tune" else sid[len(sid) // 2:], "ratio": singles["ratio"]}
        halves[name] = (queries[mask].reset_index(drop=True), best_s1[mask], best_p[mask], part)

    q_t, s_t, p_t, sg_t = halves["tune"]
    base_t = max((score(q_t, s_t, p_t >= t, sg_t)["f05"], t) for t in np.arange(0.5, 0.96, 0.01))
    log.info(f"tune half: best global threshold {base_t[1]:.2f} -> F0.5 {base_t[0]:.4f}")
    grid = []
    for gamma in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        for m in (0.0, 0.1, 0.25, 0.5, 1.0):
            r = score(q_t, s_t, decide(s_t, p_t, gamma, m), sg_t)
            grid.append((r["f05"], gamma, m))
    grid.sort(reverse=True)
    log.info("tune half, per-S1 decision (top 5): " + ", ".join(f"gamma={g} m={m}: {f:.4f}" for f, g, m in grid[:5]))
    _, gamma, m = grid[0]

    q_r, s_r, p_r, sg_r = halves["report"]
    base = score(q_r, s_r, p_r >= base_t[1], sg_r)
    per_s1 = score(q_r, s_r, decide(s_r, p_r, gamma, m), sg_r)
    log.info(f"REPORT half, global threshold {base_t[1]:.2f}: F0.5 {base['f05']:.4f}  P {base['precision']:.4f}  "
             f"R {base['recall']:.4f}  accepted {base['accepted']:,}")
    log.info(f"REPORT half, per-S1 decision (gamma={gamma}, m={m}): F0.5 {per_s1['f05']:.4f}  "
             f"P {per_s1['precision']:.4f}  R {per_s1['recall']:.4f}  accepted {per_s1['accepted']:,}")
    log.info(f"gain: {per_s1['f05'] - base['f05']:+.4f}")
    (T.MODELS / f"{args.model}.s1_decision.json").write_text(json.dumps(
        {"gamma": gamma, "m": m, "tune_grid": grid, "report_global": base, "report_per_s1": per_s1,
         "global_threshold": base_t[1]}, indent=2))


if __name__ == "__main__":
    main()
