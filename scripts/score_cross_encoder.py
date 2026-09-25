"""Score the candidate pairs of a retrieval run with cross-fitted cross-encoders.

Models come from train_cross_encoder.py --fold A / --fold B. A record of the training
split is scored only by the model(s) not trained on its fold (out-of-fold), so XGBoost
can be trained on those scores; eval and test records get the mean logit of all
models. Every candidate with fused rank <= the models' top is scored; per chunk of
--chunk-queries queries this writes
    artifacts/ce_scores/<name>/<run>/part-<first query row>.parquet
with columns query_row, s1_id, ce_logit. Finished chunks are skipped, so an
interrupted run resumes; a file _done marks a complete run. Candidates below the
cut get no score (NaN for XGBoost).

    python scripts/score_cross_encoder.py --models ce-minilm-A ce-minilm-B --run gpu-tfidf-eval-30
    python scripts/score_cross_encoder.py --models ce-minilm-A ce-minilm-B --run gpu-tfidf-train-30
    python scripts/score_cross_encoder.py --models ce-minilm-A ce-minilm-B --run gpu-tfidf-test-part0of4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # tokenization runs in this process only

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import train_cross_encoder as C  # noqa: E402

T, log = C.T, C.log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="cross-encoder names in artifacts/cross_encoders/")
    ap.add_argument("--run", required=True, help="candidate run in artifacts/candidates/")
    ap.add_argument("--name", help="score set name (default: the models' common prefix)")
    ap.add_argument("--top", type=int, help="fused rank cut (default: the models')")
    ap.add_argument("--chunk-queries", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()
    name = args.name or os.path.commonprefix(args.models).rstrip("-_") or "+".join(args.models)
    C.rc.setup_logging(f"ce-score-{name}-{args.run}")
    metas = [json.loads((C.CE / m / "meta.json").read_text()) for m in args.models]
    top = args.top or max(m["top"] for m in metas)
    max_len = max(m["max_len"] for m in metas)
    out = C.SCORES / name / args.run
    if (out / "_done").exists():
        log.info(f"{out} already complete")
        return
    out.mkdir(parents=True, exist_ok=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [C.load_model(C.CE / m / "model", device) for m in args.models]
    t0 = time.time()
    split = C.run_split(args.run)
    texts = T.Texts(len(os.sched_getaffinity(0)), s1_split="test" if split == "test" else "train")
    s1text = C.s1_texts(texts)
    queries = T.load_queries(args.run)
    qfold = C.folds(queries)
    out_of_fold = split == "train"
    rows = np.arange(len(queries))
    if out_of_fold:  # training records: only those some model did not train on
        rows = rows[np.isin(qfold, [f for f in "AB" if any(m["fold"] != f for m in metas)])]
    qtext = C.query_texts(texts, args.run, queries, rows)
    log.info(f"score {args.run} with {', '.join(args.models)} (top {top}, "
             f"{'out-of-fold' if out_of_fold else 'mean of the models'}) on {device}: {len(rows):,} queries, "
             f"texts ready in {time.time() - t0:.0f}s")

    t1, n_done, n_pairs = time.time(), 0, 0
    for i in range(0, len(rows), args.chunk_queries):
        chunk = rows[i:i + args.chunk_queries]
        path = out / f"part-{chunk[0]:09d}.parquet"
        if path.exists():
            continue
        cand = pd.read_parquet(T.CAND / f"{args.run}.parquet", columns=["query_row", "s1_id"],
                               filters=[("query_row", ">=", int(chunk[0])), ("query_row", "<=", int(chunk[-1])),
                                        ("fused_rank", "<=", top)])
        cand = cand[np.isin(cand.query_row.to_numpy(), chunk)].reset_index(drop=True)
        a, b = C.pair_texts(cand, qtext, s1text)
        total, count = np.zeros(len(cand)), np.zeros(len(cand))
        for (tok, model), meta in zip(models, metas):
            use = (qfold[cand.query_row.to_numpy()] != meta["fold"]) if out_of_fold else np.ones(len(cand), bool)
            idx = np.flatnonzero(use)
            total[idx] += C.predict_logits(model, tok, a[idx], b[idx], args.batch_size, max_len, device)
            count[idx] += 1
        scored = count > 0
        tmp = path.with_suffix(".tmp")
        cand[scored].assign(ce_logit=(total[scored] / count[scored]).astype(np.float32)).to_parquet(tmp, index=False)
        tmp.replace(path)
        n_done, n_pairs = n_done + len(chunk), n_pairs + int(scored.sum())
        log.info(f"  queries {i + len(chunk):,}/{len(rows):,}: {int(scored.sum()):,} pairs  "
                 f"({n_pairs / (time.time() - t1):,.0f} pairs/s, "
                 f"eta {(len(rows) - i - len(chunk)) / n_done * (time.time() - t1) / 60:,.1f} min)")
    (out / "_done").write_text(json.dumps({"models": args.models, "run": args.run, "top": top}))
    log.info(f"done: {out}")


if __name__ == "__main__":
    main()
