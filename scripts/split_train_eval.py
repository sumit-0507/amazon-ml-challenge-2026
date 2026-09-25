"""Split the training Source 2/3 records into train (95%) and eval (5%).

The split is made at the Source 1 entity level: an S1 entity and every S2/S3
record matched to it land on the same side, so eval can be scored per S1 entity
exactly like the leaderboard (F0.5 over matched_entity_ids). S2/S3 records that
match no S1 entity are split randomly with the same fraction, so eval keeps the
same share of true non-matches. The S1 index (er-train) is not affected: eval
queries still search all training S1 entities.

Writes to dataset/splits/:
    s23_train.tsv, s23_eval.tsv    S2/S3 records + their source1_entity_id ("" if unmatched)
    ground_truth_train.tsv, ground_truth_eval.tsv    ground-truth rows of each side's S1 entities

    python scripts/split_train_eval.py [--eval-frac 0.05] [--seed 42]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "dataset" / "train"
OUT = ROOT / "dataset" / "splits"


def read(name):
    return pd.read_csv(TRAIN / name, sep="\t", dtype=str, keep_default_na=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    gt = read("train_ground_truth.tsv")
    s1_eval = set(gt.source1_entity_id[rng.random(len(gt)) < args.eval_frac])

    links = gt[gt.matched_entity_ids != ""].assign(entity_id=gt.matched_entity_ids.str.split(","))
    links = links.explode("entity_id")[["entity_id", "source1_entity_id"]]

    s23 = pd.concat([read("train_source2.tsv"), read("train_source3.tsv")], ignore_index=True)
    s23 = s23.merge(links, on="entity_id", how="left").fillna({"source1_entity_id": ""})
    matched = s23.source1_entity_id != ""
    is_eval = np.where(matched, s23.source1_entity_id.isin(s1_eval), rng.random(len(s23)) < args.eval_frac)

    OUT.mkdir(parents=True, exist_ok=True)
    gt_eval = gt.source1_entity_id.isin(s1_eval)
    outputs = {
        "s23_train.tsv": s23[~is_eval], "s23_eval.tsv": s23[is_eval],
        "ground_truth_train.tsv": gt[~gt_eval], "ground_truth_eval.tsv": gt[gt_eval],
    }
    for name, df in outputs.items():
        df.to_csv(OUT / name, sep="\t", index=False)

    print(f"seed {args.seed}, eval fraction {args.eval_frac}")
    print(f"{'':6} {'S1 entities':>12} {'S2/S3 records':>14} {'matched':>10} {'unmatched':>10}")
    for side, mask, gmask in (("train", ~is_eval, ~gt_eval), ("eval", is_eval, gt_eval)):
        d = s23[mask]
        print(f"{side:6} {gmask.sum():>12,} {len(d):>14,} {(d.source1_entity_id != '').sum():>10,} "
              f"{(d.source1_entity_id == '').sum():>10,}")
    for side, mask in (("train", ~is_eval), ("eval", is_eval)):
        d = s23[mask]
        mix = (d.country + "/" + d.entity_id.str[:2]).value_counts(normalize=True).sort_index()
        print(f"{side:6} country/source mix: " + ", ".join(f"{k} {v:.1%}" for k, v in mix.items()))
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
