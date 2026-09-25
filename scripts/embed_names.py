"""Embed standardized business names for cosine k-NN search (GPU).

Each target is embedded in file order and saved as
    artifacts/embeddings/<target>.npy        float16 (N, 384), L2-normalized
    artifacts/embeddings/<target>.ids.npy    entity_id per row

Targets:
    s1-train, s1-test   Source 1 records (indexed into er-train / er-test)
    s23-eval            dataset/splits/s23_eval.tsv (queries for evaluation)
    s23-train           dataset/splits/s23_train.tsv
    s23-test            test Source 2 + Source 3 (queries for the submission)

Run on a GPU, e.g. inside the OpenSearch job:
    srun --jobid=<id> --overlap --gres=gpu:2g.48gb:1 --cpus-per-task=12 \
        python scripts/embed_names.py s1-train s1-test s23-eval
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", f"/scratch/{os.environ.get('USER', 'ckarfa')}/hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
from name_standardization import standardize_names  # noqa: E402

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUT = ROOT / "artifacts" / "embeddings"
DATA = ROOT / "dataset"
TARGETS = {
    "s1-train": [DATA / "train" / "train_source1.tsv"],
    "s1-test": [DATA / "test" / "test_source1.tsv"],
    "s23-eval": [DATA / "splits" / "s23_eval.tsv"],
    "s23-train": [DATA / "splits" / "s23_train.tsv"],
    "s23-test": [DATA / "test" / "test_source2.tsv", DATA / "test" / "test_source3.tsv"],
}


def load_names(paths):
    frames = [pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id", "business_name"])
              for p in paths]
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", choices=list(TARGETS))
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--n-jobs", type=int, default=len(os.sched_getaffinity(0)))
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    if model.device.type == "cuda":
        model.half()
    print(f"{MODEL} on {model.device}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for target in args.targets:
        start = time.time()
        df = load_names(TARGETS[target])
        names = standardize_names(df.business_name, n_jobs=args.n_jobs)
        vectors = model.encode(names, batch_size=args.batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        np.save(OUT / f"{target}.npy", vectors)
        np.save(OUT / f"{target}.ids.npy", df.entity_id.to_numpy(dtype=str))
        print(f"{target}: {vectors.shape} in {time.time() - start:,.0f}s -> {OUT / target}.npy", flush=True)


if __name__ == "__main__":
    main()
