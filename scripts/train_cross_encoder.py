"""Fine-tune a cross-encoder that scores (S2/S3 record, S1 candidate) pairs.

The XGBoost matcher compares the two records through string similarities (rapidfuzz)
and a bi-encoder cosine. A cross-encoder reads both records together, one transformer
pass over "query [SEP] candidate", fine-tuned on our labelled candidate lists; its
score becomes an extra XGBoost feature (score_cross_encoder.py writes it per run).

Text of a record: "<standardized name> | <standardized address>".

Cross-fitting: training queries are split by S1 entity (records without a match by
their own id) into folds A and B, stable across runs (crc32). One model is trained per
fold (--fold), so the two together use the whole training split; the eval split is
never used for training. A training record is later scored by the model of the other
fold only (out-of-fold), so XGBoost never learns from over-confident in-sample scores;
eval and test records get the mean of both models.

Training pairs (the fold's queries of --train-runs): every candidate with fused rank
<= --top (the pairs scored at inference), the true match wherever it is ranked, and
--rand-neg random lower-ranked negatives per query. Validation: --valid-queries
records of the eval run with their candidates of fused rank <= --top (exactly the
inference distribution): log-loss, AUC-PR, top-1 accuracy against the fused ranking.

Loss: binary cross-entropy per pair (label 1 = same business). Pairwise, not listwise:
a quarter of the records have no true match, and XGBoost needs a per-pair probability.
AdamW, linear warmup/decay, bf16 autocast. Checkpoints every --ckpt-every steps and
on SIGTERM (Slurm time limit); a re-run resumes from the last one (pairs are cached
next to it as indices into text tables). Output in artifacts/cross_encoders/<name>/:
model/ (Hugging Face format), meta.json.

    python scripts/train_cross_encoder.py --name ce-minilm-A --fold A     # full train split, fold A
    python scripts/train_cross_encoder.py --name ce-minilm-B --fold B
    ... --base xlm-roberta-base --batch-size 64                           # a larger multilingual base
"""

import argparse
import json
import os
import signal
import sys
import time
import zlib
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # tokenization runs in DataLoader workers

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "scripts"))
import retrieve_candidates as rc  # noqa: E402  (logging)
import train_xgb as T  # noqa: E402  (candidate runs, standardized texts)

CE = ROOT / "artifacts" / "cross_encoders"
SCORES = ROOT / "artifacts" / "ce_scores"
BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TRAIN_RUNS = [f"gpu-tfidf-train-part{k}of4" for k in range(4)]  # the whole training split
log = rc.log

# --------------------------------------------------------------------------
# Shared with score_cross_encoder.py
# --------------------------------------------------------------------------


def run_split(run):
    """Query set of a candidate run."""
    return "test" if "test" in run else T.query_split(run)


def folds(queries):
    """'A' or 'B' per query: by the true S1 entity, else by the record's own id (stable crc32)."""
    ids = queries.entity_id.to_numpy(dtype=str)
    if "source1_entity_id" in queries:
        s1 = queries.source1_entity_id.to_numpy(dtype=str)
        ids = np.where(s1 != "", s1, ids)
    return np.array(["A" if zlib.crc32(i.encode()) % 2 == 0 else "B" for i in ids])


TEXT_MODES = ("both", "name", "address")


def text_mode(meta):
    """Text a trained model reads (models from before --text read 'both')."""
    return meta.get("text_mode", "both")


def record_text(std, mode="both"):
    """Text per record of a standardized Texts frame: 'name | address', the name or the address."""
    if mode == "name":
        return std["name"].to_numpy(dtype=object)
    if mode == "address":
        return std["address"].to_numpy(dtype=object)
    return (std["name"] + " | " + std["address"]).to_numpy(dtype=object)


def s1_texts(texts, mode="both"):
    return pd.Series(record_text(texts.s1, mode), index=texts.s1.index)


def query_texts(texts, run, queries, rows, mode="both"):
    """Texts of the given query rows of a run, indexed by query_row."""
    q = queries.iloc[rows]
    return pd.Series(record_text(texts.queries(run_split(run), q.file_row.to_numpy()), mode),
                     index=q.query_row.to_numpy())


def pair_texts(cand, qtext, s1text):
    return qtext.loc[cand.query_row.to_numpy()].to_numpy(), s1text.loc[cand.s1_id.to_numpy()].to_numpy()


def restore_rope_buffers(model, path):
    """Models with their own code (gte-multilingual-reranker-base) keep position ids and rotary
    tables in non-persistent buffers, which transformers 5 leaves uninitialized on load: rebuild
    them from the checkpoint's config.json."""
    import torch
    from huggingface_hub import hf_hub_download
    mods = [m for m in model.modules() if hasattr(m, "_init_rope") and hasattr(m, "position_ids")]
    if not mods:
        return
    cfg_file = Path(path) / "config.json" if Path(path).is_dir() else hf_hub_download(str(path), "config.json")
    raw = json.loads(Path(cfg_file).read_text())
    cfg = model.config
    cfg.rope_scaling, cfg.rope_theta = raw.get("rope_scaling"), raw.get("rope_theta", 10000.0)
    for m in mods:
        with torch.device("cpu"):
            m._init_rope(cfg)
        m.position_ids = torch.arange(cfg.max_position_embeddings)


def load_model(path, device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    # fp32 weights (an fp16 checkpoint such as gte's would otherwise train in fp16 and overflow);
    # autocast runs the forward pass in bf16
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, trust_remote_code=True,
                                                               dtype=torch.float32)
    restore_rope_buffers(model, path)
    return tok, model.to(device)


def autocast(torch, device):
    return torch.autocast("cuda", dtype=torch.bfloat16,
                          enabled=device.type == "cuda" and torch.cuda.is_bf16_supported())


def predict_logits(model, tok, a, b, batch_size, max_len, device):
    """Logit per pair (a[i], b[i]); batches of similar length to save padding."""
    import torch
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)], kind="stable")
    out = np.empty(len(a), dtype=np.float32)
    model.eval()
    with torch.inference_mode(), autocast(torch, device):
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            enc = tok([a[j] for j in idx], [b[j] for j in idx], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            out[idx] = model(**enc).logits.float().squeeze(-1).cpu().numpy()
    return out

# --------------------------------------------------------------------------
# Pairs: (q, s, label, fused_rank) with q / s indices into query / S1 text tables
# --------------------------------------------------------------------------


def make_pairs(run, rows, queries, texts, top, rand_neg, rng, mode="both"):
    """Candidates of fused rank <= top for the (sorted) query rows; with rand_neg, also every true match
    and rand_neg random lower-ranked negatives per query (training). Returns (pairs, query texts)."""
    filters = [("query_row", "in", rows.astype(np.int32))]
    if not rand_neg:
        filters.append(("fused_rank", "<=", top))
    cand = pd.read_parquet(T.CAND / f"{run}.parquet", columns=["query_row", "s1_id", "fused_rank", "label"],
                           filters=filters).reset_index(drop=True)
    if rand_neg:
        kept = (cand.fused_rank <= top) | (cand.label == 1)
        rest = cand[~kept]
        rest = rest.iloc[np.lexsort((rng.random(len(rest)), rest.query_row.to_numpy()))]
        extra = rest[rest.groupby("query_row").cumcount().to_numpy() < rand_neg]
        cand = pd.concat([cand[kept], extra]).sort_index().reset_index(drop=True)
    pairs = pd.DataFrame({"q": np.searchsorted(rows, cand.query_row.to_numpy()).astype(np.int32),
                          "s": texts.s1_row.index.get_indexer(cand.s1_id.to_numpy()).astype(np.int32),
                          "label": cand.label.to_numpy(np.int8), "fused_rank": cand.fused_rank.to_numpy(np.int16)})
    return pairs, record_text(texts.queries(run_split(run), queries.file_row.to_numpy()[rows]), mode)


def pair_set(runs, fold, n_queries, texts, top, rand_neg, rng, mode="both", s1text=None):
    """Pairs of several runs; query indices offset into one concatenated text table. Pairs with an
    empty text on either side are dropped (address mode: nothing to compare, never scored)."""
    frames, qtexts, offset = [], [], 0
    for run in runs:
        queries = T.load_queries(run)
        rows = np.arange(len(queries)) if fold is None else np.flatnonzero(folds(queries) == fold)
        if n_queries and len(rows) > n_queries:
            rows = np.sort(rng.choice(rows, n_queries, replace=False))
        pairs, qtext = make_pairs(run, rows, queries, texts, top, rand_neg, rng, mode)
        if s1text is not None:
            pairs = pairs[(qtext[pairs.q.to_numpy()] != "") & (s1text[pairs.s.to_numpy()] != "")].reset_index(drop=True)
        frames.append(pairs.assign(q=pairs.q + offset))
        qtexts.append(qtext)
        offset += len(rows)
        log.info(f"  {run}: {len(rows):,} queries, {len(pairs):,} pairs ({pairs.label.mean():.2%} positive)")
    return pd.concat(frames, ignore_index=True), np.concatenate(qtexts)


def build_pairs(args, out):
    """{'train'|'valid': (pairs, query texts)}, S1 texts; cached in out/."""
    files = ["pairs-train", "pairs-valid", "texts-train", "texts-valid", "texts-s1"]
    if all((out / f"{f}.parquet").exists() for f in files):
        log.info(f"pair tables reused from {out}")
        t = {f: pd.read_parquet(out / f"{f}.parquet") for f in files}
        return ({k: (t[f"pairs-{k}"], t[f"texts-{k}"].text.to_numpy(dtype=object)) for k in ("train", "valid")},
                t["texts-s1"].text.to_numpy(dtype=object))
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    texts = T.Texts(len(os.sched_getaffinity(0)))
    s1text = record_text(texts.s1, args.text)
    log.info(f"S1 texts standardized in {time.time() - t0:.0f}s")
    log.info(f"train pairs (fold {args.fold}, text: {args.text}):")
    s1_filter = s1text if args.text == "address" else None
    train = pair_set(args.train_runs, args.fold, args.train_queries, texts, args.top, args.rand_neg, rng,
                     args.text, s1_filter)
    log.info(f"valid pairs:")
    valid = pair_set([args.valid_run], None, args.valid_queries, texts, args.top, 0, rng, args.text, s1_filter)
    log.info(f"{len(train[0]):,} train pairs ({train[0].label.mean():.2%} positive), {len(valid[0]):,} valid pairs")
    for k, (pairs, qtext) in (("train", train), ("valid", valid)):
        pairs.to_parquet(out / f"pairs-{k}.parquet", index=False)
        pd.DataFrame({"text": qtext}).to_parquet(out / f"texts-{k}.parquet", index=False)
    pd.DataFrame({"text": s1text}).to_parquet(out / "texts-s1.parquet", index=False)
    return {"train": train, "valid": valid}, s1text

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


class PairDataset:
    def __init__(self, qtext, s1text, pairs):
        self.qtext, self.s1text = qtext, s1text
        self.q, self.s, self.y = pairs.q.to_numpy(), pairs.s.to_numpy(), pairs.label.to_numpy(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.qtext[self.q[i]], self.s1text[self.s[i]], self.y[i]


class Collate:
    def __init__(self, tok, max_len):
        self.tok, self.max_len = tok, max_len

    def __call__(self, items):
        import torch
        a, b, y = zip(*items)
        enc = self.tok(list(a), list(b), truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        return enc, torch.tensor(y, dtype=torch.float32)


def ignore_sigterm(_worker_id):
    """DataLoader workers leave SIGTERM to the main process, which checkpoints before exiting."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


class Batches:
    """Index batches from global step `start` on; each epoch is a fresh seeded permutation."""

    def __init__(self, n, batch_size, total, start, seed):
        self.n, self.bs, self.total, self.start, self.seed = n, batch_size, total, start, seed

    def __iter__(self):
        per_epoch = self.n // self.bs
        for step in range(self.start, self.total):
            epoch, k = divmod(step, per_epoch)
            if step == self.start or k == 0:
                order = np.random.default_rng(self.seed + epoch).permutation(self.n)
            yield order[k * self.bs:(k + 1) * self.bs].tolist()


def evaluate(model, tok, valid, s1text, args, device):
    from sklearn.metrics import average_precision_score
    pairs, qtext = valid
    logits = predict_logits(model, tok, qtext[pairs.q.to_numpy()], s1text[pairs.s.to_numpy()],
                            args.eval_batch_size, args.max_len, device)
    y = pairs.label.to_numpy()
    p = np.clip(1 / (1 + np.exp(-logits.astype(np.float64))), 1e-7, 1 - 1e-7)
    df = pd.DataFrame({"q": pairs.q.to_numpy(), "y": y, "s": logits, "r": pairs.fused_rank.to_numpy()})
    d = df[df.groupby("q").y.transform("max") == 1]
    return {"logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "aucpr": float(average_precision_score(y, logits)),
            "top1_ce": float(d.loc[d.groupby("q").s.idxmax()].y.mean()),
            "top1_fused": float(d.loc[d.groupby("q").r.idxmin()].y.mean()),
            "queries_with_match": int(d.q.nunique())}


def train(args, data, s1text, out):
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_model(args.base, device)
    pairs, qtext = data["train"]
    per_epoch = len(pairs) // args.batch_size
    total = per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total), total)
    ckpt = out / "checkpoint.pt"
    step = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step = state["step"]
        log.info(f"resumed from {ckpt} at step {step:,}")
    log.info(f"base {args.base} on {device}"
             + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
             + f": {len(pairs):,} pairs, {total:,} steps of {args.batch_size} ({args.epochs} epoch(s))")

    def save_checkpoint():
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step}, tmp)
        tmp.replace(ckpt)
        log.info(f"  checkpoint at step {step:,}")

    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    loader = DataLoader(PairDataset(qtext, s1text, pairs),
                        batch_sampler=Batches(len(pairs), args.batch_size, total, step, args.seed),
                        collate_fn=Collate(tok, args.max_len), num_workers=args.workers, pin_memory=True,
                        worker_init_fn=ignore_sigterm)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    metrics = None
    model.train()
    t0, t_log, loss_sum, n_log = time.time(), time.time(), 0.0, 0
    for enc, y in loader:
        enc, y = enc.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(torch, device):
            logits = model(**enc).logits.squeeze(-1)
        loss = loss_fn(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        loss_sum, n_log = loss_sum + loss.item(), n_log + 1
        if step % args.log_every == 0 or step == total:
            now = time.time()
            rate = n_log * args.batch_size / (now - t_log)
            log.info(f"step {step:,}/{total:,}  loss {loss_sum / n_log:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                     f"{rate:,.0f} pairs/s  eta {(total - step) * args.batch_size / rate / 60:,.1f} min")
            t_log, loss_sum, n_log = now, 0.0, 0
        if step % args.valid_every == 0 or step == total:
            metrics = evaluate(model, tok, data["valid"], s1text, args, device)
            log.info("  valid: " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v:,}"
                                               for k, v in metrics.items()))
            model.train()
        if stop or (step % args.ckpt_every == 0 and step < total):
            save_checkpoint()
        if stop:
            log.info("SIGTERM: stopped after the checkpoint; re-run to resume")
            sys.exit(3)

    model.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")
    (out / "meta.json").write_text(json.dumps({
        "base": args.base, "text": {"both": "name | address", "name": "name", "address": "address"}[args.text]
        + " (standardized)", "text_mode": args.text, "fold": args.fold, "top": args.top,
        "max_len": args.max_len, "loss": "binary cross-entropy per pair", "steps": total,
        "train_pairs": len(pairs), "valid_pairs": len(data["valid"][0]), "valid": metrics,
        "train_seconds": round(time.time() - t0), "args": vars(args)}, indent=2))
    ckpt.unlink(missing_ok=True)
    log.info(f"model -> {out / 'model'} (+ meta.json)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="model name in artifacts/cross_encoders/")
    ap.add_argument("--fold", required=True, choices=["A", "B"], help="training fold (the other one is scored)")
    ap.add_argument("--base", default=BASE, help="Hugging Face model to fine-tune")
    ap.add_argument("--text", choices=TEXT_MODES, default="both",
                    help="what the model reads: 'name | address', the name only, or the address only "
                         "(pairs with an empty address are skipped)")
    ap.add_argument("--train-runs", nargs="+", default=TRAIN_RUNS, help="candidate runs of the training split")
    ap.add_argument("--valid-run", default="gpu-tfidf-eval-30")
    ap.add_argument("--train-queries", type=int, default=0, help="cap on the fold's queries per run (0: all)")
    ap.add_argument("--valid-queries", type=int, default=20_000)
    ap.add_argument("--top", type=int, default=5, help="fused rank cut: the candidates scored at inference")
    ap.add_argument("--rand-neg", type=int, default=2, help="random lower-ranked negatives per training query")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05, help="warmup fraction of the steps")
    ap.add_argument("--workers", type=int, default=max(1, min(4, len(os.sched_getaffinity(0)) - 1)))
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--valid-every", type=int, default=5000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = CE / args.name
    if (out / "meta.json").exists():
        sys.exit(f"{out} is already trained (meta.json present)")
    out.mkdir(parents=True, exist_ok=True)
    rc.setup_logging(f"ce-train-{args.name}")
    log.info(f"train cross-encoder {args.name}: {json.dumps(vars(args))}")
    data, s1text = build_pairs(args, out)
    train(args, data, s1text, out)


if __name__ == "__main__":
    main()
