"""GPU candidate retrieval with TF-IDF cosine or BM25 scoring.

Same job as retrieve_candidates.py (OpenSearch), done as sparse matrix products
on the GPU. For each search, name and address are turned into term vectors
(vocabulary and statistics fitted on Source 1):

    bi        character bigrams inside words
    tri       character trigrams inside words
    words     words (split on spaces and commas)
    phonetic  Double Metaphone codes of the words (primary + alternate, 4 chars,
              as OpenSearch's phonetic filter)
  optional (--methods):
    skip      character skip-bigrams inside words (letters i and i+2: "trading" ->
              ta rd aa di in ng); survive swapped / inserted / dropped letters
    phonkey   Indian-aware phonetic key per word (ph->f, bh->b, kh->k, gh->g, th->t,
              dh->d, sh->s, ck/q->k, x->ks, z->j, w->v, c->k, ...; doubled letters
              collapsed, vowels and h dropped after the first letter)

--scoring tfidf  sublinear-tf TF-IDF, L2-normalized: field score = cosine
--scoring bm25   Lucene BM25 (k1=1.2, b=0.75, idf = ln(1 + (N-df+0.5)/(df+0.5))):
                 S1 side holds the BM25 term weights, query side the term counts,
                 so one product gives the same score as OpenSearch (up to its
                 per-shard statistics)

score(query, s1) = name_boost * name_score + address_score, computed for every S1
record in the query's block: same country and, when the query has a recognized
state, the same state group (address_standardization.equivalent_states), exactly
like the OpenSearch filter. Each search keeps its top --per-search hits with a
score > 0; the lists are fused with RRF like retrieve_candidates.py.

Outputs have the same format as retrieve_candidates.py (artifacts/candidates/,
logs/<name>.log, logs/<name>.search_matches.tsv), so either can feed XGBoost,
plus the name and address scores kept separately for every hit:
    per_search parquet / TSV log   name_score, address_score
    fused parquet                  <search>_name_score, <search>_address_score
Source 1 matrices are cached in artifacts/tfidf/s1-<split>/<scoring>/.

    srun --jobid=<id> --overlap --gres=gpu:2g.48gb:1 --cpus-per-task=12 \\
        python scripts/gpu_retrieve.py --queries eval --limit 20000 --name gpu-eval-20k
    ... --queries train --frac 0.3        # 30% of training S1 entities and their records
    ... --queries test --parts 4 --part 0 --no-tsv   # test in 4 memory-sized parts (0..3)
    ... --methods bi tri skip words phonkey --top 20      # choose the searches, cap the fused list
"""

import argparse
import os
import re
import sys
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from metaphone import doublemetaphone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "scripts"))
import retrieve_candidates as rc  # noqa: E402  (query sets, RRF fusion, output schema, logging)
from address_standardization import equivalent_states, standardize_addresses  # noqa: E402
from name_standardization import standardize_names  # noqa: E402

METHODS = ["bi", "tri", "words", "phonetic"]              # default searches (the trained model's)
ALL_METHODS = METHODS + ["skip", "phonkey"]
FIELDS = ["name", "address"]
TFIDF = ROOT / "artifacts" / "tfidf"
log = rc.log

# --------------------------------------------------------------------------
# Tokenizers (mirror the OpenSearch analyzers)
# --------------------------------------------------------------------------

_WORD_CHARS = re.compile(r"[^\W_]+")


def tok_words(s):
    return s.replace(",", " ").split()


def _ngrams(s, n):
    return [w[i:i + n] for w in _WORD_CHARS.findall(s) if len(w) >= n for i in range(len(w) - n + 1)]


def tok_bi(s):
    return _ngrams(s, 2)


def tok_tri(s):
    return _ngrams(s, 3)


@lru_cache(maxsize=1 << 21)
def _phonetic_codes(word):
    return tuple(dict.fromkeys(code[:4] for code in doublemetaphone(word) if code))


def tok_phonetic(s):
    return [code for w in tok_words(s) for code in _phonetic_codes(w)]


def tok_skip(s):
    return [w[i] + w[i + 2] for w in _WORD_CHARS.findall(s) if len(w) >= 3 for i in range(len(w) - 2)]


_PHONKEY_RULES = [("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"),
                  ("sh", "s"), ("ch", "C"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("z", "j"),
                  ("w", "v"), ("y", "i"), ("ee", "i"), ("oo", "u"), ("c", "k"), ("C", "c")]
_DOUBLE = re.compile(r"(.)\1+")
_LATER_VOWELS = re.compile(r"(?<=.)[aeiouh]")   # vowels and aspiration h after the first letter


@lru_cache(maxsize=1 << 21)
def _phonkey(word):
    if not word.isalpha():
        return word
    for a, b in _PHONKEY_RULES:
        word = word.replace(a, b)
    return _LATER_VOWELS.sub("", _DOUBLE.sub(r"\1", word))


def tok_phonkey(s):
    return [_phonkey(w) for w in tok_words(s)]


TOKENIZERS = {"bi": tok_bi, "tri": tok_tri, "words": tok_words, "phonetic": tok_phonetic,
              "skip": tok_skip, "phonkey": tok_phonkey}

# --------------------------------------------------------------------------
# Source 1 matrices (fit once, cached)
# --------------------------------------------------------------------------


BM25_K1, BM25_B = 1.2, 0.75


def bm25_weights(counts):
    """Per (document, term) BM25 weight, so that score = query_counts @ weights.T."""
    X = counts.tocsr().astype(np.float32)
    n_docs = X.shape[0]
    df = np.bincount(X.indices, minlength=X.shape[1])
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    doc_len = np.asarray(X.sum(axis=1)).ravel()
    norm = BM25_K1 * (1 - BM25_B + BM25_B * doc_len / max(doc_len.mean(), 1e-9))
    tf = X.data
    X.data = idf[X.indices] * tf * (BM25_K1 + 1) / (tf + np.repeat(norm, np.diff(X.indptr)))
    return X


def _fit_one(job):
    method, field, texts, out, scoring = job
    if scoring == "bm25":
        vec = CountVectorizer(analyzer=TOKENIZERS[method], dtype=np.float32)
        X = bm25_weights(vec.fit_transform(texts))
    else:
        vec = TfidfVectorizer(analyzer=TOKENIZERS[method], sublinear_tf=True, dtype=np.float32)
        X = vec.fit_transform(texts).tocsr()
    sp.save_npz(out / f"{method}_{field}.npz", X, compressed=False)
    joblib.dump(vec, out / f"{method}_{field}.vectorizer.joblib")
    return method, field, X.shape, X.nnz


def build_s1(split, n_jobs, scoring, methods=METHODS, rebuild=False):
    """Fit (and cache) the S1 matrices; only (method, field) pairs not cached yet are built."""
    out = TFIDF / f"s1-{split}" / scoring
    meta_path = out / "meta.parquet"
    todo = [(m, f) for m in methods for f in FIELDS
            if rebuild or not (out / f"{m}_{f}.npz").exists() or not (out / f"{m}_{f}.vectorizer.joblib").exists()]
    if meta_path.exists() and not todo:
        return out
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    s1 = pd.read_csv(ROOT / "dataset" / split / f"{split}_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    names = standardize_names(s1.business_name, n_jobs=n_jobs)
    parts = standardize_addresses(s1.business_address, s1.country, with_state=True, n_jobs=n_jobs)
    texts = {"name": names, "address": [a for a, _ in parts]}
    jobs = [(m, f, texts[f], out, scoring) for m, f in todo]
    with Pool(max(1, min(n_jobs, len(jobs)))) as pool:
        for method, field, shape, nnz in pool.imap_unordered(_fit_one, jobs):
            log.info(f"  s1-{split} {method}/{field}: {shape[0]:,} x {shape[1]:,}, nnz {nnz / 1e6:.1f}M")
    pd.DataFrame({"entity_id": s1.entity_id, "country": s1.country,
                  "state": [st for _, st in parts]}).to_parquet(meta_path, index=False)
    log.info(f"built S1 {scoring} matrices for {split} in {time.time() - t:.0f}s -> {out}")
    return out


def _transform_one(job):
    method, field, texts, s1_dir = job
    vec = joblib.load(s1_dir / f"{method}_{field}.vectorizer.joblib")
    return method, field, vec.transform(texts).tocsr()

# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------


def load_queries(args):
    files, _ = rc.QUERY_SETS[args.queries]
    df = pd.concat([pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False) for f in files], ignore_index=True)
    rng = np.random.default_rng(args.seed)
    keep = np.ones(len(df), dtype=bool)
    if args.frac is not None:
        if "source1_entity_id" in df:  # sample S1 entities, keep all their records; unmatched at the same rate
            s1_ids = df.source1_entity_id[df.source1_entity_id != ""].unique()
            chosen = set(s1_ids[rng.random(len(s1_ids)) < args.frac])
            matched = df.source1_entity_id != ""
            keep = np.where(matched, df.source1_entity_id.isin(chosen), rng.random(len(df)) < args.frac)
        else:
            keep = rng.random(len(df)) < args.frac
    rows = np.flatnonzero(keep)
    if args.limit and args.limit < len(rows):
        rows = np.sort(rng.choice(rows, args.limit, replace=False))
    if args.parts > 1:
        rows = np.array_split(rows, args.parts)[args.part]
    return df.iloc[rows].reset_index(drop=True), rows

# --------------------------------------------------------------------------
# GPU scoring
# --------------------------------------------------------------------------


def _local(S_block, torch, device):
    """Restrict both matrices to the terms present in the block; return GPU CSR and a column map."""
    cols = np.unique(S_block.indices)
    lookup = np.full(S_block.shape[1], -1, dtype=np.int64)
    lookup[cols] = np.arange(len(cols))
    S = torch.sparse_csr_tensor(torch.from_numpy(S_block.indptr.astype(np.int64)),
                                torch.from_numpy(lookup[S_block.indices]),
                                torch.from_numpy(S_block.data), size=(S_block.shape[0], len(cols)),
                                device=device)
    return S, lookup, len(cols)


def _dense_queries(Q, lookup, n_cols, torch, device):
    """(n_cols x n_queries) dense matrix of the query vectors on the block's terms."""
    q_of_entry = np.repeat(np.arange(Q.shape[0]), np.diff(Q.indptr))
    local = lookup[Q.indices]
    ok = local >= 0
    D = torch.zeros((n_cols, Q.shape[0]), dtype=torch.float32, device=device)
    if ok.any():
        D[torch.from_numpy(local[ok]).to(device), torch.from_numpy(q_of_entry[ok]).to(device)] = \
            torch.from_numpy(Q.data[ok]).to(device)
    return D


def search_block(S1, Qm, s1_rows, q_rows, args, torch, device, methods=METHODS):
    """Top hits per method for the queries q_rows against S1 rows s1_rows.
    Returns {method: (indices into s1_rows, combined, name and address scores)}, each [k x nq]."""
    k = min(args.per_search, len(s1_rows))
    batch = int(np.clip(2e9 / (4 * max(len(s1_rows), 1)), 64, 4096))
    out = {}
    for m in methods:
        mats = {}
        for f in FIELDS:
            S, lookup, n_cols = _local(S1[m, f][s1_rows], torch, device)
            mats[f] = (S, lookup, n_cols)
        parts_out = {"idx": [], "score": [], "name": [], "address": []}
        for b0 in range(0, len(q_rows), batch):
            qb = q_rows[b0:b0 + batch]
            field_scores = {}
            for f in FIELDS:
                S, lookup, n_cols = mats[f]
                if n_cols == 0:
                    field_scores[f] = torch.zeros((len(s1_rows), len(qb)), device=device)
                else:
                    field_scores[f] = torch.sparse.mm(S, _dense_queries(Qm[m, f][qb], lookup, n_cols, torch, device))
            score = field_scores["name"] * args.name_boost + field_scores["address"]
            vals, idx = torch.topk(score, k, dim=0)
            parts_out["idx"].append(idx.cpu().numpy())
            parts_out["score"].append(vals.cpu().numpy())
            for f in FIELDS:
                parts_out[f].append(torch.gather(field_scores[f], 0, idx).cpu().numpy())
            del score, field_scores
        out[m] = tuple(np.concatenate(parts_out[key], axis=1) for key in ("idx", "score", "name", "address"))
        del mats
    return out

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", choices=list(rc.QUERY_SETS), default="eval")
    ap.add_argument("--s1", choices=["train", "test"], help="Source 1 split to search (default: test for test queries, else train)")
    ap.add_argument("--name", help="output name (default: gpu-<queries>[-<frac|limit>])")
    ap.add_argument("--frac", type=float, help="sample this fraction of S1 entities (labelled sets) or records")
    ap.add_argument("--limit", type=int, help="then cap at N random query records")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1, help="split the queries into this many contiguous parts")
    ap.add_argument("--part", type=int, default=0, help="which part to run (0-based)")
    ap.add_argument("--per-search", type=int, default=16)
    ap.add_argument("--top", type=int, default=100, help="fused candidates kept (default keeps all of them)")
    ap.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=METHODS,
                    help="searches to run (default: the ones the trained model uses)")
    ap.add_argument("--rrf-k", type=float, default=60.0)
    ap.add_argument("--name-boost", type=float, default=2.0)
    ap.add_argument("--scoring", choices=["tfidf", "bm25"], default="tfidf")
    ap.add_argument("--no-state-filter", action="store_true")
    ap.add_argument("--no-tsv", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the cached S1 TF-IDF matrices")
    ap.add_argument("--progress-every", type=float, default=15.0)
    args = ap.parse_args()
    s1_split = args.s1 or ("test" if args.queries == "test" else "train")
    suffix = f"-{args.frac:g}" if args.frac else (f"-{args.limit}" if args.limit else "")
    if args.parts > 1:
        suffix += f"-part{args.part}of{args.parts}"
    name = args.name or f"gpu-{args.scoring}-{args.queries}{suffix}"
    rc.setup_logging(name)
    n_jobs = len(os.sched_getaffinity(0))

    methods = list(dict.fromkeys(args.methods))
    extra = [m for m in methods if m not in rc.METHODS]      # searches without a column in rc.schema
    score_methods = rc.METHODS + extra
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"run {name}: scoring={args.scoring} queries={args.queries} s1={s1_split} "
             f"methods={','.join(methods)} per_search={args.per_search} top={args.top} name_boost={args.name_boost} "
             f"frac={args.frac} limit={args.limit} state_filter={not args.no_state_filter} device={device}"
             + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # Source 1
    s1_dir = build_s1(s1_split, n_jobs, args.scoring, methods, args.rebuild)
    meta = pd.read_parquet(s1_dir / "meta.parquet")
    S1 = {(m, f): sp.load_npz(s1_dir / f"{m}_{f}.npz").tocsr() for m in methods for f in FIELDS}
    s1_ids = meta.entity_id.to_numpy()

    # Queries
    t0 = time.time()
    df, file_rows = load_queries(args)
    names = standardize_names(df.business_name, n_jobs=n_jobs)
    parts = standardize_addresses(df.business_address, df.country, with_state=True, n_jobs=n_jobs)
    texts = {"name": names, "address": [a for a, _ in parts]}
    jobs = [(m, f, texts[f], s1_dir) for m in methods for f in FIELDS]
    with Pool(min(n_jobs, len(jobs))) as pool:
        Qm = {(m, f): X for m, f, X in pool.imap_unordered(_transform_one, jobs)}
    states = [() if args.no_state_filter else tuple(equivalent_states(st, c)) for (_, st), c in zip(parts, df.country)]
    labels = df.source1_entity_id.to_numpy() if "source1_entity_id" in df else None
    log.info(f"{len(df):,} queries loaded, standardized and vectorized in {time.time() - t0:.0f}s "
             f"({sum(bool(s) for s in states):,} with a state filter)")

    # Outputs (same layout as retrieve_candidates.py)
    rc.OUT.mkdir(parents=True, exist_ok=True)
    qcols = ["entity_id", "country"] + (["source1_entity_id"] if labels is not None else [])
    df[qcols].assign(query_row=np.arange(len(df), dtype=np.int32), file_row=file_rows).to_parquet(
        rc.OUT / f"{name}.queries.parquet", index=False)
    fused_schema = rc.schema(labels is not None)
    for m in extra:
        fused_schema = fused_schema.append(pa.field(f"{m}_score", pa.float32()))
        fused_schema = fused_schema.append(pa.field(f"{m}_rank", pa.float32()))
    for m in methods:
        for f in FIELDS:
            fused_schema = fused_schema.append(pa.field(f"{m}_{f}_score", pa.float32()))
    writer = pq.ParquetWriter(rc.OUT / f"{name}.parquet", fused_schema)
    ps_schema = pa.schema([("query_row", pa.int32()), ("method", pa.dictionary(pa.int8(), pa.string())),
                           ("rank", pa.int16()), ("s1_id", pa.string()), ("score", pa.float32()),
                           ("name_score", pa.float32()), ("address_score", pa.float32())]
                          + ([("label", pa.int8())] if labels is not None else []))
    ps_writer = pq.ParquetWriter(rc.OUT / f"{name}.per_search.parquet", ps_schema)
    tsv = None
    if not args.no_tsv:
        tsv = open(rc.LOGS / f"{name}.search_matches.tsv", "w", encoding="utf-8")
        tsv.write("query_entity_id\tcountry\tmethod\trank\ts1_entity_id\tscore\tname_score\taddress_score"
                  "\tis_true_match\n")
    query_ids, countries = df.entity_id.to_numpy(), df.country.to_numpy()

    # Blocks: queries grouped by (country, state group); S1 rows of the same country (and state group)
    groups = pd.Series(range(len(df))).groupby([df.country.to_numpy(), pd.Series(states).to_numpy()])
    s1_by_country = {c: np.flatnonzero(meta.country.to_numpy() == c) for c in meta.country.unique()}
    s1_state = meta.state.to_numpy()

    found = {k: np.zeros(len(df), dtype=np.int16) for k in ["fused"] + methods}
    t1, done, n_rows, next_report = time.time(), 0, 0, time.time() + args.progress_every
    for (country, group_states), q_series in groups:
        q_rows = q_series.to_numpy()
        s1_rows = s1_by_country.get(country, np.array([], dtype=np.int64))
        if group_states:
            s1_rows = s1_rows[np.isin(s1_state[s1_rows], list(group_states))]
        hits = search_block(S1, Qm, s1_rows, q_rows, args, torch, device, methods) if len(s1_rows) else {}

        cols = {f.name: [] for f in fused_schema}
        ps = {f.name: [] for f in ps_schema}
        lines = []
        for j, q in enumerate(q_rows):
            truth = labels[q] if labels is not None else ""
            lists = {m: [] for m in score_methods}
            for m, (idx, vals, name_vals, addr_vals) in hits.items():
                kept = [(i, v, nv, av) for i, v, nv, av in zip(idx[:, j], vals[:, j], name_vals[:, j], addr_vals[:, j])
                        if v > 0]
                for rank, (i, v, nv, av) in enumerate(kept, start=1):
                    cid = s1_ids[s1_rows[i]]
                    lists[m].append((cid, (float(v), float(nv), float(av))))
                    if cid == truth:
                        found[m][q] = rank
                    ps["query_row"].append(q)
                    ps["method"].append(m)
                    ps["rank"].append(rank)
                    ps["s1_id"].append(cid)
                    ps["score"].append(float(v))
                    ps["name_score"].append(float(nv))
                    ps["address_score"].append(float(av))
                    if labels is not None:
                        ps["label"].append(int(cid == truth))
                    if tsv is not None:
                        is_true = "" if labels is None else int(cid == truth)
                        lines.append(f"{query_ids[q]}\t{countries[q]}\t{m}\t{rank}\t{cid}\t{v:.4f}\t{nv:.4f}\t"
                                     f"{av:.4f}\t{is_true}\n")
            for fused_rank, (cid, rrf, feats) in enumerate(rc.fuse(lists, args.top, args.rrf_k), start=1):
                cols["query_row"].append(q)
                cols["s1_id"].append(cid)
                cols["rrf"].append(rrf)
                cols["fused_rank"].append(fused_rank)
                for m in score_methods:
                    (score, name_score, addr_score), rank = feats.get(m, ((np.nan, np.nan, np.nan), np.nan))
                    cols[f"{m}_score"].append(score)
                    cols[f"{m}_rank"].append(rank)
                    if m in methods:
                        cols[f"{m}_name_score"].append(name_score)
                        cols[f"{m}_address_score"].append(addr_score)
                if labels is not None:
                    cols["label"].append(int(cid == truth))
                    if cid == truth:
                        found["fused"][q] = fused_rank
        writer.write_table(pa.table(cols, schema=fused_schema))
        ps_writer.write_table(pa.table(ps, schema=ps_schema))
        if tsv is not None:
            tsv.writelines(lines)
        n_rows += len(cols["query_row"])
        done += len(q_rows)
        now = time.time()
        if now >= next_report or done == len(df):
            rate = done / (now - t1)
            log.info(f"progress {done:,}/{len(df):,} ({done / len(df):.1%})  {rate:,.0f} q/s  "
                     f"eta {(len(df) - done) / rate / 60:,.1f} min  fused rows {n_rows:,}")
            next_report = now + args.progress_every

    writer.close()
    ps_writer.close()
    if tsv is not None:
        tsv.close()
        log.info(f"per-search TSV log -> {rc.LOGS / name}.search_matches.tsv")
    log.info(f"per-search lists -> {rc.OUT / name}.per_search.parquet")
    log.info(f"{n_rows:,} fused candidate rows -> {rc.OUT / name}.parquet  ({time.time() - t1:,.0f}s, "
             f"{len(df) / (time.time() - t1):,.0f} q/s)")

    if labels is not None:
        has_match = np.array([bool(x) for x in labels])
        n = has_match.sum()
        cutoffs = [c for c in rc.CUTOFFS if c <= args.per_search]
        log.info(f"recall on {n:,} queries with a true S1 match ({len(df) - n:,} without one):")
        log.info(f"{'':10}" + "".join(f"{'@' + str(c):>8}" for c in cutoffs) + f"{'all':>8}")
        for key in ["fused"] + methods:
            r = found[key][has_match]
            log.info(f"{key:10}" + "".join(f"{((r > 0) & (r <= c)).mean():>8.2%}" for c in cutoffs)
                     + f"{(r > 0).mean():>8.2%}")
        log.info(f"fused candidates per query: mean {n_rows / len(df):.1f}")


if __name__ == "__main__":
    main()
