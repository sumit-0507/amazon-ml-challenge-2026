"""Retrieve candidate Source 1 matches for Source 2/3 records from OpenSearch.

Every query record (an S2/S3 record) runs four searches against the S1 index,
all filtered to the record's country and, when its address has a recognized
state, to that state (plus states used interchangeably for the same place, e.g.
AP/TG; see address_standardization.equivalent_states). On the train/eval pairs
the state filter drops 0.018% of true matches; --no-state-filter disables it.

    tri       character trigrams      name.tri^2 + address.tri
    words     BM25 on exact words     name^2 + address
    phonetic  Double Metaphone codes  name.phonetic^2 + address.phonetic
    prefix    fuzzy words: first 2 letters exact, up to 2 edits after
              (prefix_length=2, fuzziness=2) on name words (max 12, 50 expansions
              each) + alphabetic address words of 4+ letters (max 10, 20 expansions
              each); keeps queries under OpenSearch's 1024-clause limit
Available with --methods but off by default:
    bi        character bigrams (name.bi^2 + address.bi): on a 7k eval sample it was
              42% of the query time and found only 0.077% of true matches that no
              other search found
    knn       cosine k-NN on the name embedding (name_vec): recall@16 85% on the
              same sample; skipped for now

Each search returns its top --per-search hits; the lists are fused with
reciprocal rank fusion, RRF(c) = sum_m 1 / (--rrf-k + rank_m(c)), and the top
--top candidates are kept.

Outputs in artifacts/candidates/:

<name>.per_search.parquet  every search's own list, kept separately, one row per
    (query, search, hit): query_row, method, rank, s1_id, score[, label]
logs/<name>.log  progress log (every --progress-every seconds: done/total, q/s,
    ETA, failed searches, rows written) and the final recall table;
    follow a run with:  tail -f logs/<name>.log
logs/<name>.search_matches.tsv  the same per-search hits as a readable TSV log:
    query_entity_id, country, method, rank, s1_entity_id, score, is_true_match
    (~2.5 GB for eval, ~50 GB for all of test; --no-tsv skips it)

<name>.parquet  the fused candidates, one row per (query, candidate):

    query_row    row of the query in queries.parquet (same folder, <name>.queries.parquet)
    s1_id        candidate Source 1 entity_id
    rrf, fused_rank
    <m>_score, <m>_rank    per search m (NaN when the candidate is not in that list;
                           knn score converted back to cosine)
    label        1 if the candidate is the true match (only for labelled queries)

With labels (dataset/splits/s23_*.tsv) it prints recall at each cutoff, overall
and per search.

    srun --jobid=<id> --overlap --cpus-per-task=12 python scripts/retrieve_candidates.py \\
        --queries eval --index er-train --limit 20000
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from opensearchpy import OpenSearch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
from address_standardization import equivalent_states, standardize_addresses  # noqa: E402
from name_standardization import standardize_names  # noqa: E402

DATA = ROOT / "dataset"
EMBEDDINGS = ROOT / "artifacts" / "embeddings"
OUT = ROOT / "artifacts" / "candidates"
LOGS = ROOT / "logs"
QUERY_SETS = {
    # name: (files, embedding target)
    "eval": ([DATA / "splits" / "s23_eval.tsv"], "s23-eval"),
    "train": ([DATA / "splits" / "s23_train.tsv"], "s23-train"),
    "test": ([DATA / "test" / "test_source2.tsv", DATA / "test" / "test_source3.tsv"], "s23-test"),
}
METHODS = ["bi", "tri", "words", "phonetic", "prefix", "knn"]
DEFAULT_METHODS = ["tri", "words", "phonetic", "prefix"]
log = logging.getLogger("retrieve")


def setup_logging(name):
    LOGS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOGS / f"{name}.log", mode="w")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)
CUTOFFS = [1, 2, 4, 8, 16]


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------

def _match(field, text, boost, **opts):
    return {"match": {field: {"query": text, "boost": boost, **opts}}}


def _fuzzy_words(text, max_words, min_len=1, alpha_only=False):
    words = [w for w in text.replace(",", " ").split()
             if len(w) >= min_len and (w.isalpha() or not alpha_only)]
    return " ".join(words[:max_words])


def build_searches(name, address, country, states, vector, size, name_boost):
    """The six search bodies for one record (None where the record has no usable text)."""
    flt = [{"term": {"country": country}}]
    if states:
        flt.append({"terms": {"state": states}})

    def text_search(suffix, **opts):
        should = []
        if name:
            should.append(_match(f"name{suffix}", name, name_boost, **opts))
        if address:
            should.append(_match(f"address{suffix}", address, 1.0, **opts))
        if not should:
            return None
        return {"size": size, "_source": False, "track_total_hits": False,
                "query": {"bool": {"should": should, "filter": flt}}}

    prefix = None
    fuzzy_name = _fuzzy_words(name, 12)
    fuzzy_address = _fuzzy_words(address, 10, min_len=4, alpha_only=True)
    should = []
    if fuzzy_name:
        should.append(_match("name", fuzzy_name, name_boost, fuzziness=2, prefix_length=2, max_expansions=50))
    if fuzzy_address:
        should.append(_match("address", fuzzy_address, 1.0, fuzziness=2, prefix_length=2, max_expansions=20))
    if should:
        prefix = {"size": size, "_source": False, "track_total_hits": False,
                  "query": {"bool": {"should": should, "filter": flt}}}

    knn = None
    if name and vector is not None:
        knn = {"size": size, "_source": False, "track_total_hits": False,
               "query": {"knn": {"name_vec": {"vector": vector, "k": size, "filter": {"bool": {"filter": flt}}}}}}
    return {
        "bi": text_search(".bi"),
        "tri": text_search(".tri"),
        "words": text_search(""),
        "phonetic": text_search(".phonetic"),
        "prefix": prefix,
        "knn": knn,
    }


FAILURES = {}


def run_batch(client, index, batch):
    """batch: list of dicts method -> body|None. Returns list of dicts method -> [(id, score), ...].
    A failed search is counted in FAILURES and treated as returning no hits."""
    lines, slots = [], []
    for i, searches in enumerate(batch):
        for m in METHODS:
            if searches[m] is not None:
                lines += [{"index": index}, searches[m]]
                slots.append((i, m))
    results = [{m: [] for m in METHODS} for _ in batch]
    if not lines:
        return results
    resp = client.msearch(body=lines, request_timeout=600,
                          filter_path="responses.hits.hits._id,responses.hits.hits._score,responses.status,"
                                      "responses.error.root_cause.type,responses.error.root_cause.reason")
    for (i, m), r in zip(slots, resp["responses"]):
        if "error" in r:
            cause = (r["error"].get("root_cause") or [{}])[0]
            key = (m, cause.get("type"))
            if key not in FAILURES:
                log.info(f"search {m} failed: {cause.get('type')}: {str(cause.get('reason'))[:200]}")
            FAILURES[key] = FAILURES.get(key, 0) + 1
            continue
        results[i][m] = [(h["_id"], h["_score"]) for h in r.get("hits", {}).get("hits", [])]
    return results


def fuse(lists, top, rrf_k):
    """RRF over the per-method lists -> list of (s1_id, rrf, {m: (score, rank)})."""
    per_cand = {}
    for m, hits in lists.items():
        for rank, (cid, score) in enumerate(hits, start=1):
            entry = per_cand.setdefault(cid, [0.0, {}])
            entry[0] += 1.0 / (rrf_k + rank)
            entry[1][m] = (score, rank)
    ranked = sorted(per_cand.items(), key=lambda kv: -kv[1][0])[:top]
    return [(cid, rrf, feats) for cid, (rrf, feats) in ranked]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def load_queries(args):
    files, emb_target = QUERY_SETS[args.queries]
    df = pd.concat([pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False) for f in files], ignore_index=True)
    vectors = None
    emb_path = EMBEDDINGS / f"{emb_target}.npy"
    if "knn" in args.methods:
        if not emb_path.exists():
            sys.exit(f"missing {emb_path}; run scripts/embed_names.py {emb_target}")
        vectors = np.load(emb_path, mmap_mode="r")
        if not np.array_equal(np.load(EMBEDDINGS / f"{emb_target}.ids.npy"), df.entity_id.to_numpy(dtype=str)):
            sys.exit(f"{emb_path} does not match the query files; re-run embed_names.py")
    rows = np.arange(len(df))
    if args.limit and args.limit < len(df):
        rows = np.sort(np.random.default_rng(args.seed).choice(len(df), args.limit, replace=False))
    return df.iloc[rows].reset_index(drop=True), rows, vectors


def schema(labelled):
    fields = [("query_row", pa.int32()), ("s1_id", pa.string()), ("rrf", pa.float32()), ("fused_rank", pa.int16())]
    for m in METHODS:
        fields += [(f"{m}_score", pa.float32()), (f"{m}_rank", pa.float32())]
    if labelled:
        fields.append(("label", pa.int8()))
    return pa.schema(fields)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", choices=list(QUERY_SETS), default="eval")
    ap.add_argument("--index", default="er-train")
    ap.add_argument("--name", help="output name (default: <queries>[-<limit>])")
    ap.add_argument("--limit", type=int, help="random sample of N query records")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=DEFAULT_METHODS)
    ap.add_argument("--per-search", type=int, default=16, help="hits taken from each search")
    ap.add_argument("--top", type=int, default=100,
                    help="candidates kept after fusion (4 searches x 16 hits is at most 64, so all of them)")
    ap.add_argument("--rrf-k", type=float, default=60.0)
    ap.add_argument("--name-boost", type=float, default=2.0, help="weight of name vs address in text searches")
    ap.add_argument("--no-state-filter", action="store_true", help="filter by country only")
    ap.add_argument("--no-tsv", action="store_true", help="skip the per-search TSV log")
    ap.add_argument("--batch", type=int, default=32, help="query records per msearch request")
    ap.add_argument("--threads", type=int, default=12, help="concurrent msearch requests")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=int(os.environ.get("OS_PORT", 9277)))
    ap.add_argument("--progress-every", type=float, default=15.0, help="seconds between progress log lines")
    args = ap.parse_args()
    name = args.name or (f"{args.queries}-{args.limit}" if args.limit else args.queries)
    setup_logging(name)
    log.info(f"run {name}: queries={args.queries} index={args.index} methods={','.join(args.methods)} "
             f"per_search={args.per_search} top={args.top} state_filter={not args.no_state_filter} "
             f"batch={args.batch} threads={args.threads}")

    t0 = time.time()
    df, rows, vectors = load_queries(args)
    n_jobs = len(os.sched_getaffinity(0))
    names = standardize_names(df.business_name, n_jobs=n_jobs)
    parts = standardize_addresses(df.business_address, df.country, with_state=True, n_jobs=n_jobs)
    addresses = [a for a, _ in parts]
    states = [[] if args.no_state_filter else equivalent_states(st, c) for (_, st), c in zip(parts, df.country)]
    labels = df.source1_entity_id.to_numpy() if "source1_entity_id" in df else None
    log.info(f"{len(df):,} queries loaded and standardized in {time.time() - t0:.0f}s "
             f"({sum(bool(x) for x in states):,} with a state filter)")

    OUT.mkdir(parents=True, exist_ok=True)
    qcols = ["entity_id", "country"] + (["source1_entity_id"] if labels is not None else [])
    df[qcols].assign(query_row=np.arange(len(df), dtype=np.int32), file_row=rows).to_parquet(
        OUT / f"{name}.queries.parquet", index=False)
    writer = pq.ParquetWriter(OUT / f"{name}.parquet", schema(labels is not None))
    per_search_schema = pa.schema([("query_row", pa.int32()), ("method", pa.dictionary(pa.int8(), pa.string())),
                                   ("rank", pa.int16()), ("s1_id", pa.string()), ("score", pa.float32())]
                                  + ([("label", pa.int8())] if labels is not None else []))
    per_search_writer = pq.ParquetWriter(OUT / f"{name}.per_search.parquet", per_search_schema)
    tsv = None
    if not args.no_tsv:
        LOGS.mkdir(parents=True, exist_ok=True)
        tsv = open(LOGS / f"{name}.search_matches.tsv", "w", encoding="utf-8")
        tsv.write("query_entity_id\tcountry\tmethod\trank\ts1_entity_id\tscore\tis_true_match\n")
    query_ids, countries = df.entity_id.to_numpy(), df.country.to_numpy()

    client = OpenSearch(hosts=[{"host": args.host, "port": args.port}], timeout=600, maxsize=args.threads + 2)
    methods = set(args.methods)

    def make_batch(start):
        batch = []
        for q in range(start, min(start + args.batch, len(df))):
            vec = vectors[rows[q]].astype(np.float32).tolist() if vectors is not None else None
            s = build_searches(names[q], addresses[q], df.country.iat[q], states[q], vec,
                               args.per_search, args.name_boost)
            batch.append({m: (s[m] if m in methods else None) for m in METHODS})
        return start, run_batch(client, args.index, batch)

    # true-match rank per labelled query: fused and per method (0 = not found)
    found = {k: np.zeros(len(df), dtype=np.int16) for k in ["fused"] + METHODS}
    has_match = np.array([bool(x) for x in labels]) if labels is not None else None

    t1, done, n_rows = time.time(), 0, 0
    next_report = t1 + args.progress_every
    with ThreadPoolExecutor(args.threads) as pool:
        starts = range(0, len(df), args.batch)
        for start, results in pool.map(make_batch, starts):
            cols = {f.name: [] for f in schema(labels is not None)}
            per_search = {f.name: [] for f in per_search_schema}
            lines = []
            for offset, lists in enumerate(results):
                q = start + offset
                truth = labels[q] if labels is not None else ""
                for m in METHODS:
                    for rank, (cid, score) in enumerate(lists[m], start=1):
                        if cid == truth:
                            found[m][q] = rank
                        per_search["query_row"].append(q)
                        per_search["method"].append(m)
                        per_search["rank"].append(rank)
                        per_search["s1_id"].append(cid)
                        per_search["score"].append(2 * score - 1 if m == "knn" else score)
                        if labels is not None:
                            per_search["label"].append(int(cid == truth))
                        if tsv is not None:
                            is_true = "" if labels is None else int(cid == truth)
                            lines.append(f"{query_ids[q]}\t{countries[q]}\t{m}\t{rank}\t{cid}\t"
                                         f"{per_search['score'][-1]:.4f}\t{is_true}\n")
                for fused_rank, (cid, rrf, feats) in enumerate(fuse(lists, args.top, args.rrf_k), start=1):
                    cols["query_row"].append(q)
                    cols["s1_id"].append(cid)
                    cols["rrf"].append(rrf)
                    cols["fused_rank"].append(fused_rank)
                    for m in METHODS:
                        score, rank = feats.get(m, (np.nan, np.nan))
                        if m == "knn" and score == score:
                            score = 2 * score - 1  # OpenSearch cosinesimil score is (1 + cos) / 2
                        cols[f"{m}_score"].append(score)
                        cols[f"{m}_rank"].append(rank)
                    if labels is not None:
                        cols["label"].append(int(cid == truth))
                        if cid == truth:
                            found["fused"][q] = fused_rank
            writer.write_table(pa.table(cols, schema=schema(labels is not None)))
            per_search_writer.write_table(pa.table(per_search, schema=per_search_schema))
            if tsv is not None:
                tsv.writelines(lines)
            n_rows += len(cols["query_row"])
            done += len(results)
            now = time.time()
            if now >= next_report or done == len(df):
                rate = done / (now - t1)
                log.info(f"progress {done:,}/{len(df):,} ({done / len(df):.1%})  {rate:,.1f} q/s  "
                         f"eta {(len(df) - done) / rate / 60:,.1f} min  failed searches {sum(FAILURES.values()):,}  "
                         f"fused rows {n_rows:,}")
                next_report = now + args.progress_every
    writer.close()
    per_search_writer.close()
    if tsv is not None:
        tsv.close()
        log.info(f"per-search TSV log -> {LOGS / name}.search_matches.tsv")
    if FAILURES:
        log.info("failed searches (treated as no hits): "
              + ", ".join(f"{m}/{t}: {n:,}" for (m, t), n in FAILURES.items()))
    log.info(f"per-search lists -> {OUT / name}.per_search.parquet")
    log.info(f"{n_rows:,} fused candidate rows -> {OUT / name}.parquet  ({time.time() - t1:,.0f}s, "
             f"{len(df) / (time.time() - t1):,.1f} q/s)")

    if labels is not None:
        n = has_match.sum()
        cutoffs = [k for k in CUTOFFS if k <= args.per_search]
        log.info(f"recall on {n:,} queries with a true S1 match ({len(df) - n:,} without one):")
        log.info(f"{'':10}" + "".join(f"{'@' + str(k):>8}" for k in cutoffs) + f"{'all':>8}")
        for key in ["fused"] + [m for m in METHODS if m in methods]:
            r = found[key][has_match]
            log.info(f"{key:10}" + "".join(f"{((r > 0) & (r <= k)).mean():>8.2%}" for k in cutoffs)
                     + f"{(r > 0).mean():>8.2%}")
        log.info(f"fused candidates per query: mean {n_rows / len(df):.1f}")


if __name__ == "__main__":
    main()
