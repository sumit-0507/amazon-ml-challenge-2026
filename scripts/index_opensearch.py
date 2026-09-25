"""Index Source 1 records into OpenSearch for fuzzy matching.

Source 1 is the deduplicated reference: every Source 2/3 record is a noisy copy
of at most one Source 1 entity (checked on the training ground truth), so S1 is
indexed and each S2/S3 record is a query that needs only its top few hits.
One index per split (``er-train``, ``er-test``). Each document carries the raw
and standardized name and address (utils/*_standardization.py):

    name, address   standardized text, each searchable five ways:
        <field>           words
        <field>.bi        character bigrams
        <field>.tri       character trigrams
        <field>.shingle   word bigrams ("anb trading", "trading pvt")
        <field>.phonetic  Double Metaphone codes (needs the analysis-phonetic plugin)
    name.exact      whole standardized name (keyword)
    name_vec        name embedding for cosine k-NN (HNSW, faiss); from
                    artifacts/embeddings/s1-<split>.npy (scripts/embed_names.py)
    name_core       name without legal forms (pvt, ltd, inc, llc, sarl, ...)
    addr_numbers    numbers in the address (house/plot/sector numbers)
    country, state, source   keyword filters
    name_raw, address_raw    stored only, for inspection

Run inside the OpenSearch job (the server listens on the node's localhost):

    srun --jobid=<job id> --overlap python scripts/index_opensearch.py --split train
    srun --jobid=<job id> --overlap python scripts/index_opensearch.py --split train test --recreate
    srun --jobid=<job id> --overlap python scripts/index_opensearch.py --split train --limit 200000 --index-prefix er-smoke
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from opensearchpy import OpenSearch, helpers

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
from address_standardization import standardize_addresses  # noqa: E402
from name_standardization import standardize_names  # noqa: E402

DATASET = ROOT / "dataset"
EMBEDDINGS = ROOT / "artifacts" / "embeddings"
EMBED_DIM = 384
_NUMBER = re.compile(r"\d+")

_WORDS = {"type": "text", "analyzer": "words"}
_SEARCH_FIELDS = {
    "bi": {"type": "text", "analyzer": "bigram"},
    "tri": {"type": "text", "analyzer": "trigram"},
    "shingle": {"type": "text", "analyzer": "shingle"},
    "phonetic": {"type": "text", "analyzer": "phonetic"},
}

INDEX_BODY = {
    "settings": {
        "index": {
            "number_of_shards": 4,
            "number_of_replicas": 0,
            "refresh_interval": "-1",  # re-enabled after the bulk load
            "knn": True,
        },
        "analysis": {
            "tokenizer": {
                # standardized text only has spaces and commas as separators
                "words": {"type": "pattern", "pattern": "[\\s,]+"},
                "bigram": {"type": "ngram", "min_gram": 2, "max_gram": 2,
                           "token_chars": ["letter", "digit"]},
                "trigram": {"type": "ngram", "min_gram": 3, "max_gram": 3,
                            "token_chars": ["letter", "digit"]},
            },
            "filter": {
                "word_pairs": {"type": "shingle", "min_shingle_size": 2, "max_shingle_size": 2,
                               "output_unigrams": False},
                "metaphone": {"type": "phonetic", "encoder": "double_metaphone", "replace": True},
            },
            "analyzer": {
                "words": {"type": "custom", "tokenizer": "words", "filter": ["lowercase"]},
                "bigram": {"type": "custom", "tokenizer": "bigram", "filter": ["lowercase"]},
                "trigram": {"type": "custom", "tokenizer": "trigram", "filter": ["lowercase"]},
                "shingle": {"type": "custom", "tokenizer": "words", "filter": ["lowercase", "word_pairs"]},
                "phonetic": {"type": "custom", "tokenizer": "words", "filter": ["lowercase", "metaphone"]},
            },
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "entity_id": {"type": "keyword"},
            "source": {"type": "keyword"},
            "country": {"type": "keyword"},
            "state": {"type": "keyword"},
            "name": {**_WORDS, "fields": {**_SEARCH_FIELDS,
                                          "exact": {"type": "keyword", "ignore_above": 256}}},
            "name_core": _WORDS,
            "address": {**_WORDS, "fields": _SEARCH_FIELDS},
            "addr_numbers": {"type": "keyword"},
            "name_vec": {
                "type": "knn_vector",
                "dimension": EMBED_DIM,
                "space_type": "cosinesimil",
                "method": {"name": "hnsw", "engine": "faiss",
                           "parameters": {"m": 16, "ef_construction": 128}},
            },
            "name_raw": {"type": "keyword", "index": False},
            "address_raw": {"type": "keyword", "index": False},
        },
    },
}


def source_files(split, sources):
    return [DATASET / split / f"{split}_source{i}.tsv" for i in sources]


def load_vectors(split, sources):
    """Name embeddings aligned with the Source 1 file, or (None, None) if not indexing S1 / not embedded."""
    path = EMBEDDINGS / f"s1-{split}.npy"
    if list(sources) != [1] or not path.exists():
        return None, None
    return np.load(path, mmap_mode="r"), np.load(EMBEDDINGS / f"s1-{split}.ids.npy")


def build_docs(chunk, index, n_jobs, vectors=None):
    names = standardize_names(chunk.business_name, n_jobs=n_jobs)
    cores = standardize_names(chunk.business_name, drop_legal=True, n_jobs=n_jobs)
    addresses = standardize_addresses(chunk.business_address, chunk.country, with_state=True, n_jobs=n_jobs)
    for i, (row, name, core, (address, state)) in enumerate(
            zip(chunk.itertuples(index=False), names, cores, addresses)):
        doc = {
            "_index": index,
            "_id": row.entity_id,
            "_source": {
                "entity_id": row.entity_id,
                "source": row.entity_id.split("-", 1)[0],
                "country": row.country,
                "state": state,
                "name": name,
                "name_core": core,
                "address": address,
                "addr_numbers": sorted(set(_NUMBER.findall(address))),
                "name_raw": row.business_name,
                "address_raw": row.business_address,
            },
        }
        if vectors is not None:
            doc["_source"]["name_vec"] = vectors[i].astype(np.float32).tolist()
        yield doc


def index_split(client, split, index, args):
    if client.indices.exists(index=index):
        if not args.recreate:
            sys.exit(f"index {index} already exists; pass --recreate to rebuild it")
        client.indices.delete(index=index)
    client.indices.create(index=index, body=INDEX_BODY)

    vectors, vector_ids = load_vectors(split, args.sources)
    print(f"{index}: name vectors {'from ' + str(EMBEDDINGS) if vectors is not None else 'not indexed'}", flush=True)
    total, failed, start = 0, 0, time.time()
    for path in source_files(split, args.sources):
        reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                             chunksize=args.chunk_rows, nrows=args.limit)
        for chunk in reader:
            chunk_vectors = None
            if vectors is not None:
                rows = slice(total, total + len(chunk))
                if not np.array_equal(vector_ids[rows], chunk.entity_id.to_numpy(dtype=str)):
                    sys.exit(f"embeddings for rows {rows} do not match {path.name}; re-run embed_names.py")
                chunk_vectors = vectors[rows]
            docs = build_docs(chunk, index, args.n_jobs, chunk_vectors)
            for ok, info in helpers.parallel_bulk(client, docs, thread_count=args.bulk_threads,
                                                  chunk_size=args.bulk_size, raise_on_error=False,
                                                  request_timeout=300):
                total += 1
                if not ok:
                    failed += 1
                    if failed <= 5:
                        print("  failed:", info, flush=True)
            rate = total / (time.time() - start)
            print(f"  {path.name}: {total:,} docs indexed ({rate:,.0f}/s, {failed} failed)", flush=True)

    client.indices.put_settings(index=index, body={"index": {"refresh_interval": "1s"}})
    client.indices.refresh(index=index)
    if args.forcemerge:
        print("  force-merging segments ...", flush=True)
        client.indices.forcemerge(index=index, max_num_segments=1, request_timeout=3600)
    count = client.count(index=index)["count"]
    print(f"{index}: {count:,} docs, {failed} failed, {time.time() - start:,.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", nargs="+", choices=["train", "test"], default=["train"])
    ap.add_argument("--sources", nargs="+", type=int, default=[1], help="source files to index (default: 1)")
    ap.add_argument("--index-prefix", default="er", help="index name is <prefix>-<split>")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=int(os.environ.get("OS_PORT", 9277)))
    ap.add_argument("--recreate", action="store_true", help="delete and rebuild existing indices")
    ap.add_argument("--limit", type=int, default=None, help="index only the first N rows of each file")
    ap.add_argument("--chunk-rows", type=int, default=500_000, help="rows standardized per batch")
    ap.add_argument("--n-jobs", type=int, default=len(os.sched_getaffinity(0)),
                    help="processes for standardization (default: CPUs available to this job)")
    ap.add_argument("--bulk-threads", type=int, default=6)
    ap.add_argument("--bulk-size", type=int, default=2_000, help="docs per bulk request")
    ap.add_argument("--forcemerge", action="store_true", help="merge to 1 segment per shard (faster queries)")
    args = ap.parse_args()

    client = OpenSearch(hosts=[{"host": args.host, "port": args.port}], timeout=120)
    print(f"connected to OpenSearch {client.info()['version']['number']} at {args.host}:{args.port}", flush=True)
    for split in args.split:
        index_split(client, split, f"{args.index_prefix}-{split}", args)


if __name__ == "__main__":
    main()
