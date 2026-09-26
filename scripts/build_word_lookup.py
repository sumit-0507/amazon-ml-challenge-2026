"""Mine the word variants that standardization leaves unmerged, from training matches.

Standardization unifies only the spellings in its tables (legal forms, street types,
cities, ...) and the learned native-script word map; any other variant passes through
unchanged, so "mfg" and "manufacturing" stay different tokens. Here every matched pair
of the train split (S1 record, S2/S3 record; the eval split is left out) is
standardized, the two token lists are aligned (difflib: equal blocks anchor, replaced
blocks of the same size pair up word by word), and every aligned pair of different
words is counted, for names and addresses separately. Tokens with a digit are skipped.

For a variant v (S2/S3 side) aligned with a word c (S1 side):
    pairs      times v was aligned with c
    same       times v was aligned with itself
    precision  pairs / all alignments of v   (how safe rewriting v -> c is)
    s1_count_* occurrences in standardized S1 texts (a frequent S1 variant is suspect)

Outputs in artifacts/word_lookup/:
    name_variants.tsv, address_variants.tsv   every pair seen >= --min-report times
    name_lookup.json, address_lookup.json     v -> c with >= --min-count pairs and
                                              precision >= --min-precision
    test_unseen.tsv                           test tokens seen >= --min-unseen times
                                              and never in the training files
                                              (France, untranslated native words, ...)

    python scripts/build_word_lookup.py        # CPU only; ~15 min on 16 cores
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "scripts"))
import retrieve_candidates as rc  # noqa: E402  (logging)
from address_standardization import standardize_addresses  # noqa: E402
from name_standardization import standardize_names  # noqa: E402

DATA = ROOT / "dataset"
OUT = ROOT / "artifacts" / "word_lookup"
FIELDS = ["name", "address"]
log = rc.log


def read(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def standardize(df, n_jobs):
    """{field: standardized texts} for a source frame."""
    return {"name": standardize_names(df.business_name, n_jobs=n_jobs),
            "address": standardize_addresses(df.business_address, df.country, n_jobs=n_jobs)}


def words(text):
    return [t for t in text.replace(",", " ").split() if not any(ch.isdigit() for ch in t)]


def count_words(texts):
    counts = Counter()
    for t in texts:
        counts.update(words(t))
    return counts


def _align_chunk(pairs):
    """(S1 text, S2/S3 text) pairs -> Counter of (variant, canonical), Counter of variants aligned to themselves."""
    diff, same = Counter(), Counter()
    for s1, s23 in pairs:
        a, b = words(s1), words(s23)
        if a == b:
            same.update(b)
            continue
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if tag == "equal":
                same.update(b[j1:j2])
            elif tag == "replace" and i2 - i1 == j2 - j1:
                diff.update(zip(b[j1:j2], a[i1:i2]))
    return diff, same


def align(s1_texts, s23_texts, n_jobs, chunk=100_000):
    pairs = list(zip(s1_texts, s23_texts))
    diff, same = Counter(), Counter()
    with Pool(n_jobs) as pool:
        for d, s in pool.imap_unordered(_align_chunk, (pairs[i:i + chunk] for i in range(0, len(pairs), chunk))):
            diff.update(d)
            same.update(s)
    return diff, same


def variant_table(diff, same, s1_counts, min_report):
    total = Counter(same)
    for (v, _), n in diff.items():
        total[v] += n
    rows = [(v, c, n, same[v], n / total[v], s1_counts[v], s1_counts[c])
            for (v, c), n in diff.items() if n >= min_report]
    return pd.DataFrame(rows, columns=["variant", "canonical", "pairs", "same", "precision",
                                       "s1_count_variant", "s1_count_canonical"]
                        ).sort_values(["pairs", "variant"], ascending=[False, True]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-report", type=int, default=3, help="pairs needed to be listed in the TSV")
    ap.add_argument("--min-count", type=int, default=10, help="pairs needed to enter the JSON lookup")
    ap.add_argument("--min-precision", type=float, default=0.6, help="precision needed to enter the JSON lookup")
    ap.add_argument("--min-unseen", type=int, default=20, help="test count needed to be listed as unseen")
    ap.add_argument("--n-jobs", type=int, default=len(os.sched_getaffinity(0)))
    args = ap.parse_args()
    rc.setup_logging("word-lookup")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # training files: every record standardized once
    s1 = read(DATA / "train" / "train_source1.tsv")
    s1_std = standardize(s1, args.n_jobs)
    s23 = pd.concat([read(DATA / "train" / f"train_source{i}.tsv") for i in (2, 3)], ignore_index=True)
    s23_std = standardize(s23, args.n_jobs)
    log.info(f"train standardized: {len(s1):,} S1 + {len(s23):,} S2/S3 records ({time.time() - t0:.0f}s)")

    # matched pairs of the train split only
    split = read(DATA / "splits" / "s23_train.tsv")[["entity_id", "source1_entity_id"]]
    split = split[split.source1_entity_id != ""]
    s1_pos = pd.Series(range(len(s1)), index=s1.entity_id)
    s23_pos = pd.Series(range(len(s23)), index=s23.entity_id)
    i1 = s1_pos.loc[split.source1_entity_id].to_numpy()
    i23 = s23_pos.loc[split.entity_id].to_numpy()
    log.info(f"{len(split):,} matched train-split pairs")

    train_vocab = {}
    for field in FIELDS:
        t1 = time.time()
        s1_counts = count_words(s1_std[field])
        train_vocab[field] = set(s1_counts) | set(count_words(s23_std[field]))
        diff, same = align([s1_std[field][i] for i in i1], [s23_std[field][i] for i in i23], args.n_jobs)
        table = variant_table(diff, same, s1_counts, args.min_report)
        table.to_csv(OUT / f"{field}_variants.tsv", sep="\t", index=False, float_format="%.4f")
        keep = table[(table.pairs >= args.min_count) & (table.precision >= args.min_precision)]
        lookup = dict(keep.drop_duplicates("variant")[["variant", "canonical"]].itertuples(index=False))
        (OUT / f"{field}_lookup.json").write_text(json.dumps(lookup, ensure_ascii=False, indent=0, sort_keys=True),
                                                 encoding="utf-8")
        log.info(f"{field}: {sum(diff.values()):,} aligned differing words, {len(diff):,} distinct pairs; "
                 f"{len(table):,} pairs seen >= {args.min_report} times -> {field}_variants.tsv; "
                 f"{len(lookup):,} in the lookup (>= {args.min_count} pairs, precision >= {args.min_precision}) "
                 f"({time.time() - t1:.0f}s)")
        log.info(f"  top {field} variants: " + ", ".join(
            f"{r.variant}->{r.canonical} ({r.pairs:,}, p {r.precision:.2f})" for r in keep.head(25).itertuples()))
    del s1, s1_std, s23, s23_std, split

    # test tokens never seen in any training file
    test = pd.concat([read(DATA / "test" / f"test_source{i}.tsv") for i in (1, 2, 3)], ignore_index=True)
    test_std = standardize(test, args.n_jobs)
    rows = []
    for field in FIELDS:
        by_country = Counter()
        for text, country in zip(test_std[field], test.country):
            by_country.update((w, country) for w in words(text))
        totals, top = Counter(), {}
        for (w, country), n in by_country.items():
            if w not in train_vocab[field]:
                totals[w] += n
                if n > top.get(w, ("", 0))[1]:
                    top[w] = (country, n)
        rows += [(field, w, n, top[w][0], top[w][1] / n) for w, n in totals.items() if n >= args.min_unseen]
    unseen = pd.DataFrame(rows, columns=["field", "token", "test_count", "main_country", "main_country_share"]
                          ).sort_values(["field", "test_count"], ascending=[True, False])
    unseen.to_csv(OUT / "test_unseen.tsv", sep="\t", index=False, float_format="%.3f")
    log.info(f"test: {len(test):,} records; {len(unseen):,} tokens seen >= {args.min_unseen} times and never in "
             f"training -> test_unseen.tsv ({(unseen.field == 'name').sum():,} name, "
             f"{(unseen.field == 'address').sum():,} address)")
    log.info(f"done in {time.time() - t0:,.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
