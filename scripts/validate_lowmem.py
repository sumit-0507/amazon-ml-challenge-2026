"""The official submission checks with bounded memory (for a multi-GB candidate_pairs.tsv).

utils/validate_submission.py keeps every candidate list in memory (~400M IDs for our
test candidates: far beyond a 24 GB allocation). Every rule it applies is per row, so
here the candidate file is streamed: header, one row per test S1 entity, no duplicate
rows, no repeated ID within a list, S2-/S3- prefixes only, and every matched ID among
its S1 entity's candidates (the validator's warning). matching_results.tsv is still
checked by the official validator itself.

    python scripts/validate_lowmem.py --matching output/matching_results.tsv \\
        --candidate output/candidate_pairs.tsv --test-dir dataset/test
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# the official validator: $VALIDATOR, else the copy in the challenge resource folder
VALIDATOR = Path(os.environ.get("VALIDATOR") or ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "utils" / "validate_submission.py")
sys.path.insert(0, str(VALIDATOR.parent))
import validate_submission as V  # noqa: E402  (the official ID reader and header)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args()

    print("== official validator on matching_results.tsv", flush=True)
    official = subprocess.run([sys.executable, str(VALIDATOR), "--matching", args.matching,
                               "--candidate", "/nonexistent", "--test-dir", args.test_dir])

    print("== streamed checks on candidate_pairs.tsv", flush=True)
    required = V.read_ids(Path(args.test_dir) / "test_source1.tsv")
    matched = {}
    with open(args.matching, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            if rest:
                matched[s1] = rest.split(",")
    issues = {"malformed": [], "duplicate_row": [], "repeated_id": [], "s1_id": [], "bad_prefix": [],
              "match_not_candidate": []}
    seen, n_rows, empties = set(), 0, 0
    with open(args.candidate, encoding="utf-8") as f:
        header = [c.strip().lower() for c in f.readline().rstrip("\n").split("\t")]
        if header != V.CANDIDATE_HEADER:
            print(f"FAIL: unexpected header {header}, expected {V.CANDIDATE_HEADER}")
            sys.exit(1)
        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition("\t")
            if not tab:
                if s1.strip():
                    issues["malformed"].append(line_num)
                continue
            n_rows += 1
            if s1 in seen:
                issues["duplicate_row"].append(s1)
            seen.add(s1)
            ids = rest.rstrip("\n").split(",") if rest.strip() else []
            if not ids:
                empties += 1
            id_set = set(ids)
            if len(ids) != len(id_set):
                issues["repeated_id"].append(s1)
            for mid in id_set:
                if mid.startswith("S1-"):
                    issues["s1_id"].append(s1)
                    break
                if not mid.startswith(("S2-", "S3-")):
                    issues["bad_prefix"].append(s1)
                    break
            if s1 in matched and not set(matched[s1]) <= id_set:
                issues["match_not_candidate"].append(s1)
    issues["missing_s1"] = list(required - seen)
    issues["unknown_s1"] = list(seen - required)
    print(f"  candidate_pairs.tsv: {n_rows} rows ({empties} empty, {n_rows - empties} non-empty).")
    failed = False
    for kind, found in issues.items():
        if found:
            level = "WARNING" if kind == "match_not_candidate" else "ERROR"
            failed |= level == "ERROR"
            print(f"  {level} {kind}: {len(found)}, e.g. {found[:5]}")
    ok = official.returncode == 0 and not failed
    print("PASS — matching_results.tsv (official validator) and candidate_pairs.tsv (streamed checks)" if ok
          else "FAIL — see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
