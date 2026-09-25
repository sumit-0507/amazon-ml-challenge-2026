"""Business-name standardization for the entity-resolution challenge.

Maps the noisy ``business_name`` values from the three sources onto one
canonical, lowercase English/ASCII form:

    >>> standardize_name("Anb Trading Pvt. Ltd.")
    'anb trading pvt ltd'
    >>> standardize_name("Sri Anb Trading Private Limted")
    'anb trading pvt ltd'
    >>> standardize_name("Limited Gyan (India) Pharmaceuticals")
    'gyan india pharmaceuticals ltd'
    >>> standardize_name("Zetajax t/a Ps Wisdom Private Limited")
    'ps wisdom pvt ltd'
    >>> standardize_name(">> Keeton Trading L.L.C.")
    'keeton trading llc'
    >>> standardize_name("keetontrading.com")
    'keetontrading'
    >>> standardize_name("T0tal Realty 5tudios L.L.C.")
    'total realty studios llc'
    >>> standardize_name("Fractales Amis Groupe S.A.S")
    'fractales amis groupe sas'
    >>> standardize_name("Rolston, Swafford & Mangat Incorp", drop_legal=True)
    'rolston swafford and mangat'

Steps:
  * repair mojibake; keep only the real name after an alias marker
    (``Zetajax t/a <name>``, ``d/b/a``, ``a/k/a``); strip websites
    (``www.x.com`` -> ``x``, ``NAME | www.x.com`` -> ``NAME``)
  * native-script words -> English (``मॉडर्न फाइनेंस`` -> ``modern finance``) using
    the word map learned from training matches, with ``anyascii`` as fallback;
    anyascii also flattens accents and any other script to ASCII
  * lowercase, ``&``/``et`` -> ``and``, join dotted acronyms (``L.L.C.`` -> ``llc``),
    punctuation -> space, look-alike digits in words -> letters (``T0tal`` -> ``total``)
  * drop the honorifics the noisy sources prepend (``Mr``, ``Smt``, ``M/s``,
    ``Dr``, ``Sri``, ``Shri``); ``Shree``/``Sree`` are real name words and kept
  * legal forms, including typos, -> one token each (``Limited``/``Lmtd``/``Limted``/
    ``1imited`` -> ``ltd``, ``Incorp``/``Incorporated`` -> ``inc``, ``S.A.R.L.`` ->
    ``sarl``, ...) and moved to the end; ``drop_legal=True`` removes them

Native-script word map: build it once with
    python utils/name_standardization.py --build-word-map
which writes ``artifacts/native_word_map.json``. Without it, native-script
words fall back to anyascii (phonetic, much less accurate).
"""

import json
import multiprocessing
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from anyascii import anyascii
from rapidfuzz.distance import OSA

from address_standardization import _ACRONYM, _APOSTROPHES, _PUNCT_TABLE, _fix_mojibake

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
LEARNED_WORD_MAP = ARTIFACTS_DIR / "native_word_map.json"
LLM_WORD_MAP = ARTIFACTS_DIR / "llm_word_map.json"  # optional, from an LLM pass

# --------------------------------------------------------------------------
# Lookup tables
# --------------------------------------------------------------------------

# Every spelling -> canonical legal-form token.
LEGAL_ABBR = {
    # India / UK / generic
    "private": "pvt", "pvt": "pvt", "pvte": "pvt", "pte": "pvt",
    "limited": "ltd", "ltd": "ltd", "lmtd": "ltd", "limtd": "ltd",
    "public": "public",
    # US
    "incorporated": "inc", "incorporation": "inc", "incorp": "inc", "inc": "inc",
    "corporation": "corp", "corp": "corp", "corpn": "corp",
    "company": "co", "co": "co", "cie": "co", "compagnie": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc", "pc": "pc", "plc": "plc",
    "opc": "opc",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "ei": "ei",
    # other common forms (open set of countries)
    "gmbh": "gmbh", "srl": "srl", "pty": "pty", "bhd": "bhd",
}

# Multi-word forms, applied after LEGAL_ABBR (so on canonical tokens).
_LEGAL_PHRASES = [
    (re.compile(r"\bltd liability (?:co|partnership)\b"),
     lambda m: "llc" if m.group(0).endswith("co") else "llp"),
    (re.compile(r"\bprofessional corp\b"), "pc"),
    (re.compile(r"\bpublic ltd(?: co)?\b"), "plc"),
    (re.compile(r"\bp ltd\b"), "pvt ltd"),
    (re.compile(r"\bsociete a responsabilite ltd(?:ee)?\b"), "sarl"),
    (re.compile(r"\bsociete par actions simplifiee\b"), "sas"),
    (re.compile(r"\bsociete civile immobiliere\b"), "sci"),
    (re.compile(r"\bsociete anonyme\b"), "sa"),
]

LEGAL_FORMS = set(LEGAL_ABBR.values()) - {"public"} | {"plc"}

# Long legal words whose typos are corrected by edit distance (OSA <= 1:
# limted, limtied, 1imited, c0rporation, lncorporated, priviate, comapny ...).
# "corporate" / "cooperation" are 2+ edits away and are left alone.
_FUZZY_LEGAL = {"limited": "ltd", "private": "pvt", "company": "co",
                "corporation": "corp", "incorporated": "inc"}

# Prefixes injected as noise by sources 2/3 (each on ~1.3% of their India
# names, almost never in source 1). "m s" is "M/s" after punctuation removal.
HONORIFICS = {"mr", "smt", "dr", "sri", "shri"}

# Spelling variants of the same word.
NAME_VARIANTS = {"sree": "shree", "et": "and", "etablissements": "ets", "etablissement": "ets"}

# Sources 2/3 swap letters for look-alike digits (t0tal, 5cholarship, 1imited).
# Measured on training matches this is always 0->o 1->l 5->s 8->b 6->g, and no
# genuine source-1 word mixes 3+ letters with a digit.
_LEET = str.maketrans("01568", "olsbg")
_LEET_TOKEN = re.compile(r"(?=(?:[^a-z]*[a-z]){3})[a-z01568]*[01568][a-z01568]*")

_ALIAS = re.compile(r"\s(?:d/b/a|a/k/a|t/a|doing business as|also known as|trading as)\s",
                    re.IGNORECASE)
_TLD = r"(?:com|net|org|in|co|io|biz|info|fr|us|ai|app|shop|store|online|site|tech)"
_DOMAIN = re.compile(rf"^(?:https?://)?(?:www\.)?([a-z0-9-]+)\.{_TLD}(?:\.[a-z]{{2}})?(?:/\S*)?$",
                     re.IGNORECASE)
_NAME_TOKEN = re.compile(r"[^\s,.()&/\-]+")
_NATIVE_TOKEN = re.compile(r"[^\s,.()&/\-]*[ऀ-෿][^\s,.()&/\-]*")

# --------------------------------------------------------------------------
# Native script -> English
# --------------------------------------------------------------------------

_word_map = None


def name_tokens(s):
    return _NAME_TOKEN.findall(unicodedata.normalize("NFC", s))


def learn_native_word_map(english_names, native_names, min_count=2):
    """Learn native-script word -> English word from matched name pairs.

    The noisy sources transliterate word by word, so pairs with equal token
    counts are aligned position by position; each native word takes its
    majority English word (seen at least ``min_count`` times).
    """
    votes = defaultdict(Counter)
    for eng, nat in zip(english_names, native_names):
        if not isinstance(eng, str) or not isinstance(nat, str):
            continue
        et, nt = name_tokens(eng.casefold()), name_tokens(nat)
        if len(et) != len(nt):
            continue
        for n, e in zip(nt, et):
            if _NATIVE_TOKEN.fullmatch(n) and e.isascii():
                votes[n][e] += 1
    word_map = {}
    for n, c in votes.items():
        e, count = c.most_common(1)[0]
        if count >= min_count:
            word_map[n] = e
    return word_map


def load_word_map(paths=(LLM_WORD_MAP, LEARNED_WORD_MAP)):
    """Load and merge JSON word maps (later paths win); missing files are skipped."""
    global _word_map
    merged = {}
    for path in map(Path, paths):
        if path.exists():
            merged.update(json.loads(path.read_text(encoding="utf-8")))
    _word_map = merged
    _standardize_name.cache_clear()
    return merged


def _get_word_map():
    if _word_map is None:
        load_word_map()
    return _word_map


def to_english_script(text):
    """Replace native-script words with their English spelling; return ASCII text."""
    word_map = _get_word_map()
    text = _NATIVE_TOKEN.sub(lambda m: word_map.get(m.group(0), m.group(0)),
                             unicodedata.normalize("NFC", text))
    return anyascii(text)

# --------------------------------------------------------------------------
# Standardization
# --------------------------------------------------------------------------


def _strip_alias(s):
    """'Zetajax t/a Real Name LLC' -> 'Real Name LLC' (sources 2/3 prepend a fake alias)."""
    parts = _ALIAS.split(s)
    return parts[-1] if len(parts) > 1 and parts[-1].strip() else s


def _strip_website(s):
    s = s.split("|")[0] if "|" in s and s.split("|")[0].strip() else s
    return " ".join(m.group(1) if (m := _DOMAIN.match(t)) else t for t in s.split())


@lru_cache(maxsize=1 << 16)
def _legal_token(t):
    if t in LEGAL_ABBR:
        return LEGAL_ABBR[t]
    if len(t) >= 6:
        for word, canon in _FUZZY_LEGAL.items():
            if abs(len(t) - len(word)) <= 1 and OSA.distance(t, word) <= 1:
                return canon
    return t


def _drop_honorifics(tokens):
    i = 0
    while i < len(tokens):
        if tokens[i] in HONORIFICS:
            i += 1
        elif tokens[i] == "m" and i + 1 < len(tokens) and tokens[i + 1] == "s":
            i += 2
        else:
            break
    return tokens[i:] or tokens


@lru_cache(maxsize=1 << 20)
def _standardize_name(name, drop_legal):
    s = _fix_mojibake(unicodedata.normalize("NFKC", name))
    s = _strip_website(_strip_alias(s))
    s = to_english_script(s).casefold()
    s = _APOSTROPHES.sub("", s).replace("&", " and ")
    s = _ACRONYM.sub(lambda m: m.group(0).replace(".", ""), s)
    s = s.translate(_PUNCT_TABLE).replace("-", " ").replace("/", " ")

    tokens = [t.translate(_LEET) if _LEET_TOKEN.fullmatch(t) else t for t in s.split()]
    tokens = _drop_honorifics(tokens)
    s = " ".join(_legal_token(NAME_VARIANTS.get(t, t)) for t in tokens)
    for pattern, repl in _LEGAL_PHRASES:
        s = pattern.sub(repl, s)

    tokens = s.split()
    body = [t for t in tokens if t not in LEGAL_FORMS]
    if drop_legal:
        return " ".join(body or tokens)
    legal = list(dict.fromkeys(t for t in tokens if t in LEGAL_FORMS))
    return " ".join(body + legal)


def standardize_name(name, drop_legal=False):
    """Return the canonical English/ASCII form of a business name ("" if missing).

    Args:
        name: raw ``business_name``; None/NaN are treated as empty.
        drop_legal: remove legal forms (pvt, ltd, inc, llc, sarl, ...) instead
            of moving them to the end. Useful as a second, "core name" feature:
            the sources often drop or add them ("Adhavan It Private" vs
            "Adhavan It Private Limited"). A name made only of legal forms is
            returned as is.
    """
    if name is None or name != name:  # None / NaN
        return ""
    return _standardize_name(str(name), drop_legal)


def standardize_names(names, drop_legal=False, n_jobs=1, chunksize=20_000):
    """Standardize a column of names; returns a list aligned with the input.

        df["name_std"] = standardize_names(df.business_name, n_jobs=16)

    ``n_jobs > 1`` spreads the work over processes (-1 = all cores).
    """
    names = list(names)
    _get_word_map()  # load once before forking
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    if n_jobs <= 1 or len(names) < 2 * chunksize:
        return [standardize_name(n, drop_legal) for n in names]
    with multiprocessing.Pool(n_jobs) as pool:
        return pool.starmap(standardize_name, [(n, drop_legal) for n in names], chunksize=chunksize)

# --------------------------------------------------------------------------
# Building the native-script word map from the training data
# --------------------------------------------------------------------------


def build_word_map(dataset_dir=ROOT / "dataset", out_path=LEARNED_WORD_MAP, min_count=2):
    """Learn the native-script word map from training matches and save it as JSON."""
    import pandas as pd

    train = Path(dataset_dir) / "train"
    read = lambda f, cols: pd.read_csv(train / f, sep="\t", dtype=str, usecols=cols, keep_default_na=False)
    gt = read("train_ground_truth.tsv", [0, 1])
    gt = gt[gt.matched_entity_ids != ""]
    gt = gt.assign(match=gt.matched_entity_ids.str.split(",")).explode("match")
    s1 = read("train_source1.tsv", [0, 1]).set_index("entity_id").business_name
    others = pd.concat([read(f"train_source{i}.tsv", [0, 1]) for i in (2, 3)])
    others = others[others.business_name.str.contains(_NATIVE_TOKEN)].set_index("entity_id").business_name
    gt = gt[gt.match.isin(others.index)]

    word_map = learn_native_word_map(gt.source1_entity_id.map(s1), gt.match.map(others), min_count)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(word_map, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    print(f"learned {len(word_map)} native words from {len(gt)} matched pairs -> {out_path}")
    load_word_map()
    return word_map


if __name__ == "__main__":
    if "--build-word-map" in sys.argv:
        build_word_map()
    else:
        print(__doc__)
