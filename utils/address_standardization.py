"""Address standardization for the business entity-resolution challenge.

Maps the noisy ``business_address`` values from the three sources onto one
canonical form so that downstream similarity features compare like with like
(business names: see ``name_standardization.py``):

    >>> standardize_address("3315 FREMONT SAINT, PEORIA, IL", "US")
    '3315 fremont st, peoria, il'
    >>> standardize_address("Missouri, 630 45th Terrace, Kansas City", "US")
    '630 45 ter, kansas, mo'
    >>> standardize_address("KANSAS CITY, MO, 630 45ND TERRACE, null", "US")
    '630 45 ter, kansas, mo'
    >>> standardize_address("हरियाणा, DOOR NO 1038 SECTOR 9, FARIDABAD", "India")
    '1038 sector 9, faridabad, hr'
    >>> standardize_address("#1038 Sector 9, Faridabad, Haryana", "India")
    '1038 sector 9, faridabad, hr'
    >>> standardize_address("63 R. DE DIEPPE, LILLE, Nord", "France")
    '63 rue de dieppe, lille, hdf'

What it does, per comma-separated component:
  * repairs mojibake, drops placeholder components (null, <NULL>, N/A, ...)
  * Unicode NFKC, strips Latin accents (Indic scripts are left intact), lowercases
  * maps state/region names in any spelling or script to one code
    (``Tamil Nadu`` / ``TN`` / ``தமிழ்நாடு`` -> ``tn``; ``Nord`` -> ``hdf``)
  * canonicalizes street-type and other abbreviations (``Street``/``St``/``Saint``
    -> ``st``, ``R.`` -> ``rue``), ordinals (``10RD``/``10th``/``Tenth`` -> ``10``)
    and a few city aliases (``Bombay`` -> ``mumbai``)
  * drops house-number markers (``H.NO``, ``Door No``, ``#``) and leading zeros
  * moves numbered components first and the state last, then de-duplicates

Countries without a profile (the country field is an open set) still get all
the language-neutral steps; only the state tables and local abbreviations are
country specific.
"""

import multiprocessing
import os
import re
import unicodedata
from functools import lru_cache, partial

# --------------------------------------------------------------------------
# Lookup tables
# --------------------------------------------------------------------------

US_STATES = {
    "al": ["Alabama"], "ak": ["Alaska"], "az": ["Arizona"], "ar": ["Arkansas"],
    "ca": ["California"], "co": ["Colorado"], "ct": ["Connecticut"],
    "de": ["Delaware"], "fl": ["Florida"], "ga": ["Georgia"], "hi": ["Hawaii"],
    "id": ["Idaho"], "il": ["Illinois"], "in": ["Indiana"], "ia": ["Iowa"],
    "ks": ["Kansas"], "ky": ["Kentucky"], "la": ["Louisiana"], "me": ["Maine"],
    "md": ["Maryland"], "ma": ["Massachusetts"], "mi": ["Michigan"],
    "mn": ["Minnesota"], "ms": ["Mississippi"], "mo": ["Missouri"],
    "mt": ["Montana"], "ne": ["Nebraska"], "nv": ["Nevada"],
    "nh": ["New Hampshire"], "nj": ["New Jersey"], "nm": ["New Mexico"],
    "ny": ["New York"], "nc": ["North Carolina"], "nd": ["North Dakota"],
    "oh": ["Ohio"], "ok": ["Oklahoma"], "or": ["Oregon"], "pa": ["Pennsylvania"],
    "ri": ["Rhode Island"], "sc": ["South Carolina"], "sd": ["South Dakota"],
    "tn": ["Tennessee"], "tx": ["Texas"], "ut": ["Utah"], "vt": ["Vermont"],
    "va": ["Virginia"], "wa": ["Washington"], "wv": ["West Virginia"],
    "wi": ["Wisconsin"], "wy": ["Wyoming"],
    "dc": ["District of Columbia", "Washington DC", "Washington D.C."],
    "pr": ["Puerto Rico"],
}

# English name(s), then the name in the state's own language(s), then Hindi.
INDIA_STATES = {
    "ap": ["Andhra Pradesh", "ఆంధ్రప్రదేశ్", "ఆంధ్ర ప్రదేశ్", "आंध्र प्रदेश"],
    "ar": ["Arunachal Pradesh", "अरुणाचल प्रदेश"],
    "as": ["Assam", "অসম", "असम"],
    "br": ["Bihar", "बिहार"],
    "cg": ["Chhattisgarh", "Chattisgarh", "छत्तीसगढ़"],
    "ga": ["Goa", "गोंय", "गोवा"],
    "gj": ["Gujarat", "ગુજરાત", "गुजरात"],
    "hr": ["Haryana", "हरियाणा"],
    "hp": ["Himachal Pradesh", "हिमाचल प्रदेश"],
    "jh": ["Jharkhand", "झारखंड", "झारखण्ड"],
    "ka": ["Karnataka", "ಕರ್ನಾಟಕ", "कर्नाटक"],
    "kl": ["Kerala", "Keralam", "കേരളം", "केरल"],
    "mp": ["Madhya Pradesh", "मध्य प्रदेश"],
    "mh": ["Maharashtra", "महाराष्ट्र"],
    "mn": ["Manipur", "মণিপুর", "ꯃꯅꯤꯄꯨꯔ", "मणिपुर"],
    "ml": ["Meghalaya", "मेघालय"],
    "mz": ["Mizoram", "मिज़ोरम", "मिजोरम"],
    "nl": ["Nagaland", "नागालैंड"],
    "od": ["Odisha", "Orissa", "ଓଡ଼ିଶା", "ओडिशा", "उड़ीसा"],
    "pb": ["Punjab", "ਪੰਜਾਬ", "पंजाब"],
    "rj": ["Rajasthan", "राजस्थान"],
    "sk": ["Sikkim", "सिक्किम"],
    "tn": ["Tamil Nadu", "தமிழ்நாடு", "तमिलनाडु", "तमिल नाडु"],
    "tg": ["Telangana", "తెలంగాణ", "तेलंगाना"],
    "tr": ["Tripura", "ত্রিপুরা", "त्रिपुरा"],
    "up": ["Uttar Pradesh", "उत्तर प्रदेश", "اتر پردیش"],
    "uk": ["Uttarakhand", "Uttaranchal", "उत्तराखंड", "उत्तराखण्ड"],
    "wb": ["West Bengal", "পশ্চিমবঙ্গ", "पश्चिम बंगाल"],
    "an": ["Andaman and Nicobar Islands", "Andaman and Nicobar",
           "अंडमान और निकोबार द्वीपसमूह", "अंडमान और निकोबार"],
    "ch": ["Chandigarh", "ਚੰਡੀਗੜ੍ਹ", "चंडीगढ़"],
    "dn": ["Dadra and Nagar Haveli and Daman and Diu", "Daman and Diu",
           "Dadra and Nagar Haveli", "દમણ અને દીવ", "दादरा और नगर हवेली और दमन और दीव",
           "दमन और दीव", "दादरा और नगर हवेली"],
    "dl": ["Delhi", "NCT of Delhi", "दिल्ली", "ਦਿੱਲੀ", "دہلی"],
    "jk": ["Jammu and Kashmir", "جموں و کشمیر", "जम्मू और कश्मीर", "जम्मू-कश्मीर"],
    "la": ["Ladakh", "लद्दाख", "ལ་དྭགས"],
    "ld": ["Lakshadweep", "ലക്ഷദ്വീപ്", "लक्षद्वीप"],
    "py": ["Puducherry", "Pondicherry", "புதுச்சேரி", "पुडुचेरी"],
}

# Regions (ISO 3166-2:FR). Departments are folded into their region because
# sources disagree on which of the two they record.
FRANCE_REGIONS = {
    "hdf": ["Hauts-de-France", "Nord", "Pas-de-Calais", "Aisne", "Oise", "Somme"],
    "naq": ["Nouvelle-Aquitaine", "Gironde", "Charente", "Charente-Maritime",
            "Correze", "Creuse", "Dordogne", "Landes", "Lot-et-Garonne",
            "Pyrenees-Atlantiques", "Deux-Sevres", "Vienne", "Haute-Vienne"],
    "pdl": ["Pays de la Loire", "Loire-Atlantique", "Maine-et-Loire", "Mayenne",
            "Sarthe", "Vendee"],
    "idf": ["Ile-de-France"],
    "ara": ["Auvergne-Rhone-Alpes"],
    "bfc": ["Bourgogne-Franche-Comte"],
    "bre": ["Bretagne", "Brittany"],
    "cvl": ["Centre-Val de Loire"],
    "cor": ["Corse", "Corsica"],
    "ges": ["Grand Est"],
    "nor": ["Normandie", "Normandy"],
    "occ": ["Occitanie"],
    "pac": ["Provence-Alpes-Cote d'Azur", "PACA"],
}

# Applied to every country. Several spellings collapse onto one token; that is
# deliberate (e.g. the noisy sources expand "St" to "Saint" for streets).
COMMON_ABBR = {
    "street": "st", "str": "st", "saint": "st",
    "road": "rd",
    "avenue": "ave", "av": "ave", "avn": "ave", "aven": "ave", "avenu": "ave",
    "drive": "dr", "drv": "dr", "doctor": "dr",
    "lane": "ln",
    "court": "ct", "crt": "ct",
    "boulevard": "blvd", "boul": "blvd", "bd": "blvd", "blv": "blvd",
    "place": "pl",
    "circle": "cir", "circ": "cir",
    "terrace": "ter", "terr": "ter",
    "highway": "hwy",
    "parkway": "pkwy", "pky": "pkwy",
    "expressway": "expy",
    "square": "sq",
    "trail": "trl",
    "mount": "mt", "mountain": "mtn",
    "fort": "ft",
    "township": "twp",
    "route": "rte", "rt": "rte",
    "apartment": "apt", "apartments": "apt", "apts": "apt", "appt": "apt",
    "building": "bldg", "bldng": "bldg",
    "floor": "fl", "flr": "fl",
    "suite": "ste",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "nr": "near",
    "opposite": "opp",
}

ORDINAL_WORDS = {
    word: str(i) for i, word in enumerate(
        ["first", "second", "third", "fourth", "fifth", "sixth", "seventh",
         "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
         "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
         "nineteenth", "twentieth"], start=1)
}

INDIA_ABBR = {
    "ngr": "nagar",
    "col": "colony",
    "society": "soc",
    "sec": "sector",
    "district": "dist", "distt": "dist",
    "taluk": "taluka", "tal": "taluka", "tq": "taluka",
    "village": "vill", "vil": "vill",
    "extension": "extn", "ext": "extn",
    "estate": "est",
    # city aliases
    "bombay": "mumbai", "calcutta": "kolkata", "bengaluru": "bangalore",
    "madras": "chennai", "gurugram": "gurgaon", "poona": "pune",
    "hyd": "hyderabad", "ahmadabad": "ahmedabad", "baroda": "vadodara",
    "trivandrum": "thiruvananthapuram", "cochin": "kochi", "mysuru": "mysore",
    "mangaluru": "mangalore", "belagavi": "belgaum", "vizag": "visakhapatnam",
    "pondicherry": "puducherry",
}

FRANCE_ABBR = {
    "r": "rue",
    "all": "allee",
    "impasse": "imp",
    "chemin": "ch", "chem": "ch",
    "cours": "crs",
    "quai": "quai", "q": "quai",
    "residence": "res", "resid": "res",
    "faubourg": "fbg", "fg": "fbg",
    "sainte": "ste",
    "etage": "fl",
    "appartement": "apt",
    "batiment": "bldg", "bat": "bldg",
    "general": "gen", "gal": "gen",
    "marechal": "mal",
    "docteur": "dr",
    "president": "pdt",
}

NULL_TOKENS = {
    "", "null", "<null>", "none", "n/a", "na", "nan", "<na>", "nil", "-", "--",
    "unknown", "not available",
}

# "h no 12", "door no 12", "no 12", "#12" -> "12"
_HOUSE_MARKERS = {"h", "house", "hse", "door", "d"}
_NUMBER_WORDS = {"no", "nos", "number", "num"}
_STANDALONE_MARKERS = {"hn", "hno", "dno"}

# Trailing words the noisy sources append to city names ("CHARLOTTE CITY").
_CITY_DESIGNATORS = {"city", "twp", "cdp"}

# --------------------------------------------------------------------------
# Regexes and character tables
# --------------------------------------------------------------------------

_MOJIBAKE_HINT = re.compile(r"[ÂÃâÏ\x80-\x9f�]")
_MOJIBAKE_QUOTE = re.compile(r"[Ââ]\x80[\x98\x99]")      # mangled ‘ ’
_MOJIBAKE = re.compile(r"[ÂÃâ][\x80-\x9f]+|Ï¿½|�|[\x80-\x9f]")
_SPLIT = re.compile(r"[,;|\n\t]")
_KEY_SEPARATORS = re.compile(r"[\s.\-_'’\"“”()]+")
_APOSTROPHES = re.compile(r"[’‘'`´ʼ]")
_ACRONYM = re.compile(r"\b(?:[a-z]\.){2,}")               # c.i.t. -> cit
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")         # 45nd -> 45
_FR_ORDINAL = re.compile(r"\b(\d+)(?:er|re|ere|eme|e)\b")  # 2eme -> 2
_NUMBER_WORD_HYPHEN = re.compile(r"(?<=\bno)-(?=\d)")      # no-2 -> no 2
_LETTER_DIGIT_HYPHEN = re.compile(r"(?<=[^\W\d_])-(?=\d)|(?<=\d)-(?=[^\W\d_])")
_LOOSE_HYPHEN_SLASH = re.compile(r"(?<!\d)[-/]|[-/](?!\d)")
_LEADING_ZEROS = re.compile(r"(?<!\d)0+(?=\d)")
_NUMBER_RANGE = re.compile(r"^(\d+)-(\d+)(?=\s|$)")        # 1056-1060 belden
_DIGIT = re.compile(r"\d")

# Punctuation, symbols and control characters -> space; "-" and "/" are kept
# here and resolved by the hyphen rules. Combining marks are untouched, so
# Devanagari, Tamil, etc. words stay whole.
_PUNCT_TABLE = {
    cp: " " for cp in range(0x10000)
    if unicodedata.category(chr(cp))[0] in "PS" or unicodedata.category(chr(cp)) == "Cc"
}
del _PUNCT_TABLE[ord("-")], _PUNCT_TABLE[ord("/")]


def _strip_accents(s):
    """Drop diacritics from Latin letters only; marks in other scripts (Indic
    vowel signs, Arabic, Tibetan, ...) are part of the word and are kept."""
    if s.isascii():
        return s
    out, after_latin = [], False
    for c in unicodedata.normalize("NFKD", s):
        if unicodedata.combining(c):
            if after_latin:
                continue
        else:
            after_latin = c < "ɐ"  # Basic Latin .. Latin Extended-B
        out.append(c)
    return unicodedata.normalize("NFC", "".join(out))


def _prep(s):
    return _strip_accents(unicodedata.normalize("NFKC", s).strip()).casefold()


def _key(s):
    return _KEY_SEPARATORS.sub(" ", s.replace("&", " and ")).strip()


def _state_lookup(table):
    lookup = {}
    for code, names in table.items():
        for name in [code, *names]:
            lookup[_key(_prep(name))] = code
    return lookup


def _profile(states=None, abbr=None, ordinal=None, collapse_ranges=False,
             prefer_state_code=False, priority_states=()):
    return {
        "states": _state_lookup(states or {}),
        "abbr": {**COMMON_ABBR, **ORDINAL_WORDS, **(abbr or {})},
        "ordinal": ordinal,
        "collapse_ranges": collapse_ranges,
        "prefer_state_code": prefer_state_code,
        "priority_states": set(priority_states),
    }


PROFILES = {
    # US cities share state names (Washington, Delaware): trust the code.
    # India/France: stray 2-letter tokens (Ar, Sk, An) collide with codes: trust the name.
    "us": _profile(US_STATES, collapse_ranges=True, prefer_state_code=True, priority_states=["dc"]),
    "india": _profile(INDIA_STATES, INDIA_ABBR),
    "france": _profile(FRANCE_REGIONS, FRANCE_ABBR, ordinal=_FR_ORDINAL),
    "generic": _profile(),
}

# States that sources use interchangeably for the same place: Telangana was
# part of Andhra Pradesh until 2014, and older records still say AP.
STATE_GROUPS = {
    "india": [{"ap", "tg"}],
}


def equivalent_states(state, country):
    """``state`` plus the states that records of the same place may carry instead."""
    if not state:
        return []
    for group in STATE_GROUPS.get(_profile_name(country), []):
        if state in group:
            return sorted(group)
    return [state]


COUNTRY_ALIASES = {
    "us": "us", "usa": "us", "united states": "us", "united states of america": "us","america": "us","u.s.": "us", "u.s.a.": "us","amrica": "us", "u.s.a": "us", "u.s": "us",
    "india": "india", "in": "india", "ind": "india","bharat": "india", "bharath": "india", "bhārat": "india","hindustan": "india", "hindustan": "india", "republic of india": "india",
    "france": "france", "fr": "france", "fra": "france","republique francaise": "france", "republique française": "france",
}

# --------------------------------------------------------------------------
# Standardization
# --------------------------------------------------------------------------


def _fix_mojibake(s):
    if not _MOJIBAKE_HINT.search(s):
        return s
    for encoding in ("cp1252", "latin-1"):
        try:
            s = s.encode(encoding).decode("utf-8")
            break
        except UnicodeError:
            pass
    return _MOJIBAKE.sub(" ", _MOJIBAKE_QUOTE.sub("'", s))


def _drop_number_markers(tokens):
    out = []
    for t in tokens:
        if t in _NUMBER_WORDS:
            if out and out[-1] in _HOUSE_MARKERS:
                out.pop()
            continue
        if t in _STANDALONE_MARKERS:
            continue
        out.append(t)
    return out


def _collapse_range(m):
    lo, hi = m.group(1), m.group(2)
    return lo if len(lo) == len(hi) and int(hi) > int(lo) else m.group(0)


def _clean_component(s, profile):
    """Normalize one already-prepped (NFKC, accent-stripped, casefolded) component."""
    s = _APOSTROPHES.sub("", s)
    s = s.replace("&", " and ")
    s = _ACRONYM.sub(lambda m: m.group(0).replace(".", ""), s)
    s = _ORDINAL.sub(r"\1", s)
    if profile["ordinal"]:
        s = profile["ordinal"].sub(r"\1", s)
    s = s.translate(_PUNCT_TABLE)
    s = _NUMBER_WORD_HYPHEN.sub(" ", s)
    s = _LETTER_DIGIT_HYPHEN.sub("", s)
    s = _LOOSE_HYPHEN_SLASH.sub(" ", s)

    abbr = profile["abbr"]
    tokens = _drop_number_markers([abbr.get(t, t) for t in s.split()])
    s = " ".join(tokens)
    if len(tokens) > 1 and tokens[-1] in _CITY_DESIGNATORS and not _DIGIT.search(s):
        s = s.rsplit(" ", 1)[0]

    s = _LEADING_ZEROS.sub("", s)
    if profile["collapse_ranges"]:
        s = _NUMBER_RANGE.sub(_collapse_range, s)
    return s


@lru_cache(maxsize=1 << 20)
def _standardize(address, profile_name, reorder):
    profile = PROFILES[profile_name]
    states = profile["states"]
    body, found_states = [], []

    for comp in _SPLIT.split(_fix_mojibake(address)):
        comp = _prep(comp)
        if comp in NULL_TOKENS:
            continue
        key = _key(comp)
        code = states.get(key)
        if code is not None:
            found_states.append((code, key == code))
            continue
        comp = _clean_component(comp, profile)
        if comp not in NULL_TOKENS:
            body.append(comp)

    if reorder:
        body = [c for c in body if _DIGIT.search(c)] + [c for c in body if not _DIGIT.search(c)]
    codes = [code for code, _ in found_states]
    return ", ".join(dict.fromkeys(body + codes)), _pick_state(found_states, profile)


def _pick_state(found_states, profile):
    """The record's state when several components look like one.

    US: a city can share a state's name ("Washington, 2 I St, DC", "Delaware, OH"),
    so a component written as a code wins, and DC beats Washington. India/France:
    short tokens collide with state codes ("Sk", "Ar"), so a full name wins.
    Otherwise the last one. Chosen on the train/eval pairs: the state filter then
    loses 0.018% of true matches (0.17% when taking the first state found).
    """
    if not found_states:
        return ""
    codes = [code for code, _ in found_states]
    for code in codes:
        if code in profile["priority_states"]:
            return code
    preferred = [code for code, is_code in found_states if is_code == profile["prefer_state_code"]]
    return (preferred or codes)[-1]


def _profile_name(country):
    if country is None or country != country:  # None / NaN
        return "generic"
    name = str(country).strip().casefold()
    name = COUNTRY_ALIASES.get(name, name)
    return name if name in PROFILES else "generic"


def standardize_address(address, country=None, reorder=True):
    """Return the canonical form of ``address`` ("" for missing/placeholder values).

    Args:
        address: raw ``business_address`` value; None/NaN are treated as empty.
        country: the record's ``country`` label. Selects the state table and
            local abbreviations; unknown countries get the generic rules only.
        reorder: put numbered components (house number/street) first and the
            state last, so reordered sources line up. Relative order of the
            remaining components is kept.
    """
    return standardize_address_parts(address, country, reorder)[0]


def standardize_address_parts(address, country=None, reorder=True):
    """Return ``(standardized_address, state_code)``; state_code is "" if none was found.

        >>> standardize_address_parts("Missouri, 630 45th Terrace, Kansas City", "US")
        ('630 45 ter, kansas, mo', 'mo')
        >>> standardize_address_parts("Washington, 2 I Street, Unit 724, DC", "US")[1]
        'dc'
    """
    if address is None or address != address:  # None / NaN
        return "", ""
    return _standardize(str(address), _profile_name(country), reorder)


def standardize_addresses(addresses, countries, reorder=True, with_state=False, n_jobs=1, chunksize=20_000):
    """Standardize a column of addresses; returns a list aligned with the input.

        df["address_std"] = standardize_addresses(df.business_address, df.country, n_jobs=16)

    ``with_state=True`` returns ``(address, state_code)`` tuples instead.
    Addresses are ~94% unique, so the work is per-string Python regex either
    way; ``n_jobs > 1`` spreads it over processes (-1 = all cores).
    """
    pairs = list(zip(addresses, countries))
    fn = partial(standardize_address_parts if with_state else standardize_address, reorder=reorder)
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    if n_jobs <= 1 or len(pairs) < 2 * chunksize:
        return [fn(a, c) for a, c in pairs]
    with multiprocessing.Pool(n_jobs) as pool:
        return pool.starmap(fn, pairs, chunksize=chunksize)
