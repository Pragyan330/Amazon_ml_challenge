"""Text normalisation for business names and addresses.

Everything here is static, rule-based string processing: Unicode folding, a small
abbreviation dictionary and legal-suffix stripping. No external data source, service or
lookup of business identities is used anywhere, per the challenge fair-play rules.
"""

import re
import unicodedata

# --- script detection -------------------------------------------------------------
# S1 names are always Latin; S2/S3 names appear in ten Indic scripts. Detecting this lets
# the scorer fall back to address-only evidence instead of scoring a guaranteed zero.
_SCRIPTS = (
    ("Devanagari", 0x0900, 0x097F),
    ("Bengali", 0x0980, 0x09FF),
    ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF),
    ("Odia", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
    ("Arabic", 0x0600, 0x06FF),
)


def script_of(s):
    """Return the name of the first non-Latin script found, else 'Latin'."""
    for ch in s:
        o = ord(ch)
        if o < 0x0300:
            continue
        for name, lo, hi in _SCRIPTS:
            if lo <= o <= hi:
                return name
    return "Latin"


def is_latin(s):
    return script_of(s) == "Latin"


# --- core normalisation -----------------------------------------------------------
_NONALNUM = re.compile(r"[^0-9a-z]+")
_WS = re.compile(r"\s+")


def fold(s):
    """Lowercase, strip accents, reduce punctuation to single spaces.

    'Blue Skill Private Limited' with an accented i  -> 'blue skill private limited'
    '@keystonechesapeake'                            -> 'keystonechesapeake'
    """
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return _WS.sub(" ", _NONALNUM.sub(" ", s)).strip()


def squash(s):
    """Folded text with every separator removed, for concatenated variants.

    'Lumyn Oriental' and 'lumynoriental.com' both contain 'lumynoriental'.
    """
    return _NONALNUM.sub("", unicodedata.normalize("NFKD", s.lower()))


# --- name handling ----------------------------------------------------------------
# Legal forms and corporate filler. Stripping these exposes the distinctive core, but note
# that ~48% of S1 entities still collide on the core name, so it is never sufficient alone.
LEGAL = frozenset(
    """
inc llc ltd limited limitee private pvt corp corporation co company llp plc
incorporated group holdings holding enterprises enterprise ventures venture
sarl sas sa sci scp snc eurl sasu societe ste
the and of for dba ms
""".split()
)

# Generic business-category words that carry little identifying signal.
GENERIC = frozenset(
    """
services service solutions solution center centre centers traders trading
partners partner associates associate consultants consulting consultant
industries industrial international global national
""".split()
)


_TLD = re.compile(r"(com|net|org|in|co|io|biz|info|us|fr)$")


def name_squash(s):
    """Squashed name with web decoration removed.

    5.2% of S2/S3 names are domains and ~0.8% are @handles or #hashtags, all derived from
    the real name by deleting spaces. Dropping the trailing TLD makes
    'lumynoriental.com' and 'Lumyn Oriental' identical after squashing.
    """
    x = squash(s)
    stripped = _TLD.sub("", x)
    # Only accept the strip if something substantial remains.
    return stripped if len(stripped) >= 4 else x


def name_tokens(s):
    """All folded tokens of a name."""
    return fold(s).split()


def name_core(s):
    """Folded tokens with legal suffixes removed; falls back to the full token list."""
    t = [x for x in fold(s).split() if x not in LEGAL]
    return t or fold(s).split()


def name_distinctive(s):
    """Core tokens with generic category words also removed."""
    t = [x for x in name_core(s) if x not in GENERIC]
    return t or name_core(s)


# --- address handling -------------------------------------------------------------
# Static abbreviation expansions. Source 2 abbreviates ('RD', 'LN'), Source 1 spells out
# ('Road', 'Lane') and Source 3 mixes both, so canonicalising recovers real overlap.
STREET = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "ln": "lane", "dr": "drive",
    "ct": "court", "cir": "circle", "pl": "place", "pkwy": "parkway",
    "hwy": "highway", "sq": "square", "ter": "terrace", "trl": "trail",
    "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor",
    "bldg": "building", "rm": "room", "no": "number", "nr": "near",
    "opp": "opposite", "n": "north", "s": "south", "e": "east", "w": "west",
    # French forms, seen only in the test set
    "r": "rue", "bvd": "boulevard", "imp": "impasse", "all": "allee", "ch": "chemin",
}

# US state abbreviation -> full spelling. Source 2 ends addresses with 'TN', Source 3 with
# 'Tennessee'; without this they share no token at all.
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "newhampshire", "nj": "newjersey", "nm": "newmexico", "ny": "newyork",
    "nc": "northcarolina", "nd": "northdakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhodeisland", "sc": "southcarolina",
    "sd": "southdakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "westvirginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "districtofcolumbia",
}

# Indian state abbreviations used by Source 3 ('WB', 'HR', 'TN', 'KA', 'MH', 'DL').
IN_STATES = {
    "ap": "andhrapradesh", "ar": "arunachalpradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana",
    "hp": "himachalpradesh", "jh": "jharkhand", "ka": "karnataka", "kl": "kerala",
    "mp": "madhyapradesh", "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya",
    "mz": "mizoram", "nl": "nagaland", "od": "odisha", "or": "odisha",
    "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamilnadu",
    "tg": "telangana", "ts": "telangana", "tr": "tripura", "up": "uttarpradesh",
    "uk": "uttarakhand", "ua": "uttarakhand", "wb": "westbengal", "dl": "delhi",
    "jk": "jammuandkashmir", "ch": "chandigarh", "py": "puducherry",
}

# Filler that appears in one source's rendering of an address but not another's.
ADDR_STOP = frozenset(
    """
india usa france number door plot near opposite behind beside off
region district dist tq taluk tehsil po village city town township urban rural
na none nil unit
""".split()
)


def addr_tokens(s, country=None):
    """Folded, abbreviation-expanded, stop-word-filtered address tokens.

    State abbreviations collapse to their full spelling so 'TN' and 'Tennessee' become the
    same token. ``country`` picks the state table; an unknown country skips both, which is
    the safe behaviour for France and any other unseen label.
    """
    states = US_STATES if country == "US" else IN_STATES if country == "India" else {}
    out = []
    for t in fold(s).split():
        t = STREET.get(t, t)
        t = states.get(t, t)
        if t in ADDR_STOP:
            continue
        out.append(t)
    return out


_HAS_DIGIT = re.compile(r"\d")


def numeric_tokens(s):
    """Tokens containing a digit - house/plot/door numbers, the sharpest address signal.

    Capped at 8 characters to exclude long ID-like strings that are rarely shared.
    """
    return {t for t in fold(s).split() if _HAS_DIGIT.search(t) and len(t) <= 8}
