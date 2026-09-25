"""Set-similarity features - the Jaccard family.

Each measure fails differently on this data, which is why several are kept:

* **Plain Jaccard** treats every token alike, so sharing 'road' counts as much as sharing
  'circuler'. Cheap, and good enough to gate on.
* **IDF-weighted (generalised) Jaccard** fixes exactly that. For binary token presence with
  IDF weights it reduces to ``sum(idf over intersection) / sum(idf over union)``, so rare
  agreement dominates. This is the workhorse.
* **Containment** (overlap coefficient) handles truncation, which is rife here: 'C & G'
  inside 'C & G Homes LLC', 'Sterling Eastern' inside 'Sterling Eastern Inc'. Asymmetric by
  nature, so it needs a discount when used as evidence.
* **Character n-gram Jaccard/Dice** is the only thing that catches typos, transposition and
  the 5.2% of names rendered as concatenated domains ('lumynoriental.com').
* **Numeric agreement** on house/plot numbers is the sharpest single address signal, but
  ~21% of true pairs share no number at all, so absence must be distinguished from
  disagreement.
"""


def jaccard(a, b):
    """|a & b| / |a | b|. Empty input means no evidence, not disagreement."""
    if not a or not b:
        return 0.0
    n = len(a & b)
    if not n:
        return 0.0
    return n / (len(a) + len(b) - n)


def dice(a, b):
    """2|a & b| / (|a| + |b|). More forgiving than Jaccard when sizes differ."""
    if not a or not b:
        return 0.0
    n = len(a & b)
    if not n:
        return 0.0
    return 2.0 * n / (len(a) + len(b))


def containment(a, b):
    """|a & b| / min(|a|, |b|) - the overlap coefficient."""
    if not a or not b:
        return 0.0
    n = len(a & b)
    if not n:
        return 0.0
    return n / (len(a) if len(a) < len(b) else len(b))


def weighted_jaccard(a, b, idf, default):
    """Generalised Jaccard with IDF weights: sum(idf over a&b) / sum(idf over a|b).

    Prefer this to cosine for scoring: cosine normalises each side independently and so
    rewards a single rare shared token even when the rest of the record disagrees, while
    this form charges for every unmatched token.
    """
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    num = 0.0
    for t in inter:
        num += idf.get(t, default)
    den = num
    for t in a ^ b:  # symmetric difference: tokens on exactly one side
        den += idf.get(t, default)
    return num / den if den > 0.0 else 0.0


def idf_cosine(a, b, idf, default):
    """IDF-weighted cosine over token sets. Kept for comparison against weighted_jaccard."""
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    num = sum(idf.get(t, default) ** 2 for t in inter)
    na = sum(idf.get(t, default) ** 2 for t in a)
    nb = sum(idf.get(t, default) ** 2 for t in b)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return num / ((na ** 0.5) * (nb ** 0.5))


def numeric_agreement(a, b):
    """(score, had_evidence) for house/plot numbers.

    Returning the flag separately matters: 21% of true pairs share no numeric token, and
    scoring those as 0.0 would penalise them as if the numbers actively disagreed.
    """
    if not a or not b:
        return 0.0, False
    n = len(a & b)
    if not n:
        return 0.0, True
    return n / (len(a) if len(a) < len(b) else len(b)), True
