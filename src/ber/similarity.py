"""Pairwise similarity primitives used by the rule-based scorer.

All functions are pure and cheap; the pipeline calls them a few hundred million times, so
they take pre-computed token sets rather than raw strings wherever possible.
"""


def jaccard(a, b):
    """|a and b| / |a or b|. Two empty sets are treated as no evidence (0.0)."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def dice(a, b):
    """2|a and b| / (|a| + |b|) - more forgiving than Jaccard on size mismatch."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return 2.0 * inter / (len(a) + len(b))


def containment(a, b):
    """|a and b| / min(|a|, |b|).

    Handles truncation, which is common here: 'Sterling Eastern' inside
    'Sterling Eastern Inc', or 'C & G' inside 'C & G Homes LLC'.
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / min(len(a), len(b))


def idf_cosine(a, b, idf, default):
    """IDF-weighted cosine over two token sets.

    Weighting matters a lot on this data: sharing 'road' or 'delhi' is nearly meaningless
    while sharing 'circuler' is decisive, yet plain Jaccard scores them the same.
    """
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


def ngrams(s, n=3):
    """Character n-gram set. Short strings fall back to the whole string so that names
    under n characters ('C & G' -> 'cg') still produce a usable key."""
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def numeric_agreement(a, b):
    """Compare house/plot numbers, returning (score, had_evidence).

    Street numbers are the single most decisive address component, but ~21% of true pairs
    share none, so the caller must know whether evidence existed rather than reading a
    0.0 as disagreement.
    """
    if not a or not b:
        return 0.0, False
    inter = len(a & b)
    if not inter:
        return 0.0, True
    return inter / min(len(a), len(b)), True
