"""Rule-based pair scorer.

Stage 3 of the pipeline. Given a prepared S1 record and a prepared S2/S3 record it returns
a single score in [0, 1]. There is no learned component here by design - this is the
baseline we measure everything else against.

Two ideas do most of the work:

* **Evidence-weighted averaging.** Name, address and street-number each contribute only
  when that evidence actually exists. A record whose name is in Devanagari carries no
  usable name evidence against a romanised S1 name, and ~4.4% of true-match records have
  an empty address, so a fixed weighted sum would systematically under-score both. We sum
  only the components present and renormalise by the weight actually used.
* **A missing-evidence discount.** Renormalising alone would let a record with one strong
  signal score as high as one that agrees on everything. Since F_0.5 punishes false merges
  roughly four times harder than misses, partial evidence is damped.
"""

from .normalize import (
    addr_tokens,
    fold,
    is_latin,
    name_core,
    name_distinctive,
    name_squash,
    numeric_tokens,
)
from .similarity import containment, dice, idf_cosine, ngrams, numeric_agreement

# Index positions inside a prepared record tuple. Plain tuples rather than a class: the
# pipeline builds millions of these and attribute lookup is measurably slower.
N_TOK, N_CORE, N_DIST, N_GRAM, A_TOK, A_GRAM, A_NUM, LATIN, CTRY, RAW = range(10)


def prepare(name, addr, country):
    """Pre-compute every token set a pair score needs, once per record."""
    at = set(addr_tokens(addr, country))
    return (
        set(fold(name).split()),
        set(name_core(name)),
        set(name_distinctive(name)),
        ngrams(name_squash(name), 3),
        at,
        # n-grams over the sorted token string, so component reordering
        # ('Tennessee, Hixson, Kensley Ln' vs '... Kensley Ln, Lakesite, TN') still matches.
        ngrams(" ".join(sorted(at)), 3) if at else set(),
        numeric_tokens(addr),
        is_latin(name),
        country,
        (name, addr),
    )


class Weights:
    """Tunable knobs for the blend. Defaults are grid-searched on a validation split."""

    __slots__ = ("name", "addr", "num", "missing_penalty", "containment_discount")

    def __init__(self, name=0.42, addr=0.45, num=0.13,
                 missing_penalty=0.18, containment_discount=0.90):
        self.name = name
        self.addr = addr
        self.num = num
        self.missing_penalty = missing_penalty
        self.containment_discount = containment_discount

    def __repr__(self):
        return (f"Weights(name={self.name:.2f}, addr={self.addr:.2f}, num={self.num:.2f}, "
                f"missing_penalty={self.missing_penalty:.2f}, "
                f"containment_discount={self.containment_discount:.2f})")


def name_similarity(r1, r2, idf_name, default_idf, w):
    """Best available name evidence, or None when the names are not comparable.

    Three views are taken because the corruption modes are qualitatively different:
    token-level IDF cosine for word-order and legal-suffix noise, character n-grams for
    typos and concatenated/domain forms, and containment for truncation.
    """
    if not r2[LATIN]:
        return None  # romanised S1 vs Indic-script S2/S3: no shared alphabet
    return max(
        idf_cosine(r1[N_CORE], r2[N_CORE], idf_name, default_idf),
        dice(r1[N_GRAM], r2[N_GRAM]),
        w.containment_discount * containment(r1[N_DIST], r2[N_DIST]),
    )


def addr_similarity(r1, r2, idf_addr, default_idf):
    """Best available address evidence, or None when either address is empty."""
    if not r1[A_TOK] or not r2[A_TOK]:
        return None
    return max(
        idf_cosine(r1[A_TOK], r2[A_TOK], idf_addr, default_idf),
        dice(r1[A_GRAM], r2[A_GRAM]),
    )


def score_pair(r1, r2, idf_name, idf_addr, default_idf, w):
    """Blended score in [0, 1] for one (S1, S2/S3) pair."""
    total_w = 0.0
    acc = 0.0

    ns = name_similarity(r1, r2, idf_name, default_idf, w)
    if ns is not None:
        acc += w.name * ns
        total_w += w.name

    ads = addr_similarity(r1, r2, idf_addr, default_idf)
    if ads is not None:
        acc += w.addr * ads
        total_w += w.addr

    num, had_num = numeric_agreement(r1[A_NUM], r2[A_NUM])
    if had_num:
        acc += w.num * num
        total_w += w.num

    if total_w <= 0.0:
        return 0.0

    blended = acc / total_w
    # Damp the score when some evidence channels were unavailable.
    available = total_w / (w.name + w.addr + w.num)
    return blended * (1.0 - w.missing_penalty * (1.0 - available))
