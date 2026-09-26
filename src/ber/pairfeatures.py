"""Pairwise feature vectors for the learned matcher.

The rule scorer collapses everything into one number; a model wants the parts separately so
it can learn how they trade off. Three groups of features exist here that a straight port of
the rule scorer would miss:

* **Per-entity rank features.** How this candidate ranks against the *other* candidates for
  the same Source-1 entity, and how far behind the leader it sits. Matching is a competition
  within an entity - a 0.6 score means something completely different when it is the best
  candidate than when it is the fifth - and an absolute score cannot express that.
* **Source identity.** Source 2 and Source 3 corrupt differently (measured address token
  Jaccard to Source 1: 0.726 for S2, 0.558 for S3), so the same similarity carries different
  evidence depending on which file it came from.
* **Encoder cosine.** Present for non-Latin target names, where every string feature is
  structurally zero and the model would otherwise be blind.

Missing evidence is encoded as an explicit flag rather than a zero, so the model can tell
"no address to compare" apart from "addresses disagree".
"""

from .features import (
    containment,
    dice,
    jaccard,
    numeric_agreement,
    weighted_jaccard,
)

FEATURE_NAMES = [
    # name channel
    "n_jac", "n_dice", "n_cont", "n_wjac", "n_gram_dice", "n_gram_jac",
    "nd_jac", "nd_cont",
    # address channel
    "a_jac", "a_dice", "a_cont", "a_wjac", "a_gram_dice", "a_gram_jac",
    # numeric
    "num_score", "num_both_present", "num_disagree",
    # presence / shape
    "has_name_ev", "has_addr_ev", "target_latin",
    "n1_len", "n2_len", "a1_len", "a2_len", "len_ratio_name", "len_ratio_addr",
    # provenance
    "is_s3",
    # encoder
    "emb_cos", "emb_present",
    # rule score and per-entity competition
    "base_score", "rank", "score_gap_top", "score_ratio_top", "n_candidates",
    "base_minus_mean",
]


def pair_features(r1, r2, base_score, idf_name, idf_addr, default_idf, emb_cos, is_s3):
    """Feature vector for one (Source-1, Source-2/3) pair, minus the rank block."""
    if r2.latin:
        g1, g2 = r1.name_grams(), r2.name_grams()
        n_jac = jaccard(r1.nc, r2.nc)
        n_dice = dice(r1.nc, r2.nc)
        n_cont = containment(r1.nc, r2.nc)
        n_wjac = weighted_jaccard(r1.nc, r2.nc, idf_name, default_idf)
        n_gram_dice = dice(g1, g2)
        n_gram_jac = jaccard(g1, g2)
        nd_jac = jaccard(r1.nd, r2.nd)
        nd_cont = containment(r1.nd, r2.nd)
        has_name = 1.0
    else:
        n_jac = n_dice = n_cont = n_wjac = n_gram_dice = n_gram_jac = 0.0
        nd_jac = nd_cont = 0.0
        has_name = 0.0

    if r1.at and r2.at:
        ag1, ag2 = r1.addr_grams(), r2.addr_grams()
        a_jac = jaccard(r1.at, r2.at)
        a_dice = dice(r1.at, r2.at)
        a_cont = containment(r1.at, r2.at)
        a_wjac = weighted_jaccard(r1.at, r2.at, idf_addr, default_idf)
        a_gram_dice = dice(ag1, ag2)
        a_gram_jac = jaccard(ag1, ag2)
        has_addr = 1.0
    else:
        a_jac = a_dice = a_cont = a_wjac = a_gram_dice = a_gram_jac = 0.0
        has_addr = 0.0

    num_score, had_num = numeric_agreement(r1.an, r2.an)
    # Split "no numbers to compare" from "numbers present but different": the first is
    # neutral, the second is positive evidence *against* a match.
    num_both = 1.0 if had_num else 0.0
    num_disagree = 1.0 if (had_num and num_score == 0.0) else 0.0

    n1, n2 = len(r1.nc), len(r2.nc)
    a1, a2 = len(r1.at), len(r2.at)
    return [
        n_jac, n_dice, n_cont, n_wjac, n_gram_dice, n_gram_jac, nd_jac, nd_cont,
        a_jac, a_dice, a_cont, a_wjac, a_gram_dice, a_gram_jac,
        num_score, num_both, num_disagree,
        has_name, has_addr, 1.0 if r2.latin else 0.0,
        float(n1), float(n2), float(a1), float(a2),
        (min(n1, n2) / max(n1, n2)) if (n1 and n2) else 0.0,
        (min(a1, a2) / max(a1, a2)) if (a1 and a2) else 0.0,
        1.0 if is_s3 else 0.0,
        emb_cos if emb_cos is not None else 0.0,
        1.0 if emb_cos is not None else 0.0,
    ]


def add_rank_features(rows, base_scores):
    """Append the per-entity competition block to every row of one entity.

    ``rows`` are the feature vectors for a single Source-1 entity, ``base_scores`` their rule
    scores in the same order.
    """
    n = len(rows)
    if n == 0:
        return rows
    top = max(base_scores)
    mean = sum(base_scores) / n
    order = sorted(range(n), key=lambda i: -base_scores[i])
    rank_of = {}
    for pos, i in enumerate(order):
        rank_of[i] = pos
    for i, row in enumerate(rows):
        b = base_scores[i]
        row.extend([
            b,
            float(rank_of[i]),
            top - b,
            (b / top) if top > 0 else 0.0,
            float(n),
            b - mean,
        ])
    return rows
