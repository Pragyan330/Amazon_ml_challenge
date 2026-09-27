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

from array import array

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
    # cross-source corroboration, appended by add_cross_source_features. ORDER MATTERS:
    # these are appended after pair_features returns and before add_rank_features, so the
    # names must sit here, not next to the other per-pair features.
    "xs_max_opp", "xs_mean_opp", "xs_n_opp_strong", "xs_max_same",
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


CROSS_FEATURE_NAMES = ["xs_max_opp", "xs_mean_opp", "xs_n_opp_strong", "xs_max_same"]

_MASK32 = 0xFFFFFFFF


def token_signature(rec):
    """Compact, order-independent signature of a record's name+address tokens.

    Sorted 32-bit token hashes in an ``array('I')``. Measured at ~140 bytes against 1,126
    for the token set and 1,686 for the whole ``Rec``, which is what makes cross-source
    features affordable inside the submission workers: holding token sets for every
    candidate would need 8.3 GB across five workers, and whole records 12.4 GB.

    Unlike a MinHash sketch this keeps Jaccard **exact** (up to 32-bit hash collisions,
    negligible at ~15 tokens), so the feature means the same thing at training and at
    inference. Training on exact string Jaccard and inferring on an approximation would be
    the same class of silent train/serve mismatch that cost 0.026 with the encoder.
    """
    return array("I", sorted({hash(t) & _MASK32 for t in rec.nc | rec.at}))


def _sig_jaccard(a, b):
    """Jaccard over two sorted uint32 arrays, by merge."""
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    i = j = inter = 0
    while i < la and j < lb:
        x, y = a[i], b[j]
        if x == y:
            inter += 1; i += 1; j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    if not inter:
        return 0.0
    return inter / (la + lb - inter)


def add_cross_source_features(rows, sigs, is_s3_flags, strong=0.5):
    """Append cross-source agreement features, appended per entity.

    The strongest unexploited signal in this dataset. Measured on the training data, a
    Source-2 and a Source-3 record matching the *same* Source-1 entity have blended
    name+address similarity averaging **0.447**, while records matching *different* entities
    in the same country average **0.015** - and 0.00% of the cross-cluster pairs exceed 0.5
    against 42.7% of the within-cluster ones. Agreement between two candidates is therefore
    close to diagnostic on its own.

    A pairwise scorer cannot see this: it only ever compares a candidate to the Source-1
    record. Here each candidate is also compared to its *rivals* for the same entity, split
    by source. A Source-2 record that closely matches one of the Source-3 candidates is
    corroborated by an independent source; one that matches none of them is not.

    ``xs_max_same`` covers the opposite case - two near-identical records from the *same*
    source competing for one slot, where at most one is usually right.

    ``sigs`` are :func:`token_signature` arrays, not records, so the same code runs in the
    trainer and inside the submission workers with identical semantics.
    """
    n = len(rows)
    if n == 0:
        return rows
    sims = [[0.0] * n for _ in range(n)]
    for i in range(n):
        si = sigs[i]
        for j in range(i + 1, n):
            v = _sig_jaccard(si, sigs[j])
            sims[i][j] = sims[j][i] = v
    for i in range(n):
        opp = [sims[i][j] for j in range(n) if j != i and is_s3_flags[j] != is_s3_flags[i]]
        same = [sims[i][j] for j in range(n) if j != i and is_s3_flags[j] == is_s3_flags[i]]
        rows[i].extend([
            max(opp) if opp else 0.0,
            (sum(opp) / len(opp)) if opp else 0.0,
            float(sum(1 for s in opp if s >= strong)),
            max(same) if same else 0.0,
        ])
    return rows


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
