"""Rule-based pair scorer.

Three changes from the first version, each driven by a measurement:

**1. A cheap gate (speed).** Plain token Jaccard on name and address is computed first from
sets that already exist. If neither reaches ``GATE`` the pair is rejected before any
character n-gram is built. Measured on 191,388 true pairs, a gate of 0.10 loses 0.021% of
them - effectively free - while most negatives in a candidate set share only the single rare
token that blocked them together, putting them near 1/19 ~ 0.05.

**2. Weighted view combination instead of max() (accuracy).** Taking the best of several
views is optimistic by construction: a negative pair gets credit for whichever view happens
to be most generous. That showed up as badly degraded separation - negatives at mean 0.512
and p90 0.806 against positives at 0.906. Views are now combined with weights.

**3. Channel-specific confidence (accuracy).** The previous single symmetric
``missing_penalty`` priced a perfect name-only match at 0.8956 and a perfect address-only
match at 0.9010 - nearly identical. The data says they are nothing alike:

    perfect name match, no address   : 10,194 positives vs 131,231 negatives  (1 : 12.9)
    perfect address match, no name   :    710 positives vs     136 negatives  (5.2 : 1)

That single name-only spike held 131,231 of the 3.21M negatives, and the optimal threshold
of 0.900 was earning its score purely by sitting just above it. Confidence is now indexed by
*which* channels carried evidence, so name-only can be priced down to what it is worth.
"""

from .features import (
    containment,
    dice,
    jaccard,
    numeric_agreement,
    weighted_jaccard,
)

# Cheap-gate threshold on max(plain name Jaccard, plain address Jaccard).
GATE = 0.10

# Evidence mask bits.
HAS_NAME, HAS_ADDR, HAS_NUM = 1, 2, 4


class Weights:
    """Tunable scorer parameters.

    ``conf`` is indexed by the evidence mask (name=1, addr=2, num=4) and encodes how much to
    trust a score built from only those channels. The defaults follow the positive/negative
    ratios measured per channel combination rather than a single symmetric penalty.
    """

    __slots__ = ("n_wj", "n_ct", "n_ng", "a_wj", "a_ng", "c_name", "c_addr", "c_num",
                 "emb_trust", "emb_floor", "conf")

    def __init__(self, n_wj=0.45, n_ct=0.20, n_ng=0.35, a_wj=0.60, a_ng=0.40,
                 c_name=0.45, c_addr=0.42, c_num=0.13,
                 emb_trust=0.85, emb_floor=0.55, conf=None):
        self.n_wj = n_wj      # name: IDF-weighted Jaccard
        self.n_ct = n_ct      # name: containment (truncation)
        self.n_ng = n_ng      # name: character trigram Dice (typos, domains)
        self.a_wj = a_wj      # address: IDF-weighted Jaccard
        self.a_ng = a_ng      # address: character trigram Dice
        self.c_name = c_name  # channel weight: name
        self.c_addr = c_addr  # channel weight: address
        self.c_num = c_num    # channel weight: street number
        # Encoder-derived name evidence is softer than a string match, so it is discounted;
        # and below emb_floor the cosine is treated as no evidence rather than disagreement,
        # since multilingual encoders put unrelated short texts around 0.5-0.6.
        self.emb_trust = emb_trust
        self.emb_floor = emb_floor
        self.conf = conf if conf is not None else [
            0.00,  # 0  no evidence
            0.45,  # 1  name only      <- deliberately low: 13x more likely a false merge
            0.90,  # 2  address only   <- 5x more likely correct
            1.00,  # 3  name + address
            0.25,  # 4  number only
            0.55,  # 5  name + number
            0.95,  # 6  address + number
            1.00,  # 7  everything
        ]

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self):
        return ("Weights(name[wj=%.2f ct=%.2f ng=%.2f] addr[wj=%.2f ng=%.2f] "
                "chan[%.2f/%.2f/%.2f] conf=%s)" % (
                    self.n_wj, self.n_ct, self.n_ng, self.a_wj, self.a_ng,
                    self.c_name, self.c_addr, self.c_num,
                    "[" + ",".join("%.2f" % c for c in self.conf) + "]"))


def score_pair(r1, r2, idf_name, idf_addr, default_idf, w, emb_sim=None):
    """Score one (Source-1, Source-2/3) pair in [0, 1].

    ``emb_sim`` is an optional multilingual-encoder cosine, rescaled to [0, 1]. It is used
    only when the target name is non-Latin - exactly the case where string similarity is
    structurally zero and the pair currently has to survive on address evidence alone.
    """
    # --- cheap gate: reuses sets already built, no n-grams ---
    jn = jaccard(r1.nc, r2.nc)
    ja = jaccard(r1.at, r2.at)
    if jn < GATE and ja < GATE:
        # A non-Latin name can never clear the name side of the gate, so an encoder score
        # is the only thing that can rescue such a pair.
        if not (emb_sim is not None and not r2.latin and emb_sim >= w.emb_floor):
            return 0.0

    mask = 0
    acc = 0.0
    tot = 0.0

    # --- address channel ---
    if r1.at and r2.at:
        s = (w.a_wj * weighted_jaccard(r1.at, r2.at, idf_addr, default_idf)
             + w.a_ng * dice(r1.addr_grams(), r2.addr_grams()))
        acc += w.c_addr * s
        tot += w.c_addr
        mask |= HAS_ADDR

    # --- street number channel ---
    num, had_num = numeric_agreement(r1.an, r2.an)
    if had_num:
        acc += w.c_num * num
        tot += w.c_num
        mask |= HAS_NUM

    # --- name channel: string views when alphabets match, encoder when they do not ---
    if r2.latin:
        s = (w.n_wj * weighted_jaccard(r1.nc, r2.nc, idf_name, default_idf)
             + w.n_ct * containment(r1.nd, r2.nd)
             + w.n_ng * dice(r1.name_grams(), r2.name_grams()))
        acc += w.c_name * s
        tot += w.c_name
        mask |= HAS_NAME
        return (acc / tot) * w.conf[mask] if tot > 0.0 else 0.0

    if emb_sim is not None and emb_sim >= w.emb_floor:
        # Stretch [emb_floor, 1] onto [0, 1]: multilingual encoders place unrelated short
        # texts around 0.5-0.6, so the raw cosine floor is not meaningful signal.
        s = (emb_sim - w.emb_floor) / (1.0 - w.emb_floor)
        with_name = ((acc + w.c_name * w.emb_trust * s) / (tot + w.c_name)
                     * w.conf[mask | HAS_NAME])
        # Take the better of using the encoder and ignoring it, so adding evidence can never
        # lower a score. Without this, a strong address match (conf 0.95, base 1.0 -> 0.95)
        # was being dragged down to 0.86 by a perfectly good 0.92 cosine.
        without = (acc / tot) * w.conf[mask] if tot > 0.0 else 0.0
        return with_name if with_name > without else without

    if tot <= 0.0:
        return 0.0
    return (acc / tot) * w.conf[mask]
