"""One record, normalised exactly once.

The previous version normalised every record twice - once inside ``Blocker.keys()`` and
again inside ``prepare()`` - which was 57% of validation runtime (41.0 us + 38.6 us per
record, both re-running ``fold``, ``name_core`` and ``addr_tokens``). A ``Rec`` is built once
and serves both blocking and scoring.

Character n-grams are the expensive part (a 40-character address yields ~38 trigrams), so
they are built lazily: only pairs that survive the cheap gate in ``scorer`` ever need them,
and a record that never reaches that point never pays for them.
"""

from .normalize import (
    GENERIC,
    LEGAL,
    addr_tokens,
    fold,
    is_latin,
    name_squash,
    numeric_tokens,
)


class Rec:
    """Normalised record. Attributes are slots, so access stays cheap in the hot loop."""

    __slots__ = ("nc", "nd", "at", "an", "nsq", "latin", "rare_n", "rare_a",
                 "_ng", "_ag")

    def __init__(self, nc, nd, at, an, nsq, latin, rare_n, rare_a):
        self.nc = nc          # frozenset: name tokens, legal suffixes stripped
        self.nd = nd          # frozenset: also generic category words stripped
        self.at = at          # frozenset: address tokens, abbreviations expanded
        self.an = an          # frozenset: numeric address tokens (house/plot numbers)
        self.nsq = nsq        # str: squashed name, web decoration removed
        self.latin = latin    # bool: False means no usable name evidence vs a romanised S1
        self.rare_n = rare_n  # list: name tokens, rarest first (for blocking keys)
        self.rare_a = rare_a  # list: address tokens, rarest first
        self._ng = None       # lazy: name character trigrams
        self._ag = None       # lazy: address character trigrams

    def name_grams(self):
        g = self._ng
        if g is None:
            s = self.nsq
            g = self._ng = ({s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3
                            else ({s} if s else frozenset()))
        return g

    def addr_grams(self):
        """Trigrams over the sorted address tokens.

        Sorting makes this invariant to component reordering, which Source 3 does constantly
        ('Tennessee, Hixson, Kensley Ln' against '8526 Kensley Ln, Lakesite, TN').
        """
        g = self._ag
        if g is None:
            s = " ".join(sorted(self.at))
            g = self._ag = ({s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3
                            else ({s} if s else frozenset()))
        return g


def build(name, addr, country, df_name=None, df_addr=None, min_len=3):
    """Normalise one record in a single pass.

    When ``df_name``/``df_addr`` are supplied the rarest-first token lists needed for
    blocking keys are computed here too, so the caller never re-tokenises.
    """
    toks = fold(name).split()
    nc = frozenset(t for t in toks if t not in LEGAL) or frozenset(toks)
    nd = frozenset(t for t in nc if t not in GENERIC) or nc
    at = frozenset(addr_tokens(addr, country))

    if df_name is None:
        rare_n = rare_a = ()
    else:
        rare_n = sorted((t for t in nc if len(t) >= min_len),
                        key=lambda t: (df_name.get(t, 0), t))
        rare_a = sorted((t for t in at if len(t) >= min_len),
                        key=lambda t: (df_addr.get(t, 0), t))

    return Rec(nc, nd, at, numeric_tokens(addr), name_squash(name), is_latin(name),
               rare_n, rare_a)
