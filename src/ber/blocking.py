"""Candidate generation (blocking) plus the corpus statistics it depends on.

Blocking decides the recall ceiling of the whole pipeline, so the design follows directly
from what the training data showed:

* **Country is a hard partition.** All 191,388 sampled true pairs agree on country, with
  zero exceptions, so a cross-country candidate is never generated.
* **No single key family is enough.** Name-token overlap reaches only 85.6% recall on its
  own and address-token overlap 95.6%, but their union reaches 99.98%. Both paths are
  therefore always active, which is also what rescues the 1.76% of true matches whose name
  has been replaced outright and the 7.3% written in a non-Latin script.
* **Rare tokens only.** Keying on 'road' or 'delhi' produces enormous useless blocks. Keys
  are built from the rarest tokens in a record and any key whose posting list grows beyond
  ``max_posting`` is dropped as non-discriminative.

An earlier version keyed on bare numeric tokens; common values like '1' and '100' produced
a candidate explosion that reached 13.9 GB of RAM. Numbers now only appear inside composite
keys.
"""

import collections
import math

from .normalize import addr_tokens, fold, name_core, numeric_tokens


def build_corpus_stats(paths, sample_every=10, min_len=3):
    """Document frequencies for name and address tokens, from a strided sample.

    A 1-in-10 stride over 10.3M records is ~1M documents, plenty for stable IDF while
    keeping the pass fast. Returns (df_name, df_addr, n_docs).
    """
    df_name = collections.Counter()
    df_addr = collections.Counter()
    n_docs = 0
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            fh.readline()
            for i, line in enumerate(fh):
                if i % sample_every:
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                n_docs += 1
                country = parts[3]
                df_name.update({t for t in name_core(parts[1]) if len(t) >= min_len})
                df_addr.update({t for t in addr_tokens(parts[2], country) if len(t) >= min_len})
    return df_name, df_addr, n_docs


def idf_from_df(df, n_docs):
    """Smoothed inverse document frequency, plus the value to use for unseen tokens.

    A token missing from the sample is rarer than anything in it, so it gets the ceiling.
    """
    idf = {t: math.log(n_docs / (1.0 + c)) for t, c in df.items()}
    default = math.log(n_docs / 1.0)
    return idf, default


class Blocker:
    """Generates candidate keys for a record and holds the Source-1 inverted index.

    ``df_cap`` is expressed in sampled-document counts: a token seen at most ``df_cap``
    times in the ~1M-document sample is treated as rare enough to key on alone.
    """

    def __init__(self, df_name, df_addr, df_cap=60, n_addr_keys=3, n_name_keys=2,
                 max_posting=200, min_len=3):
        self.df_name = df_name
        self.df_addr = df_addr
        self.df_cap = df_cap
        self.n_addr_keys = n_addr_keys
        self.n_name_keys = n_name_keys
        self.max_posting = max_posting
        self.min_len = min_len
        self.index = collections.defaultdict(list)

    def _rare_first(self, tokens, df):
        """Tokens sorted rarest-first; ties broken alphabetically so that both sides of a
        pair choose the same tokens deterministically."""
        return sorted({t for t in tokens if len(t) >= self.min_len},
                      key=lambda t: (df.get(t, 0), t))

    def keys(self, name, addr, country):
        """Blocking keys for one record.

        Single-token keys fire only for genuinely rare tokens. Composite keys always fire:
        pairing a name token with an address token is selective by construction even when
        each token alone is common, which is what covers generically named businesses.
        """
        ks = set()
        at = self._rare_first(addr_tokens(addr, country), self.df_addr)
        nt = self._rare_first(name_core(name), self.df_name)
        top_a = at[:self.n_addr_keys]
        top_n = nt[:self.n_name_keys]

        for t in top_a:
            if self.df_addr.get(t, 0) <= self.df_cap:
                ks.add(("a", country, t))
        for t in top_n:
            if self.df_name.get(t, 0) <= self.df_cap:
                ks.add(("n", country, t))
        # name x address composite - the workhorse for common names at distinct addresses
        for t in top_n[:2]:
            for u in top_a[:2]:
                ks.add(("x", country, t, u))
        # address x address, so records with no usable name still get candidates
        if len(at) >= 2:
            ks.add(("y", country, at[0], at[1]))
        # name x name, so records with an empty address still get candidates
        if len(nt) >= 2:
            ks.add(("z", country, nt[0], nt[1]))
        # street number paired with a rare address token
        if top_a:
            for num in sorted(numeric_tokens(addr))[:2]:
                ks.add(("d", country, num, top_a[0]))
        return ks

    def index_source1(self, records):
        """Index Source-1 records. ``records`` yields (slot, name, addr, country).

        ``slot`` is the caller's integer handle for the record, kept small so postings stay
        compact. Keys that end up too popular are dropped afterwards.
        """
        idx = self.index
        for slot, name, addr, country in records:
            for k in self.keys(name, addr, country):
                idx[k].append(slot)
        dropped = [k for k, v in idx.items() if len(v) > self.max_posting]
        for k in dropped:
            del idx[k]
        return {"keys": len(idx), "dropped_keys": len(dropped),
                "postings": sum(len(v) for v in idx.values())}

    def candidates(self, name, addr, country):
        """Source-1 slots that share at least one key with this record."""
        idx = self.index
        out = set()
        for k in self.keys(name, addr, country):
            v = idx.get(k)
            if v:
                out.update(v)
        return out
