"""Candidate generation (blocking) and the corpus statistics it needs.

Blocking sets the recall ceiling for everything downstream, so the design follows what the
training data showed:

* **Country is a hard partition.** All 191,388 sampled true pairs agree on country, zero
  exceptions, so a cross-country candidate is never generated.
* **No single key family suffices.** Name-token overlap alone reaches 85.6% recall and
  address-token overlap 95.6%, but their union reaches 99.98%. Both paths stay active, which
  is what rescues the 1.76% of true matches whose name was replaced outright and the 7.3%
  written in a non-Latin script.
* **Rare tokens only.** Keying on 'road' or 'delhi' builds enormous useless blocks, so keys
  come from a record's rarest tokens and any key whose posting list grows past
  ``max_posting`` is dropped as non-discriminative.

Keys are derived from a pre-built :class:`~ber.record.Rec`, which already holds the
rarest-first token lists. An earlier version re-tokenised inside this module, duplicating
more than a third of total runtime.

Cost note: numeric tokens appear only inside composite keys. Keying on them directly - where
values like '1' and '100' are common - caused a candidate explosion that reached 13.9 GB.
"""

import collections
import math

from .normalize import addr_tokens, name_core, numeric_tokens


def build_corpus_stats(paths, sample_every=10, min_len=3):
    """Document frequencies for name and address tokens, from a strided sample.

    A 1-in-10 stride over 10.3M records gives ~1M documents, ample for stable IDF.
    Returns (df_name, df_addr, n_docs).
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
                df_name.update({t for t in name_core(parts[1]) if len(t) >= min_len})
                df_addr.update({t for t in addr_tokens(parts[2], parts[3])
                                if len(t) >= min_len})
    return df_name, df_addr, n_docs


def idf_from_df(df, n_docs):
    """Smoothed IDF plus the value for unseen tokens.

    A token absent from a million-document sample is rarer than anything in it, so it gets
    the ceiling rather than being silently treated as common.
    """
    idf = {t: math.log(n_docs / (1.0 + c)) for t, c in df.items()}
    return idf, math.log(n_docs)


class Blocker:
    """Generates keys for a record and owns the Source-1 inverted index.

    ``df_cap`` is in sampled-document units: a token appearing at most ``df_cap`` times in
    the ~1M-document sample is rare enough to key on by itself.
    """

    def __init__(self, df_name, df_addr, df_cap=60, n_addr_keys=3, n_name_keys=2,
                 max_posting=200):
        self.df_name = df_name
        self.df_addr = df_addr
        self.df_cap = df_cap
        self.n_addr_keys = n_addr_keys
        self.n_name_keys = n_name_keys
        self.max_posting = max_posting
        self.index = collections.defaultdict(list)

    def keys(self, rec, country, addr_raw=None):
        """Blocking keys for a prepared record.

        Single-token keys fire only for genuinely rare tokens. Composite keys always fire,
        because pairing two tokens is selective by construction even when each is
        individually common - that is what covers generically named businesses, and 39.5% of
        Source-1 entities share a name with another entity in the same country.
        """
        ks = set()
        top_a = rec.rare_a[:self.n_addr_keys]
        top_n = rec.rare_n[:self.n_name_keys]
        df_a, df_n, cap = self.df_addr, self.df_name, self.df_cap

        for t in top_a:
            if df_a.get(t, 0) <= cap:
                ks.add(("a", country, t))
        for t in top_n:
            if df_n.get(t, 0) <= cap:
                ks.add(("n", country, t))
        # name x address: the workhorse for a common name at a distinct address
        for t in top_n[:2]:
            for u in top_a[:2]:
                ks.add(("x", country, t, u))
        # address x address: reaches records whose name is unusable (replaced or Indic)
        if len(rec.rare_a) >= 2:
            ks.add(("y", country, rec.rare_a[0], rec.rare_a[1]))
        # name x name: reaches records with an empty address
        if len(rec.rare_n) >= 2:
            ks.add(("z", country, rec.rare_n[0], rec.rare_n[1]))
        # street number paired with the rarest address token
        if top_a and rec.an:
            for n in sorted(rec.an)[:2]:
                ks.add(("d", country, n, top_a[0]))
        return ks

    def index_source1(self, items, country):
        """Index Source-1 records. ``items`` yields (slot, Rec)."""
        idx = self.index
        for slot, rec in items:
            for k in self.keys(rec, country):
                idx[k].append(slot)
        dropped = [k for k, v in idx.items() if len(v) > self.max_posting]
        for k in dropped:
            del idx[k]
        return {"keys": len(idx), "dropped_keys": len(dropped),
                "postings": sum(len(v) for v in idx.values())}

    def candidates(self, rec, country):
        """Source-1 slots sharing at least one key with this record."""
        idx = self.index
        out = set()
        for k in self.keys(rec, country):
            v = idx.get(k)
            if v:
                out.update(v)
        return out
