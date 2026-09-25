"""Candidate generation and scoring, one country shard at a time.

Source 1 is held in memory and indexed; Sources 2 and 3 stream past it exactly once. That
direction is deliberate - S2+S3 is ~10M records, roughly five times Source 1, so it is the
side we refuse to materialise.

Each streamed record is normalised **once** into a :class:`~ber.record.Rec` that serves both
blocking and scoring. The previous version normalised twice (once for keys, once for
scoring), which accounted for 57% of validation runtime.

Only candidates clearing ``prefilter`` are retained. Without it the candidate set for the
full test set would be on the order of a billion pairs.
"""

import heapq
import os
import time

from .dataio import read_source
from .record import build
from .scorer import score_pair


class ShardResult:
    """Scored candidates for one country shard."""

    __slots__ = ("country", "s1_ids", "candidates", "stats")

    def __init__(self, country, s1_ids, candidates, stats):
        self.country = country
        self.s1_ids = s1_ids          # slot -> entity id
        self.candidates = candidates  # slot -> list of (score, target id), best first
        self.stats = stats


def run_shard(country, s1_path, s23_paths, blocker, idf_name, idf_addr, default_idf,
              weights, prefilter=0.34, topk=40, s1_keep=None, log=None):
    """Generate and score candidates for every Source-1 entity in one country.

    ``prefilter`` is the final blocking stage and is what ``candidate_pairs.tsv`` reports;
    ``topk`` caps survivors per entity, bounding memory on pathological blocks.
    """
    t0 = time.time()
    df_name, df_addr = blocker.df_name, blocker.df_addr

    def say(msg):
        if log:
            log(f"[{country}] {msg}")

    # --- load and index Source 1 ---
    s1_ids, s1_recs = [], []
    for eid, name, addr, ctry in read_source(s1_path, country=country, keep=s1_keep):
        s1_ids.append(eid)
        s1_recs.append(build(name, addr, ctry, df_name, df_addr))
    say(f"loaded {len(s1_ids):,} S1 records in {time.time() - t0:.0f}s")

    st = blocker.index_source1(enumerate(s1_recs), country)
    say(f"indexed: {st['keys']:,} keys, {st['postings']:,} postings, "
        f"{st['dropped_keys']:,} dropped as too common")

    # --- stream Sources 2 and 3 past the index ---
    cand = {}
    scanned = considered = kept = gated = 0
    index = blocker.index
    keys_of = blocker.keys
    for path in s23_paths:
        for eid, name, addr, ctry in read_source(path, country=country):
            scanned += 1
            rec = build(name, addr, ctry, df_name, df_addr)
            slots = set()
            for k in keys_of(rec, country):
                v = index.get(k)
                if v:
                    slots.update(v)
            if not slots:
                continue
            considered += len(slots)
            for slot in slots:
                s = score_pair(s1_recs[slot], rec, idf_name, idf_addr, default_idf, weights)
                if s < prefilter:
                    if s == 0.0:
                        gated += 1
                    continue
                bucket = cand.get(slot)
                if bucket is None:
                    cand[slot] = [(s, eid)]
                    kept += 1
                elif len(bucket) < topk:
                    heapq.heappush(bucket, (s, eid))
                    kept += 1
                elif s > bucket[0][0]:
                    heapq.heapreplace(bucket, (s, eid))
            if log and scanned % 1_000_000 == 0:
                say(f"scanned {scanned:,}, kept {kept:,} ({time.time() - t0:.0f}s)")
        say(f"finished {os.path.basename(path)} (scanned {scanned:,})")

    for bucket in cand.values():
        bucket.sort(reverse=True)

    elapsed = time.time() - t0
    stats = {
        "s1_records": len(s1_ids),
        "s23_scanned": scanned,
        "pairs_considered": considered,
        "pairs_gated": gated,
        "pairs_kept": kept,
        "candidates_per_s1": kept / max(1, len(s1_ids)),
        "considered_per_s1": considered / max(1, len(s1_ids)),
        "gate_reject_rate": gated / max(1, considered),
        "seconds": elapsed,
        "us_per_pair": elapsed / max(1, considered) * 1e6,
    }
    say(f"done: {stats['considered_per_s1']:.1f} considered/S1 -> "
        f"{stats['candidates_per_s1']:.1f} kept/S1, gate rejected "
        f"{stats['gate_reject_rate']:.1%}, {elapsed:.0f}s "
        f"({stats['us_per_pair']:.1f} us/pair)")
    return ShardResult(country, s1_ids, cand, stats)


def as_id_map(result):
    """{entity_id: [(score, target_id), ...]}, including entities with no candidates."""
    return {eid: result.candidates.get(slot, [])
            for slot, eid in enumerate(result.s1_ids)}
