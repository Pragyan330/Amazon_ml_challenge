"""End-to-end candidate generation and scoring, one country shard at a time.

Shape of the computation: Source 1 is held in memory and indexed, then Sources 2 and 3 are
streamed past it exactly once. That direction matters - S2+S3 is ~10M records and roughly
five times the size of S1, so it is the side we refuse to materialise.

Scores are computed during the stream and only candidates clearing ``prefilter`` are
retained. Without that filter the candidate set for the full test set would be on the order
of a billion pairs; with it, storage collapses to a few per entity because the negative
score distribution is concentrated far below the positives (measured means: 0.13 negative
vs 0.70 positive).
"""

import heapq
import time

from .dataio import read_source
from .scorer import prepare, score_pair


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

    ``prefilter`` is the last blocking stage: it is what ``candidate_pairs.tsv`` reports.
    ``topk`` caps how many survive per entity, bounding memory on pathological blocks.
    """
    t0 = time.time()

    def say(msg):
        if log:
            log(f"[{country}] {msg}")

    # --- load and index Source 1 ---
    s1_ids, s1_prepared = [], []
    for eid, name, addr, ctry in read_source(s1_path, country=country, keep=s1_keep):
        s1_ids.append(eid)
        s1_prepared.append(prepare(name, addr, ctry))
    say(f"loaded {len(s1_ids):,} S1 records in {time.time() - t0:.0f}s")

    idx_stats = blocker.index_source1(
        (slot, *_raw(s1_prepared[slot]), country) for slot in range(len(s1_ids))
    )
    say(f"indexed: {idx_stats['keys']:,} keys, {idx_stats['postings']:,} postings, "
        f"{idx_stats['dropped_keys']:,} keys dropped as too common")

    # --- stream Sources 2 and 3 past the index ---
    cand = {}
    scanned = considered = kept = 0
    keys_of = blocker.keys
    index = blocker.index
    for path in s23_paths:
        for eid, name, addr, ctry in read_source(path, country=country):
            scanned += 1
            slots = set()
            for k in keys_of(name, addr, ctry):
                v = index.get(k)
                if v:
                    slots.update(v)
            if not slots:
                continue
            r2 = prepare(name, addr, ctry)
            considered += len(slots)
            for slot in slots:
                s = score_pair(s1_prepared[slot], r2, idf_name, idf_addr, default_idf,
                               weights)
                if s < prefilter:
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
                say(f"scanned {scanned:,} S2/S3 records, kept {kept:,} candidates "
                    f"({time.time() - t0:.0f}s)")
        say(f"finished {path.rsplit('/', 1)[-1]} (scanned {scanned:,})")

    for slot, bucket in cand.items():
        bucket.sort(reverse=True)

    stats = {
        "s1_records": len(s1_ids),
        "s23_scanned": scanned,
        "pairs_considered": considered,
        "pairs_kept": kept,
        "candidates_per_s1": kept / max(1, len(s1_ids)),
        "considered_per_s1": considered / max(1, len(s1_ids)),
        "seconds": time.time() - t0,
    }
    say(f"done: {stats['candidates_per_s1']:.1f} candidates/S1 kept from "
        f"{stats['considered_per_s1']:.1f} considered, {stats['seconds']:.0f}s")
    return ShardResult(country, s1_ids, cand, stats)


def _raw(prepared):
    """Recover the original (name, address) a prepared record was built from."""
    from .scorer import RAW
    return prepared[RAW]


def as_id_map(result):
    """{entity_id: [(score, target_id), ...]} for a shard, including empty entities."""
    out = {}
    for slot, eid in enumerate(result.s1_ids):
        out[eid] = result.candidates.get(slot, [])
    return out
