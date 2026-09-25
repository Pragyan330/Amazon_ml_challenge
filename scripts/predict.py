"""Generate submission files for the test set.

Writes both files the challenge requires into ``output/``:

* ``matching_results.tsv`` - the final matches, the only file scored on the leaderboard.
* ``candidate_pairs.tsv``  - the candidate set fed to the matcher, i.e. everything that
  cleared the prefilter. Final matches are a strict subset by construction.

Every Source-1 entity gets exactly one row, with an empty second field when we predict no
matches; a missing row causes outright rejection.

    python scripts/predict.py --threshold 0.60

Countries are processed one at a time and Source 1 can be further chunked with
``--s1-chunk`` if memory is tight - each chunk costs one extra pass over Source 2/3, so
leave it unset unless a shard does not fit.
"""

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber.blocking import Blocker, build_corpus_stats, idf_from_df
from ber.dataio import CANDIDATE_HEADER, MATCHING_HEADER, country_counts, write_id_lists
from ber.pipeline import as_id_map, run_shard
from ber.scorer import Weights
from ber.select import select_threshold

ROOT = os.path.join(os.path.dirname(__file__), "..")
TEST = os.path.join(ROOT, "student_resource", "dataset", "test")
CACHE = os.path.join(ROOT, "artifacts")
OUT = os.path.join(ROOT, "output")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def chunker(chunk_index, n_chunks):
    """Keep Source-1 entities whose id hashes into this chunk."""
    if n_chunks <= 1:
        return None

    def keep(eid):
        try:
            return int(eid.split("-", 1)[1]) % n_chunks == chunk_index
        except (IndexError, ValueError):
            return chunk_index == 0
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, required=True,
                    help="final decision threshold, tuned by scripts/validate.py")
    ap.add_argument("--df-cap", type=int, default=60)
    ap.add_argument("--max-posting", type=int, default=200)
    ap.add_argument("--prefilter", type=float, default=0.34)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--max-k", type=int, default=None,
                    help="optional cap on matches emitted per entity")
    ap.add_argument("--s1-chunk", type=int, default=1,
                    help="split each country's Source 1 into this many chunks")
    args = ap.parse_args()

    s1 = os.path.join(TEST, "test_source1.tsv")
    s23 = [os.path.join(TEST, "test_source2.tsv"), os.path.join(TEST, "test_source3.tsv")]

    # Corpus statistics come from the *test* sources: IDF must reflect the pool we are
    # actually scoring against, and the test set is 23% denser in S2/S3 per S1 than train.
    stats_path = os.path.join(CACHE, "corpus_stats_test.pkl")
    if os.path.exists(stats_path):
        log("loading cached test corpus stats")
        with open(stats_path, "rb") as fh:
            df_name, df_addr, n_docs = pickle.load(fh)
    else:
        log("building test corpus statistics...")
        df_name, df_addr, n_docs = build_corpus_stats(s23, sample_every=10)
        os.makedirs(CACHE, exist_ok=True)
        with open(stats_path, "wb") as fh:
            pickle.dump((df_name, df_addr, n_docs), fh)
    log(f"corpus: {n_docs:,} sampled docs")

    idf_name, default_idf = idf_from_df(df_name, n_docs)
    idf_addr, _ = idf_from_df(df_addr, n_docs)
    weights = Weights()

    counts = country_counts(s1)
    log(f"test Source 1 by country: {counts}")

    matches, candidates = {}, {}
    for country in sorted(counts, key=lambda c: -counts[c]):
        for chunk in range(args.s1_chunk):
            blocker = Blocker(df_name, df_addr, df_cap=args.df_cap,
                              max_posting=args.max_posting)
            result = run_shard(country, s1, s23, blocker, idf_name, idf_addr, default_idf,
                               weights, prefilter=args.prefilter, topk=args.topk,
                               s1_keep=chunker(chunk, args.s1_chunk), log=log)
            for eid, scored in as_id_map(result).items():
                candidates[eid] = [tid for _, tid in scored]
                matches[eid] = select_threshold(scored, args.threshold, args.max_k)
            del blocker, result

    # Read the id order straight from the test file so every entity appears exactly once.
    order = []
    with open(s1, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            eid = line[:line.find("\t")]
            if eid:
                order.append(eid)
    missing = [e for e in order if e not in matches]
    if missing:
        log(f"WARNING: {len(missing):,} entities never reached a shard; emitting empty rows")
        for e in missing:
            matches[e] = []
            candidates[e] = []

    n_m = write_id_lists(os.path.join(OUT, "matching_results.tsv"),
                         ((e, matches[e]) for e in order), MATCHING_HEADER)
    n_c = write_id_lists(os.path.join(OUT, "candidate_pairs.tsv"),
                         ((e, candidates[e]) for e in order), CANDIDATE_HEADER)
    total_m = sum(len(v) for v in matches.values())
    total_c = sum(len(v) for v in candidates.values())
    empties = sum(1 for v in matches.values() if not v)
    log("")
    log(f"matching_results.tsv : {n_m:,} rows, {total_m:,} links, "
        f"{total_m / max(1, n_m):.2f}/entity, {empties:,} empty ({empties / max(1, n_m):.2%})")
    log(f"candidate_pairs.tsv  : {n_c:,} rows, {total_c:,} candidates, "
        f"{total_c / max(1, n_c):.2f}/entity")
    log("")
    log("now validate the format:")
    log("  python student_resource/utils/validate_submission.py \\")
    log("      --matching output/matching_results.tsv \\")
    log("      --candidate output/candidate_pairs.tsv \\")
    log("      --test-dir student_resource/dataset/test")


if __name__ == "__main__":
    main()
