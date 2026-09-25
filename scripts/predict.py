"""Generate submission files for the test set.

Writes both files the challenge requires into ``output/``:

* ``matching_results.tsv`` - the final matches, the only file scored on the leaderboard.
* ``candidate_pairs.tsv``  - the candidate set the matcher scored, i.e. everything that
  cleared the prefilter. Final matches are a strict subset by construction.

Every Source-1 entity gets exactly one row, with an empty second field where we predict no
matches. A missing row causes outright rejection, so the row order is read back from the test
file itself rather than from whatever the shards happened to produce.

    python scripts/predict.py --threshold 0.725 --workers 16

IDF is rebuilt from the *test* sources, not reused from training: the statistics must
describe the pool being scored against, and the test set carries 23% more Source-2/3 records
per Source-1 entity than training does.
"""

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber.blocking import build_corpus_stats
from ber.config import ARTIFACTS, EMBEDDINGS, OUTPUT, TEST, WORK, ensure_dirs, test_paths
from ber.dataio import CANDIDATE_HEADER, MATCHING_HEADER, country_counts, write_id_lists
from ber.parallel import run_parallel
from ber.select import select_threshold


def log(msg):
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, required=True,
                    help="final decision threshold, tuned by scripts/validate.py")
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count - 1")
    ap.add_argument("--chunks-per-country", type=int, default=0, help="0 = same as workers")
    ap.add_argument("--df-cap", type=int, default=60)
    ap.add_argument("--max-posting", type=int, default=200)
    ap.add_argument("--prefilter", type=float, default=0.34)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--max-k", type=int, default=None,
                    help="optional cap on matches emitted per entity")
    ap.add_argument("--embeddings", default=None,
                    help="basename under the embeddings dir, e.g. test_names")
    args = ap.parse_args()

    ensure_dirs()
    paths = test_paths()
    s1, s23 = paths["s1"], paths["s23"]

    stats_path = os.path.join(ARTIFACTS, "corpus_stats_test.pkl")
    if os.path.exists(stats_path):
        log(f"using cached test corpus stats: {stats_path}")
    else:
        log("building test corpus statistics (1-in-10 sample of test S2/S3) ...")
        df_name, df_addr, n_docs = build_corpus_stats(s23, sample_every=10)
        with open(stats_path, "wb") as fh:
            pickle.dump((df_name, df_addr, n_docs), fh)
        log(f"corpus: {n_docs:,} sampled docs, {len(df_name):,} name / "
            f"{len(df_addr):,} address tokens")

    counts = country_counts(s1)
    log(f"test Source 1 by country: {counts}")
    countries = sorted(counts, key=lambda c: -counts[c])

    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)
    scored, _, shard_stats = run_parallel(
        countries, s1, s23, stats_path, os.path.join(WORK, "tmp"),
        workers=workers, chunks_per_country=args.chunks_per_country or workers,
        prefilter=args.prefilter, topk=args.topk, df_cap=args.df_cap,
        max_posting=args.max_posting,
        emb_dir=EMBEDDINGS if args.embeddings else None, emb_name=args.embeddings,
        log=log)

    total_considered = sum(s["pairs_considered"] for s in shard_stats)
    total_scanned = sum(s["s23_scanned"] for s in shard_stats)
    log(f"{total_scanned:,} record-scans, {total_considered:,} pairs considered")

    # --- write output in test-file order so no entity is missing ---
    order = []
    with open(s1, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            i = line.find("\t")
            if i > 0:
                order.append(line[:i])

    missing = sum(1 for e in order if e not in scored)
    if missing:
        log(f"WARNING: {missing:,} entities produced no shard entry; emitting empty rows")

    def rows(final):
        for eid in order:
            sc = scored.get(eid) or []
            yield eid, (select_threshold(sc, args.threshold, args.max_k) if final
                        else [t for _, t in sc])

    n_m = write_id_lists(os.path.join(OUTPUT, "matching_results.tsv"), rows(True),
                         MATCHING_HEADER)
    n_c = write_id_lists(os.path.join(OUTPUT, "candidate_pairs.tsv"), rows(False),
                         CANDIDATE_HEADER)

    links = sum(len(select_threshold(scored.get(e) or [], args.threshold, args.max_k))
                for e in order)
    cands = sum(len(scored.get(e) or []) for e in order)
    empties = sum(1 for e in order
                  if not select_threshold(scored.get(e) or [], args.threshold, args.max_k))
    log("")
    log(f"matching_results.tsv : {n_m:,} rows, {links:,} links, "
        f"{links / max(1, n_m):.2f}/entity, {empties:,} empty ({empties / max(1, n_m):.2%})")
    log(f"candidate_pairs.tsv  : {n_c:,} rows, {cands:,} candidates, "
        f"{cands / max(1, n_c):.2f}/entity")
    log("")
    log("validate the format before submitting:")
    log("  python student_resource/utils/validate_submission.py \\")
    log("      --matching output/matching_results.tsv \\")
    log("      --candidate output/candidate_pairs.tsv \\")
    log("      --test-dir student_resource/dataset/test")


if __name__ == "__main__":
    main()
