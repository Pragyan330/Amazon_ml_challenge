"""Produce the submission pair for the test set.

Writes into ``output/``:

* ``matching_results.tsv`` - final matches, the leaderboard file.
* ``candidate_pairs.tsv``  - the candidate set the matcher ran inference over. Final matches
  are a strict subset by construction.

Every Source-1 entity gets exactly one row, in test-file order, with an empty second field
where nothing is predicted. A missing row is an outright rejection, so the order is read back
from ``test_source1.tsv`` rather than from whatever the shards produced.

Each worker runs the whole chain for its Source-1 slice - blocking, rule score, feature
extraction, LightGBM, expected-F_0.5 selection - and returns only the two id lists. Features
are built inside the stream because both records are in hand there; deferring them would mean
rebuilding up to 10M ``Rec`` objects afterwards, which does not fit in memory.

IDF for blocking comes from the *test* sources, since the statistics must describe the pool
being searched, but the IDF used for *model features* comes from training, because that is
what the model was fitted against.

    D:\\ml_challenge\\venv\\Scripts\\python.exe scripts/submit.py \\
        --matcher matcher.pkl --embeddings test_names --workers 14
"""

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber.blocking import build_corpus_stats
from ber.config import ARTIFACTS, EMBEDDINGS, OUTPUT, WORK, ensure_dirs, test_paths
from ber.dataio import CANDIDATE_HEADER, MATCHING_HEADER, country_counts, write_id_lists


def log(msg):
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matcher", default="matcher.pkl")
    ap.add_argument("--embeddings", default="test_names")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--chunks-per-country", type=int, default=0)
    ap.add_argument("--prefilter", type=float, default=0.34)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--df-cap", type=int, default=60)
    ap.add_argument("--max-posting", type=int, default=200)
    ap.add_argument("--s1-limit", type=int, default=0,
                    help="smoke test: only the first N Source-1 entities")
    args = ap.parse_args()

    ensure_dirs()
    paths = test_paths()
    s1, s23 = paths["s1"], paths["s23"]

    if args.s1_limit:
        sliced = os.path.join(WORK, "tmp", "slice_test_source1.tsv")
        os.makedirs(os.path.dirname(sliced), exist_ok=True)
        with open(s1, encoding="utf-8") as src, \
                open(sliced, "w", encoding="utf-8", newline="\n") as dst:
            dst.write(src.readline())
            for i, line in enumerate(src):
                if i >= args.s1_limit:
                    break
                dst.write(line)
        s1 = sliced
        log(f"SMOKE TEST: only {args.s1_limit:,} Source-1 entities")

    stats_path = os.path.join(ARTIFACTS, "corpus_stats_test.pkl")
    if os.path.exists(stats_path):
        log("using cached test corpus stats")
    else:
        log("building test corpus statistics (1-in-10 sample) ...")
        st = build_corpus_stats(s23, sample_every=10)
        with open(stats_path, "wb") as fh:
            pickle.dump(st, fh)
        log(f"corpus: {st[2]:,} sampled docs")

    counts = country_counts(s1)
    log(f"test Source 1 by country: {counts}")
    countries = sorted(counts, key=lambda c: -counts[c])
    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)
    chunks = args.chunks_per_country or workers

    from ber.submitworker import run_submission
    sel, cand, stats = run_submission(
        countries, s1, s23, stats_path,
        train_stats_path=os.path.join(ARTIFACTS, "corpus_stats.pkl"),
        matcher_path=os.path.join(ARTIFACTS, args.matcher),
        out_dir=os.path.join(WORK, "tmp"), workers=workers, chunks_per_country=chunks,
        prefilter=args.prefilter, topk=args.topk, df_cap=args.df_cap,
        max_posting=args.max_posting,
        emb_dir=EMBEDDINGS if args.embeddings else None, emb_name=args.embeddings,
        log=log)

    order = []
    with open(s1, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            i = line.find("\t")
            if i > 0:
                order.append(line[:i])

    missing = sum(1 for e in order if e not in cand)
    if missing:
        log(f"note: {missing:,} entities produced no candidates; emitting empty rows")

    n_m = write_id_lists(os.path.join(OUTPUT, "matching_results.tsv"),
                         ((e, sel.get(e, ())) for e in order), MATCHING_HEADER)
    n_c = write_id_lists(os.path.join(OUTPUT, "candidate_pairs.tsv"),
                         ((e, cand.get(e, ())) for e in order), CANDIDATE_HEADER)

    links = sum(len(sel.get(e, ())) for e in order)
    cands = sum(len(cand.get(e, ())) for e in order)
    empties = sum(1 for e in order if not sel.get(e))
    scanned = sum(s["s23_scanned"] for s in stats)
    considered = sum(s["pairs_considered"] for s in stats)
    log("")
    log(f"record scans {scanned:,}, pairs considered {considered:,}")
    log(f"matching_results.tsv : {n_m:,} rows, {links:,} links, "
        f"{links / max(1, n_m):.2f}/entity, {empties:,} empty ({empties / max(1, n_m):.1%})")
    log(f"candidate_pairs.tsv  : {n_c:,} rows, {cands:,} candidates, "
        f"{cands / max(1, n_c):.2f}/entity")


if __name__ == "__main__":
    main()
