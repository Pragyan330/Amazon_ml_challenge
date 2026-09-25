"""Validation run for the rule-based baseline.

Holds out a deterministic subsample of training Source-1 entities, runs the full pipeline
for them against **all** training Source-2/3 records, then reports blocking recall and the
macro F_0.5 the rule achieves.

The subsample is on the Source-1 side only. Every sampled entity is still scored against the
complete 10.3M-record opposing pool, so precision is measured against the real distractor
density rather than an easier subset.

    python scripts/validate.py --sample-every 20
"""

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber.blocking import Blocker, build_corpus_stats, idf_from_df
from ber.dataio import read_ground_truth
from ber.evaluate import breakdown
from ber.pipeline import as_id_map, run_shard
from ber.scorer import Weights
from ber.select import sweep_threshold

from ber.config import ARTIFACTS as CACHE, TRAIN as DATA, ensure_dirs

ensure_dirs()


def log(msg):
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def keeper(sample_every):
    """Deterministic 1-in-N filter on the numeric part of an entity id.

    Deterministic rather than random so reruns and later stages see the same split.
    """
    def keep(eid):
        try:
            return int(eid.split("-", 1)[1]) % sample_every == 0
        except (IndexError, ValueError):
            return False
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-every", type=int, default=20,
                    help="keep 1 in N Source-1 entities for validation")
    ap.add_argument("--df-cap", type=int, default=60)
    ap.add_argument("--max-posting", type=int, default=200)
    ap.add_argument("--prefilter", type=float, default=0.34)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--countries", default="US,India")
    ap.add_argument("--embeddings", default=None,
                    help="basename under the embeddings dir, e.g. train_s1every20_names")
    ap.add_argument("--out", default=os.path.join(CACHE, "val_scored.pkl"))
    args = ap.parse_args()

    s1 = os.path.join(DATA, "train_source1.tsv")
    s23 = [os.path.join(DATA, "train_source2.tsv"), os.path.join(DATA, "train_source3.tsv")]

    # --- corpus statistics (cached: it is a full strided pass over 10.3M records) ---
    stats_path = os.path.join(CACHE, "corpus_stats.pkl")
    if os.path.exists(stats_path):
        log(f"loading cached corpus stats from {stats_path}")
        with open(stats_path, "rb") as fh:
            df_name, df_addr, n_docs = pickle.load(fh)
    else:
        log("building corpus statistics (1-in-10 sample of S2/S3)...")
        df_name, df_addr, n_docs = build_corpus_stats(s23, sample_every=10)
        os.makedirs(CACHE, exist_ok=True)
        with open(stats_path, "wb") as fh:
            pickle.dump((df_name, df_addr, n_docs), fh)
    log(f"corpus: {n_docs:,} sampled docs, {len(df_name):,} name tokens, "
        f"{len(df_addr):,} address tokens")

    idf_name, default_idf = idf_from_df(df_name, n_docs)
    idf_addr, _ = idf_from_df(df_addr, n_docs)

    keep = keeper(args.sample_every)
    truth = read_ground_truth(os.path.join(DATA, "train_ground_truth.tsv"), keep=keep)
    log(f"validation truth: {len(truth):,} S1 entities, "
        f"{sum(len(v) for v in truth.values()):,} true links, "
        f"{sum(1 for v in truth.values() if not v):,} singletons")

    weights = Weights()
    log(f"weights: {weights}")

    emb = None
    if args.embeddings:
        from ber.config import EMBEDDINGS
        from ber.embeddings import load
        emb = load(EMBEDDINGS, args.embeddings)
        if emb is None:
            log(f"WARNING: no embeddings found at {EMBEDDINGS}/{args.embeddings}.* "
                f"- running without the encoder")
        else:
            log(f"encoder: {emb.model} dim={emb.dim} vectors={emb.count:,}")

    scored_map = {}
    country_of = {}
    all_stats = {}
    for country in args.countries.split(","):
        blocker = Blocker(df_name, df_addr, df_cap=args.df_cap,
                          max_posting=args.max_posting)
        result = run_shard(country, s1, s23, blocker, idf_name, idf_addr, default_idf,
                           weights, prefilter=args.prefilter, topk=args.topk,
                           s1_keep=keep, log=log, emb=emb)
        shard_map = as_id_map(result)
        scored_map.update(shard_map)
        for eid in shard_map:
            country_of[eid] = country
        all_stats[country] = result.stats
        del blocker, result

    # --- blocking quality -------------------------------------------------------------
    truth = {k: v for k, v in truth.items() if k in scored_map}
    total_links = sum(len(v) for v in truth.values())
    recovered = sum(1 for eid, ts in truth.items()
                    for t in ts if any(t == c[1] for c in scored_map[eid]))
    log("")
    log("=" * 78)
    log(f"CANDIDATE SET (after prefilter {args.prefilter}, topk {args.topk})")
    log(f"  entities            : {len(truth):,}")
    log(f"  candidates kept/S1  : "
        f"{sum(len(v) for v in scored_map.values()) / max(1, len(scored_map)):.2f}")
    log(f"  pair recall         : {recovered:,}/{total_links:,} = "
        f"{recovered / max(1, total_links):.3%}   <-- recall ceiling of the rule")

    # --- threshold sweep --------------------------------------------------------------
    thresholds = [round(0.34 + 0.02 * i, 2) for i in range(29)]
    results, best = sweep_threshold(scored_map, truth, thresholds)
    log("")
    log("THRESHOLD SWEEP (macro F_0.5)")
    for th, sc in results:
        bar = "#" * int(sc * 60)
        log(f"  {th:.2f}  {sc:.4f}  {bar}")
    log("")
    log(f"BEST threshold {best[0]:.2f} -> macro F_0.5 = {best[1]:.4f}")

    # --- diagnostics at the best threshold --------------------------------------------
    from ber.select import select_threshold
    pred = {eid: set(select_threshold(sc, best[0])) for eid, sc in scored_map.items()}
    d = breakdown(pred, truth, group_of=country_of)
    log("")
    log("DIAGNOSTICS at best threshold")
    log(f"  macro F_0.5        : {d['macro_f05']:.4f}")
    log(f"  micro precision    : {d['micro_precision']:.4f}")
    log(f"  micro recall       : {d['micro_recall']:.4f}")
    log(f"  predicted links    : {d['predicted_links']:,} (true {d['true_links']:,})")
    log(f"  singletons kept empty: {d['singletons_kept_empty']:,}/{d['singletons']:,} = "
        f"{d['singletons_kept_empty'] / max(1, d['singletons']):.2%}")
    for g, (sc, n) in d["per_group"].items():
        log(f"  {g:<18}: {sc:.4f}  (n={n:,})")
    log("")
    for c, st in all_stats.items():
        log(f"  [{c}] {st['s23_scanned']:,} scanned, "
            f"{st['considered_per_s1']:.1f} considered/S1, {st['seconds']:.0f}s")

    os.makedirs(CACHE, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"scored": scored_map, "truth": truth, "country": country_of,
                     "best_threshold": best[0], "sweep": results,
                     "weights": repr(weights), "args": vars(args)}, fh)
    log(f"saved scored candidates to {args.out}")


if __name__ == "__main__":
    main()
