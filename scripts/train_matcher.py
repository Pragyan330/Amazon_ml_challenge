"""Train the learned matcher and select sets with it.

Reuses the scored candidates from ``scripts/validate.py`` so the expensive blocking pass is
not repeated. The model reranks those candidates; blocking and the prefilter are unchanged.

Three entity-disjoint splits: **train** fits the booster, **calib** drives early stopping and
the isotonic calibrator, **test** is the only split any reported number comes from. Fitting
the calibrator on the data used to report would make the probabilities look honest while
being fitted to that exact sample, and expected-F_0.5 consumes those probabilities directly.
Splits are on entities, never pairs, because one entity's candidates are not independent.

**Memory is bounded by construction.** An earlier version held every target ``Rec`` at once
and accumulated features as Python lists, which needed ~21.6 GB on the 441k-entity dataset
and crashed the machine. Now:

* feature rows are written straight into preallocated float32 arrays, sized from a counting
  pass, instead of lists of boxed Python floats (1,176 bytes per row against 156);
* target records are built in groups, so only a fraction of them exist at any moment, at the
  cost of one extra streaming pass per group.

Peak usage is roughly ``0.16 GB per million candidate pairs`` plus the Source-1 records plus
one group of targets - about 4 GB for the 8.1M-pair dataset.

    python scripts/train_matcher.py --scored val_scored_s5.pkl --cross-source
"""

import argparse
import gc
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber.blocking import idf_from_df
from ber.config import ARTIFACTS, EMBEDDINGS, TRAIN
from ber.dataio import read_source
from ber.evaluate import breakdown, macro_f_beta
from ber.pairfeatures import (FEATURE_NAMES, add_cross_source_features,
                              add_rank_features, pair_features, token_signature)
from ber.record import build
from ber.select import select_threshold
from ber.setselect import select_expected_f05

TRAIN_, CALIB_, TEST_ = 0, 1, 2


def log(msg):
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def free_gb():
    try:
        import ctypes

        class S(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = S(); s.dwLength = ctypes.sizeof(S)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.ullAvailPhys / 1e9
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="val_scored_s5.pkl")
    ap.add_argument("--embeddings", default="")
    ap.add_argument("--cross-source", action="store_true")
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--lr", type=float, default=0.04)
    ap.add_argument("--groups", type=int, default=10,
                    help="target records are built in this many groups; higher = less RAM, "
                         "one extra pass over Source 2/3 each")
    ap.add_argument("--threads", type=int, default=0,
                    help="0 = leave 4 cores free so the machine stays usable")
    ap.add_argument("--out-model", default="matcher_v2.pkl")
    args = ap.parse_args()

    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    threads = args.threads or max(1, (os.cpu_count() or 8) - 4)
    log(f"threads {threads}, free RAM {free_gb():.1f} GB")

    with open(os.path.join(ARTIFACTS, args.scored), "rb") as fh:
        data = pickle.load(fh)
    scored, truth, country_of = data["scored"], data["truth"], data["country"]
    n_pairs = sum(len(v) for v in scored.values())
    log(f"{len(truth):,} entities, {sum(len(v) for v in truth.values()):,} true links, "
        f"{n_pairs:,} candidate pairs")

    with open(os.path.join(ARTIFACTS, "corpus_stats.pkl"), "rb") as fh:
        df_name, df_addr, n_docs = pickle.load(fh)
    idf_name, default_idf = idf_from_df(df_name, n_docs)
    idf_addr, _ = idf_from_df(df_addr, n_docs)

    emb = None
    if args.embeddings:
        from ber.embeddings import load as load_emb
        emb = load_emb(EMBEDDINGS, args.embeddings)
        log(f"encoder: {emb.model if emb else 'NOT FOUND'}")

    names = FEATURE_NAMES if args.cross_source else [
        n for n in FEATURE_NAMES if not n.startswith("xs_")]
    n_feat = len(names)

    # --- entity-disjoint splits, and a counting pass so arrays can be preallocated ---
    ents = np.array(sorted(scored.keys()))
    rng = np.random.default_rng(42)
    rng.shuffle(ents)
    n = len(ents)
    split_of = {}
    for i, e in enumerate(ents):
        split_of[e] = TRAIN_ if i < 0.60 * n else (CALIB_ if i < 0.80 * n else TEST_)
    counts = [0, 0, 0]
    for e in ents:
        counts[split_of[e]] += len(scored[e])
    log(f"splits: train {counts[0]:,} / calib {counts[1]:,} / test {counts[2]:,} pairs")
    log(f"preallocating {sum(counts) * n_feat * 4 / 1e9:.2f} GB of float32")

    X = [np.empty((c, n_feat), dtype=np.float32) for c in counts]
    Y = [np.empty(c, dtype=np.int8) for c in counts]
    IDX = [[] for _ in range(3)]
    fill = [0, 0, 0]

    # --- both sides built per group, so neither is ever held whole ---
    # A Rec measures ~6 KB in practice, so all 441k Source-1 plus 3.9M target records would
    # be ~26 GB. Re-streaming costs one extra pass over each source per group and is the
    # difference between bounded memory and crashing the machine.
    t0 = time.time()
    s1_path = os.path.join(TRAIN, "train_source1.tsv")
    s2 = os.path.join(TRAIN, "train_source2.tsv")
    s3 = os.path.join(TRAIN, "train_source3.tsv")
    for g in range(args.groups):
        grp = [e for i, e in enumerate(ents) if i % args.groups == g]
        grp_set = set(grp)
        need = {t for e in grp for _, t in scored[e]}
        s1_rec = {}
        for eid, name, addr, ctry in read_source(s1_path):
            if eid in grp_set:
                s1_rec[eid] = build(name, addr, ctry, df_name, df_addr)
        t_rec = {}
        for path in (s2, s3):
            for eid, name, addr, ctry in read_source(path):
                if eid in need:
                    t_rec[eid] = build(name, addr, ctry, df_name, df_addr)
        for eid in grp:
            cands = scored[eid]
            r1 = s1_rec.get(eid)
            if not cands or r1 is None:
                continue
            rows, bases, tids, sigs, flags = [], [], [], [], []
            for base, tid in cands:
                r2 = t_rec.get(tid)
                if r2 is None:
                    continue
                cos = emb.similarity(eid, tid) if emb is not None else None
                is_s3 = tid[1] == "3"
                rows.append(pair_features(r1, r2, base, idf_name, idf_addr, default_idf,
                                          cos, is_s3))
                bases.append(base); tids.append(tid); flags.append(is_s3)
                if args.cross_source:
                    sigs.append(token_signature(r2))
            if not rows:
                continue
            if args.cross_source:
                add_cross_source_features(rows, sigs, flags)
            add_rank_features(rows, bases)
            s = split_of[eid]
            k = fill[s]
            m = len(rows)
            X[s][k:k + m] = rows
            ts = truth[eid]
            Y[s][k:k + m] = [1 if t in ts else 0 for t in tids]
            if s == TEST_:
                IDX[s].extend((eid, t) for t in tids)
            fill[s] = k + m
        del t_rec, s1_rec
        gc.collect()
        log(f"  group {g + 1}/{args.groups}: {len(grp):,} entities, "
            f"filled {sum(fill):,}/{n_pairs:,}, free {free_gb():.1f} GB")

    for s in range(3):
        X[s] = X[s][:fill[s]]
        Y[s] = Y[s][:fill[s]]
    gc.collect()
    log(f"features done in {time.time() - t0:.0f}s, free {free_gb():.1f} GB")
    log(f"positive rate: train {Y[TRAIN_].mean():.4%}, test {Y[TEST_].mean():.4%}")

    # --- train ---
    params = {"objective": "binary", "metric": ["binary_logloss", "auc"],
              "learning_rate": args.lr, "num_leaves": args.leaves,
              "min_data_in_leaf": 200, "feature_fraction": 0.9,
              "bagging_fraction": 0.8, "bagging_freq": 1,
              "max_bin": 127, "verbose": -1, "num_threads": threads, "seed": 42}
    dtrain = lgb.Dataset(X[TRAIN_], label=Y[TRAIN_], feature_name=names,
                         free_raw_data=False)
    dcalib = lgb.Dataset(X[CALIB_], label=Y[CALIB_], reference=dtrain, free_raw_data=False)
    t1 = time.time()
    gbm = lgb.train(params, dtrain, num_boost_round=args.rounds,
                    valid_sets=[dcalib], valid_names=["calib"],
                    callbacks=[lgb.early_stopping(80, verbose=False),
                               lgb.log_evaluation(250)])
    log(f"trained {gbm.best_iteration} rounds in {time.time() - t1:.0f}s "
        f"(calib auc {gbm.best_score['calib']['auc']:.5f}), free {free_gb():.1f} GB")
    del dtrain, dcalib
    gc.collect()

    log("top features by gain:")
    for nm, gain in sorted(zip(names, gbm.feature_importance("gain")),
                           key=lambda x: -x[1])[:15]:
        log(f"    {nm:18} {gain:14,.0f}")

    p_calib = gbm.predict(X[CALIB_], num_iteration=gbm.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_calib, Y[CALIB_])
    p_test = gbm.predict(X[TEST_], num_iteration=gbm.best_iteration)

    test_e = {e for e in ents if split_of[e] == TEST_}
    test_truth = {e: truth[e] for e in test_e}
    by_ent = {}
    for (eid, tid), p in zip(IDX[TEST_], p_test):
        by_ent.setdefault(eid, []).append((float(p), tid))
    for e in test_truth:
        by_ent.setdefault(e, [])

    log("")
    log("=" * 68)
    base_pred = {e: set(select_threshold(scored.get(e, []), 0.725)) for e in test_truth}
    log(f"RULE BASELINE (threshold 0.725)     macro F_0.5 = "
        f"{macro_f_beta(base_pred, test_truth):.4f}")
    best = max(((th, macro_f_beta({e: set(select_threshold(v, th))
                                   for e, v in by_ent.items()}, test_truth))
                for th in [round(0.02 * i, 2) for i in range(1, 50)]), key=lambda r: r[1])
    log(f"GBM + best fixed threshold ({best[0]:.2f})  macro F_0.5 = {best[1]:.4f}")
    ef = {e: set(select_expected_f05(v)) for e, v in by_ent.items()}
    log(f"GBM + expected-F0.5 (exact)         macro F_0.5 = "
        f"{macro_f_beta(ef, test_truth):.4f}")

    d = breakdown(ef, test_truth, group_of={e: country_of[e] for e in test_truth})
    log("")
    log("DIAGNOSTICS - GBM + exact expected-F0.5")
    log(f"  micro precision / recall : {d['micro_precision']:.4f} / {d['micro_recall']:.4f}")
    log(f"  predicted links          : {d['predicted_links']:,} (true {d['true_links']:,})")
    log(f"  singletons kept empty    : {d['singletons_kept_empty']:,}/{d['singletons']:,} = "
        f"{d['singletons_kept_empty'] / max(1, d['singletons']):.2%}")
    for g, (sc, cnt) in d["per_group"].items():
        log(f"  {g:<10}: {sc:.4f} (n={cnt:,})")
    orc = {e: {t for _, t in scored.get(e, [])} & test_truth[e] for e in test_truth}
    log(f"  oracle over candidates   : {macro_f_beta(orc, test_truth):.4f}")

    out = os.path.join(ARTIFACTS, args.out_model)
    with open(out, "wb") as fh:
        pickle.dump({"model": gbm.model_to_string(), "iso_x": iso.X_thresholds_,
                     "iso_y": iso.y_thresholds_, "features": names,
                     "cross_source": bool(args.cross_source),
                     "best_iteration": gbm.best_iteration}, fh)
    log(f"saved model to {out}")


if __name__ == "__main__":
    main()
