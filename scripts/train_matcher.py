"""Train the learned matcher on the rule pipeline's candidate set, and select sets with it.

Reuses the scored candidates from ``scripts/validate.py`` so the expensive blocking pass is
not repeated. The model reranks those candidates; blocking and the prefilter are unchanged.

Three entity-disjoint splits, which matters more than it might look:

* **train**  - fits the booster
* **calib**  - early stopping *and* the isotonic calibrator
* **test**   - the only split any reported number comes from

Fitting the calibrator on the same data used to report would make the probabilities look
honest while being fitted to that exact sample, and expected-F_0.5 selection consumes those
probabilities directly - a calibrator fitted in-sample would flatter every downstream number.
The split is on entities, never on pairs, because candidates of one entity are not
independent of each other.

    D:\\ml_challenge\\venv\\Scripts\\python.exe scripts/train_matcher.py \\
        --scored val_scored_emb.pkl --embeddings train_s1every20_names
"""

import argparse
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
from ber.pairfeatures import FEATURE_NAMES, add_rank_features, pair_features
from ber.record import build
from ber.select import select_threshold
from ber.setselect import expected_f05_ratio_approx, select_expected_f05


def log(msg):
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="val_scored_emb.pkl")
    ap.add_argument("--embeddings", default="train_s1every20_names")
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    with open(os.path.join(ARTIFACTS, args.scored), "rb") as fh:
        data = pickle.load(fh)
    scored, truth, country_of = data["scored"], data["truth"], data["country"]
    log(f"{len(truth):,} entities, {sum(len(v) for v in truth.values()):,} true links, "
        f"{sum(len(v) for v in scored.values()):,} candidate pairs")

    with open(os.path.join(ARTIFACTS, "corpus_stats.pkl"), "rb") as fh:
        df_name, df_addr, n_docs = pickle.load(fh)
    idf_name, default_idf = idf_from_df(df_name, n_docs)
    idf_addr, _ = idf_from_df(df_addr, n_docs)

    emb = None
    if args.embeddings:
        from ber.embeddings import load as load_emb
        emb = load_emb(EMBEDDINGS, args.embeddings)
        log(f"encoder: {emb.model if emb else 'NOT FOUND'}")

    # --- pull only the records we actually need, rather than all 10.3M ---
    need_t = set()
    for v in scored.values():
        for _, tid in v:
            need_t.add(tid)
    log(f"loading text for {len(truth):,} S1 and {len(need_t):,} target records ...")
    s1_rec = {}
    for eid, name, addr, ctry in read_source(os.path.join(TRAIN, "train_source1.tsv")):
        if eid in scored:
            s1_rec[eid] = build(name, addr, ctry, df_name, df_addr)
    t_rec = {}
    for src in (2, 3):
        path = os.path.join(TRAIN, f"train_source{src}.tsv")
        for eid, name, addr, ctry in read_source(path):
            if eid in need_t:
                t_rec[eid] = build(name, addr, ctry, df_name, df_addr)
    log(f"built {len(s1_rec):,} S1 and {len(t_rec):,} target records")

    # --- entity-disjoint splits ---
    ents = np.array(sorted(scored.keys()))
    rng = np.random.default_rng(42)
    rng.shuffle(ents)
    n = len(ents)
    train_e = set(ents[:int(0.60 * n)])
    calib_e = set(ents[int(0.60 * n):int(0.80 * n)])
    test_e = set(ents[int(0.80 * n):])
    log(f"splits: train {len(train_e):,} / calib {len(calib_e):,} / test {len(test_e):,}")

    # --- features ---
    t0 = time.time()
    X = {"train": [], "calib": [], "test": []}
    y = {"train": [], "calib": [], "test": []}
    index = {"train": [], "calib": [], "test": []}  # (eid, tid) per row
    for eid in ents:
        cands = scored[eid]
        if not cands:
            continue
        r1 = s1_rec.get(eid)
        if r1 is None:
            continue
        split = "train" if eid in train_e else ("calib" if eid in calib_e else "test")
        rows, bases, tids = [], [], []
        for base, tid in cands:
            r2 = t_rec.get(tid)
            if r2 is None:
                continue
            cos = emb.similarity(eid, tid) if emb is not None else None
            rows.append(pair_features(r1, r2, base, idf_name, idf_addr, default_idf,
                                      cos, tid.startswith("S3-")))
            bases.append(base)
            tids.append(tid)
        if not rows:
            continue
        add_rank_features(rows, bases)
        ts = truth[eid]
        X[split].extend(rows)
        y[split].extend(1 if t in ts else 0 for t in tids)
        index[split].extend((eid, t) for t in tids)
    for k in X:
        X[k] = np.asarray(X[k], dtype=np.float32)
        y[k] = np.asarray(y[k], dtype=np.int8)
    log(f"features built in {time.time() - t0:.0f}s: "
        + ", ".join(f"{k}={X[k].shape}" for k in X))
    log(f"positive rate: train {y['train'].mean():.4%}, test {y['test'].mean():.4%}")

    # --- train ---
    params = {"objective": "binary", "metric": ["binary_logloss", "auc"],
              "learning_rate": args.lr, "num_leaves": args.leaves,
              "min_data_in_leaf": 100, "feature_fraction": 0.9,
              "bagging_fraction": 0.8, "bagging_freq": 1,
              "verbose": -1, "num_threads": os.cpu_count(), "seed": 42}
    dtrain = lgb.Dataset(X["train"], label=y["train"], feature_name=FEATURE_NAMES)
    dcalib = lgb.Dataset(X["calib"], label=y["calib"], reference=dtrain)
    t0 = time.time()
    gbm = lgb.train(params, dtrain, num_boost_round=args.rounds,
                    valid_sets=[dcalib], valid_names=["calib"],
                    callbacks=[lgb.early_stopping(60, verbose=False),
                               lgb.log_evaluation(200)])
    log(f"trained {gbm.best_iteration} rounds in {time.time() - t0:.0f}s "
        f"(calib auc {gbm.best_score['calib']['auc']:.5f})")

    imp = sorted(zip(FEATURE_NAMES, gbm.feature_importance("gain")),
                 key=lambda x: -x[1])
    log("top features by gain:")
    for nm, g in imp[:14]:
        log(f"    {nm:18} {g:12,.0f}")

    # --- calibrate on the calib split, never on test ---
    p_calib = gbm.predict(X["calib"], num_iteration=gbm.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_calib, y["calib"])
    p_test_raw = gbm.predict(X["test"], num_iteration=gbm.best_iteration)
    p_test = iso.predict(p_test_raw)

    def reliability(p, yy, bins=8):
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        out = []
        for i in range(bins):
            m = (p >= edges[i]) & (p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1])
            if m.sum() > 50:
                out.append((p[m].mean(), yy[m].mean(), int(m.sum())))
        return out

    log("calibration on TEST (predicted vs actual):")
    log("    raw                        isotonic")
    r_raw = reliability(p_test_raw, y["test"])
    r_cal = reliability(p_test, y["test"])
    for (pr, ar, nr), (pc, ac, nc) in zip(r_raw, r_cal):
        log(f"    {pr:.3f} -> {ar:.3f} (n={nr:>6,})   {pc:.3f} -> {ac:.3f}")
    ece_raw = sum(abs(a - p) * c for p, a, c in r_raw) / max(1, sum(c for _, _, c in r_raw))
    ece_cal = sum(abs(a - p) * c for p, a, c in r_cal) / max(1, sum(c for _, _, c in r_cal))
    log(f"    weighted calibration error: raw {ece_raw:.4f} -> isotonic {ece_cal:.4f}")

    # --- evaluate on test only ---
    test_truth = {e: truth[e] for e in test_e}
    by_ent_raw, by_ent_cal = {}, {}
    for (eid, tid), pr, pc in zip(index["test"], p_test_raw, p_test):
        by_ent_raw.setdefault(eid, []).append((float(pr), tid))
        by_ent_cal.setdefault(eid, []).append((float(pc), tid))
    for e in test_truth:
        by_ent_raw.setdefault(e, [])
        by_ent_cal.setdefault(e, [])

    log("")
    log("=" * 70)
    base_pred = {e: set(select_threshold(scored.get(e, []), 0.725)) for e in test_truth}
    log(f"RULE BASELINE (threshold 0.725)        macro F_0.5 = "
        f"{macro_f_beta(base_pred, test_truth):.4f}")

    best = (0.0, 0.0)
    for th in [round(0.02 * i, 2) for i in range(1, 50)]:
        f = macro_f_beta({e: set(select_threshold(v, th)) for e, v in by_ent_cal.items()},
                         test_truth)
        if f > best[1]:
            best = (th, f)
    log(f"GBM + best fixed threshold ({best[0]:.2f})   macro F_0.5 = {best[1]:.4f}")

    t0 = time.time()
    ef_pred = {e: set(select_expected_f05(v)) for e, v in by_ent_cal.items()}
    ef_f = macro_f_beta(ef_pred, test_truth)
    ef_time = time.time() - t0
    log(f"GBM + expected-F0.5 (exact)            macro F_0.5 = {ef_f:.4f}")

    ef_raw = macro_f_beta({e: set(select_expected_f05(v)) for e, v in by_ent_raw.items()},
                          test_truth)
    log(f"GBM + expected-F0.5 on UNCALIBRATED    macro F_0.5 = {ef_raw:.4f}")

    def sel_approx(v):
        if not v:
            return []
        r = sorted(v, key=lambda x: -x[0])[:24]
        return [t for _, t in r[:expected_f05_ratio_approx([s for s, _ in r])]]
    ef_ap = macro_f_beta({e: set(sel_approx(v)) for e, v in by_ent_cal.items()}, test_truth)
    log(f"GBM + expected-F0.5 ratio-approximation macro F_0.5 = {ef_ap:.4f}")
    log(f"    (selection took {ef_time:.1f}s for {len(by_ent_cal):,} entities)")

    d = breakdown(ef_pred, test_truth,
                  group_of={e: country_of[e] for e in test_truth})
    log("")
    log("DIAGNOSTICS - GBM + exact expected-F0.5")
    log(f"  micro precision / recall : {d['micro_precision']:.4f} / {d['micro_recall']:.4f}")
    log(f"  predicted links          : {d['predicted_links']:,} (true {d['true_links']:,})")
    log(f"  singletons kept empty    : {d['singletons_kept_empty']:,}/{d['singletons']:,} = "
        f"{d['singletons_kept_empty'] / max(1, d['singletons']):.2%}")
    for g, (sc, cnt) in d["per_group"].items():
        log(f"  {g:<10}: {sc:.4f} (n={cnt:,})")

    orc = {e: set(t for _, t in scored.get(e, [])) & test_truth[e] for e in test_truth}
    log(f"  oracle over candidates   : {macro_f_beta(orc, test_truth):.4f}")

    out = os.path.join(ARTIFACTS, "matcher.pkl")
    with open(out, "wb") as fh:
        pickle.dump({"model": gbm.model_to_string(), "iso_x": iso.X_thresholds_,
                     "iso_y": iso.y_thresholds_, "features": FEATURE_NAMES,
                     "best_iteration": gbm.best_iteration}, fh)
    log(f"saved model to {out}")


if __name__ == "__main__":
    main()
