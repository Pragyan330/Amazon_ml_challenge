import os
import sys
import pickle
import time
import numpy as np

import lightgbm as lgb
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ber.dataio import read_source
from ber.record import build
from ber.blocking import idf_from_df
from ber.features import jaccard, dice, containment, weighted_jaccard, numeric_agreement
from ber.evaluate import breakdown
from ber.select import sweep_threshold, select_threshold

WORK = os.environ.get("BER_WORK", r"D:\ml_challenge")
CACHE = os.path.join(WORK, "artifacts")

def extract_features(r1, r2, base_score, idf_name, idf_addr, default_idf):
    # Name channel
    if r2.latin:
        n_jac = jaccard(r1.nc, r2.nc)
        n_dice = dice(r1.nc, r2.nc)
        n_cont = containment(r1.nd, r2.nd)
        n_wjac = weighted_jaccard(r1.nc, r2.nc, idf_name, default_idf)
        n_ng_dice = dice(r1.name_grams(), r2.name_grams())
    else:
        n_jac = n_dice = n_cont = n_wjac = n_ng_dice = 0.0

    # Address channel
    if r1.at and r2.at:
        a_jac = jaccard(r1.at, r2.at)
        a_dice = dice(r1.at, r2.at)
        a_wjac = weighted_jaccard(r1.at, r2.at, idf_addr, default_idf)
        a_ng_dice = dice(r1.addr_grams(), r2.addr_grams())
    else:
        a_jac = a_dice = a_wjac = a_ng_dice = 0.0

    # Numeric (street/suite)
    num_score, has_num = numeric_agreement(r1.an, r2.an)
    if not has_num:
        num_state = 0.0
    elif num_score == 1.0:
        num_state = 1.0
    elif num_score > 0.0:
        num_state = 2.0
    else:
        num_state = 3.0

    target_is_latin = 1.0 if r2.latin else 0.0
    
    return [
        n_jac, n_dice, n_cont, n_wjac, n_ng_dice,
        a_jac, a_dice, a_wjac, a_ng_dice,
        num_state, num_score,
        target_is_latin,
        base_score
    ]

def main():
    print("Loading scored candidates...")
    with open(os.path.join(CACHE, "val_scored.pkl"), "rb") as f:
        data = pickle.load(f)
    scored = data["scored"]
    truth = data["truth"]
    country_of = data["country"]

    print("Loading corpus stats...")
    with open(os.path.join(CACHE, "corpus_stats.pkl"), "rb") as f:
        df_name, df_addr, n_docs = pickle.load(f)

    idf_name, default_idf = idf_from_df(df_name, n_docs)
    idf_addr, _ = idf_from_df(df_addr, n_docs)

    # We need the full TSVs loaded into dictionaries to build `Rec`s on demand
    print("Loading TSVs...")
    from ber.config import TRAIN
    s1_dict, s23_dict = {}, {}
    for eid, name, addr, ctry in read_source(os.path.join(TRAIN, "train_source1.tsv")):
        if eid in scored:
            s1_dict[eid] = build(name, addr, ctry, df_name, df_addr)
            
    for path in [os.path.join(TRAIN, "train_source2.tsv"), os.path.join(TRAIN, "train_source3.tsv")]:
        for eid, name, addr, ctry in read_source(path):
            s23_dict[eid] = (name, addr, ctry)

    # Note: s23 records are built lazily to save RAM since we don't need all 10M
    built_s23 = {}

    print("Building features...")
    # Split entities 80/20 train/val
    entities = sorted(list(scored.keys()))
    np.random.seed(42)
    np.random.shuffle(entities)
    split_idx = int(len(entities) * 0.8)
    train_eids = set(entities[:split_idx])
    val_eids = set(entities[split_idx:])

    X_train, y_train, group_train = [], [], []
    X_val, y_val, group_val = [], [], []

    val_scored_gbm = {}
    
    t0 = time.time()
    for eid in entities:
        cands = scored[eid]
        r1 = s1_dict[eid]
        true_set = truth[eid]
        
        cands_gbm = []
        group_size = 0
        for base_score, cid in cands:
            if cid not in built_s23:
                name, addr, ctry = s23_dict[cid]
                built_s23[cid] = build(name, addr, ctry, df_name, df_addr)
            r2 = built_s23[cid]
            
            feat = extract_features(r1, r2, base_score, idf_name, idf_addr, default_idf)
            label = 1 if cid in true_set else 0
            
            if eid in train_eids:
                X_train.append(feat)
                y_train.append(label)
                group_size += 1
            else:
                X_val.append(feat)
                y_val.append(label)
                # Keep track for evaluation
                cands_gbm.append((feat, cid))

        if eid in train_eids and group_size > 0:
            group_train.append(group_size)
        if eid in val_eids:
            val_scored_gbm[eid] = cands_gbm

    print(f"Extracted features in {time.time()-t0:.1f}s")
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    print(f"Train size: {X_train.shape[0]} pairs, Val size: {X_val.shape[0]} pairs")
    val_truth = {eid: truth[eid] for eid in val_eids}
    
    print("Saving features to artifacts/gbm_features.pkl...")
    with open(os.path.join(CACHE, "gbm_features.pkl"), "wb") as f:
        pickle.dump({
            "X_train": X_train,
            "y_train": y_train,
            "group_train": group_train,
            "X_val": X_val,
            "y_val": y_val,
            "val_scored_gbm": val_scored_gbm,
            "val_truth": val_truth,
            "val_eids": val_eids
        }, f)

    print("Training LightGBM...")
    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'random_state': 42
    }
    gbm = lgb.train(params, train_data, num_boost_round=100)

    # Feature importances
    importance = gbm.feature_importance(importance_type='gain')
    features = ['n_jac', 'n_dice', 'n_cont', 'n_wjac', 'n_ng_dice',
                'a_jac', 'a_dice', 'a_wjac', 'a_ng_dice',
                'num_state', 'num_score',
                'target_is_latin', 'base_score']
    
    print("\nFeature Importances (Gain):")
    for name, imp in sorted(zip(features, importance), key=lambda x: x[1], reverse=True):
        print(f"  {name:15s}: {imp:.1f}")

    print("\nEvaluating on validation set...")
    val_pred_map = {}
    for eid, cands in val_scored_gbm.items():
        if not cands:
            val_pred_map[eid] = []
            continue
        feats = np.array([c[0] for c in cands])
        preds = gbm.predict(feats)
        val_pred_map[eid] = [(float(p), c[1]) for p, c in zip(preds, cands)]

    # Threshold sweep
    thresholds = [round(0.1 + 0.02 * i, 2) for i in range(41)]
    results, best = sweep_threshold(val_pred_map, val_truth, thresholds)
    
    print("\nTHRESHOLD SWEEP (macro F_0.5)")
    for th, sc in results:
        bar = "#" * int(sc * 60)
        print(f"  {th:.2f}  {sc:.4f}  {bar}")

    print(f"\nBEST threshold {best[0]:.2f} -> macro F_0.5 = {best[1]:.4f}")

    from ber.evaluate import breakdown
    pred = {eid: set(select_threshold(sc, best[0])) for eid, sc in val_pred_map.items()}
    val_country = {eid: country_of[eid] for eid in val_eids}
    d = breakdown(pred, val_truth, group_of=val_country)
    
    print("\nDIAGNOSTICS at best threshold")
    print(f"  macro F_0.5        : {d['macro_f05']:.4f}")
    print(f"  micro precision    : {d['micro_precision']:.4f}")
    print(f"  micro recall       : {d['micro_recall']:.4f}")
    print(f"  predicted links    : {d['predicted_links']:,} (true {d['true_links']:,})")
    print(f"  singletons kept empty: {d['singletons_kept_empty']:,}/{d['singletons']:,} = {d['singletons_kept_empty'] / max(1, d['singletons']):.2%}")
    for g, (sc, n) in d["per_group"].items():
        print(f"  {g:<18}: {sc:.4f}  (n={n:,})")


if __name__ == "__main__":
    main()
