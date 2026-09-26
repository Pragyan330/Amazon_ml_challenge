"""End-to-End Test Pipeline with Expected-F_0.5 Selection.

Usage:
  python scripts/test_pipeline.py --test-slice 50000  # For quick sanity check
  python scripts/test_pipeline.py                      # Full 1.73M run
"""
import argparse
import os
import pickle
import argparse
import os
import pickle
import sys
import time
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ber.blocking import build_corpus_stats, idf_from_df
from ber.config import ARTIFACTS, OUTPUT, TEST, WORK, ensure_dirs, test_paths, DATA
from ber.dataio import CANDIDATE_HEADER, MATCHING_HEADER, country_counts, write_id_lists, read_source
from ber.parallel import run_parallel
from ber.record import build
from ber.features import jaccard, dice, containment, weighted_jaccard, numeric_agreement

def expected_f05_selection(probs):
    """
    Given a list of probabilities for candidates of a single S1 entity,
    returns the optimal number of candidates to select to maximize expected F0.5.
    """
    import numpy as np
    
    if not probs:
        return 0
    
    probs = np.sort(probs)[::-1]
    k = len(probs)
    
    E_T = np.sum(probs)
    best_n = 0
    best_f = np.prod(1.0 - probs)
    
    S_n = 0.0
    for n in range(1, k + 1):
        S_n += probs[n - 1]
        expected_f = (1.25 * S_n) / (0.25 * E_T + n)
        if expected_f > best_f:
            best_f = expected_f
            best_n = n
            
    return best_n

def log(msg):
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)

# Sequential feature extraction inside main

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

# process_entity removed

def get_train_gbm():
    import numpy as np
    import lightgbm as lgb
    
    log("Loading features and training LightGBM model...")
    with open(os.path.join(ARTIFACTS, "gbm_features.pkl"), "rb") as f:
        data = pickle.load(f)
        
    # Combine train and val for final inference model
    X_train = np.vstack([data["X_train"], data["X_val"]])
    y_train = np.concatenate([data["y_train"], data["y_val"]])
    
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'random_state': 42
    }
    gbm = lgb.train(params, train_data, num_boost_round=100)
    return gbm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-slice", type=int, default=0, help="Test on N entities only")
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count - 1")
    ap.add_argument("--prefilter", type=float, default=0.34)
    ap.add_argument("--topk", type=int, default=40)
    args = ap.parse_args()

    ensure_dirs()
    os.makedirs(os.path.join(WORK, "tmp"), exist_ok=True)
    t_start = time.time()
    
    # Check if we are running in the challenge environment vs local dataset
    test_dir = os.environ.get("BER_TEST") or os.path.join(DATA, "test")
    if not os.path.exists(test_dir):
        # Fallback to student_resource/dataset/test
        test_dir = os.path.join(WORK, "student_resource", "dataset", "test")
        
    s1 = os.path.join(test_dir, "test_source1.tsv")
    s23 = [os.path.join(test_dir, "test_source2.tsv"), os.path.join(test_dir, "test_source3.tsv")]

    stats_path = os.path.join(ARTIFACTS, "corpus_stats_train.pkl")
    # Actually wait! The training corpus stats is named corpus_stats.pkl in cache
    # Let's use the one generated during train_gbm.py
    train_stats_path = os.path.join(ARTIFACTS, "corpus_stats.pkl")
    log(f"Loading train corpus stats from {train_stats_path} for TF-IDF features...")
    with open(train_stats_path, "rb") as fh:
        df_name_train, df_addr_train, n_docs_train = pickle.load(fh)
    
    idf_name_train, default_idf_train = idf_from_df(df_name_train, n_docs_train)
    idf_addr_train, _ = idf_from_df(df_addr_train, n_docs_train)

    # Rebuild Test Corpus Stats for Blocking phase (as predict.py did)
    test_stats_path = os.path.join(ARTIFACTS, "corpus_stats_test.pkl")
    if os.path.exists(test_stats_path):
        log(f"Using cached test corpus stats for blocking: {test_stats_path}")
    else:
        log("Building test corpus statistics for blocking (1-in-10 sample)...")
        df_name_test, df_addr_test, n_docs_test = build_corpus_stats(s23, sample_every=10)
        with open(test_stats_path, "wb") as fh:
            pickle.dump((df_name_test, df_addr_test, n_docs_test), fh)
            
    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)
    
    # Read entities
    log("Reading test_source1.tsv...")
    s1_dict = {}
    order = []
    with open(s1, encoding="utf-8") as fh:
        fh.readline()
        for i, line in enumerate(fh):
            if args.test_slice > 0 and i >= args.test_slice:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                eid, name, addr, ctry = parts[0], parts[1], parts[2], parts[3]
                s1_dict[eid] = (name, addr, ctry)
                order.append(eid)
                
    # To limit blocking time, we can create a sliced version of test_source1.tsv
    if args.test_slice > 0:
        tmp_dir = os.path.join(WORK, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        slice_s1 = os.path.join(tmp_dir, "slice_test_source1.tsv")
        with open(slice_s1, "w", encoding="utf-8") as out, open(s1, "r", encoding="utf-8") as inf:
            out.write(inf.readline())
            for i, line in enumerate(inf):
                if i >= args.test_slice:
                    break
                out.write(line)
        s1 = slice_s1
        log(f"Testing on slice of {args.test_slice} S1 entities")

    counts = country_counts(s1)
    log(f"test Source 1 by country: {counts}")
    countries = sorted(counts, key=lambda c: -counts[c])
    
    log("Running blocking (candidate generation)...")
    scored, _, shard_stats = run_parallel(
        countries, s1, s23, test_stats_path, os.path.join(WORK, "tmp"),
        workers=workers, chunks_per_country=workers,
        prefilter=args.prefilter, topk=args.topk, df_cap=60,
        max_posting=200, log=log)
        
    total_scanned = sum(s["s23_scanned"] for s in shard_stats)
    log(f"{total_scanned:,} record-scans in blocking")
    
    # Check France specifically
    fr_cands = [len(scored.get(eid, [])) for eid in order if s1_dict[eid][2] == "France"]
    if fr_cands:
        avg_fr = sum(fr_cands) / len(fr_cands)
        log(f"FRANCE CHECK: {len(fr_cands)} entities. Avg candidates = {avg_fr:.2f}")
    else:
        log("FRANCE CHECK: No France entities in this slice.")

    log("Loading full S2 and S3 for feature extraction...")
    s23_dict = {}
    for path in s23:
        for eid, name, addr, ctry in read_source(path):
            s23_dict[eid] = (name, addr, ctry)

    gbm = get_train_gbm()

    log("Extracting features and scoring with GBM + Expected-F0.5...")
    final_matching = {}
    final_candidates = {}
    
    log(f"Length of scored dict: {len(scored)}")
    log(f"Number of keys in scored dict intersecting with order: {len(set(scored.keys()).intersection(set(order)))}")
    log(f"Sample key from scored: {list(scored.keys())[0] if scored else 'None'}")
    log(f"Sample key from order: {order[0] if order else 'None'}")
    
    import numpy as np
    
    for eid in order:
        cands = scored.get(eid, [])
        if not cands:
            final_matching[eid] = set()
            final_candidates[eid] = []
            continue
            
        s1_tuple = s1_dict[eid]
        r1 = build(s1_tuple[0], s1_tuple[1], s1_tuple[2], df_name_train, df_addr_train)
        
        feats = []
        c_ids = []
        for base_score, cid in cands:
            c_ids.append(cid)
            name, addr, ctry = s23_dict[cid]
            r2 = build(name, addr, ctry, df_name_train, df_addr_train)
            f = extract_features(r1, r2, base_score, idf_name_train, idf_addr_train, default_idf_train)
            feats.append(f)
            
        probs = gbm.predict(np.array(feats))
        best_n = expected_f05_selection(list(probs))
        
        scored_combined = sorted(zip(probs, c_ids), key=lambda x: x[0], reverse=True)
        selected = set(cid for prob, cid in scored_combined[:best_n])
        
        final_matching[eid] = selected
        final_candidates[eid] = c_ids

    # Write outputs
    log("Writing output TSVs...")
    
    def match_rows():
        for eid in order:
            yield eid, final_matching.get(eid, set())
            
    def cand_rows():
        for eid in order:
            yield eid, final_candidates.get(eid, [])
            
    n_m = write_id_lists(os.path.join(OUTPUT, "matching_results.tsv"), match_rows(), MATCHING_HEADER)
    n_c = write_id_lists(os.path.join(OUTPUT, "candidate_pairs.tsv"), cand_rows(), CANDIDATE_HEADER)

    links = sum(len(v) for v in final_matching.values())
    cands = sum(len(v) for v in final_candidates.values())
    empties = sum(1 for v in final_matching.values() if not v)
    
    log("")
    log(f"Total time taken       : {(time.time() - t_start)/60:.2f} mins")
    log(f"matching_results.tsv   : {n_m:,} rows, {links:,} links, {links / max(1, n_m):.2f}/entity, {empties:,} empty ({empties / max(1, n_m):.2%} singleton rate)")
    log(f"candidate_pairs.tsv    : {n_c:,} rows, {cands:,} candidates, {cands / max(1, n_c):.2f}/entity")
    
if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
