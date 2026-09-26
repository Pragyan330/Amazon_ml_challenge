"""Worker that runs the complete chain for one Source-1 shard.

Blocking, rule score, feature extraction, LightGBM and expected-F_0.5 selection all happen
inside the worker, which then returns only two id lists per entity. The alternative - ship
candidates to the parent and score them afterwards - would need ``Rec`` objects for up to
10M Source-2/3 records at once, which does not fit in 23 GB.

Two different IDF tables are deliberately in play:

* **blocking** uses statistics from the *test* sources, because the keys have to be rare
  within the pool actually being searched, and the test set carries 23% more Source-2/3
  records per Source-1 entity than training does;
* **model features** use statistics from *training*, because that is what the model was
  fitted against, and feeding it differently-scaled inputs at inference would silently shift
  every feature.
"""

import multiprocessing as mp
import os
import pickle
import time


def _run_one(task):
    from .blocking import Blocker, idf_from_df
    from .embeddings import load as load_emb
    from .pairfeatures import add_rank_features, pair_features
    from .parallel import chunk_keeper
    from .pipeline import run_shard
    from .scorer import Weights
    from .setselect import select_expected_f05

    (country, chunk, n_chunks, s1_path, s23_paths, test_stats, train_stats, matcher_path,
     prefilter, topk, df_cap, max_posting, emb_dir, emb_name, out_dir) = task

    import numpy as np
    import lightgbm as lgb

    t0 = time.time()
    with open(test_stats, "rb") as fh:
        df_name_b, df_addr_b, n_docs_b = pickle.load(fh)
    with open(train_stats, "rb") as fh:
        df_name_f, df_addr_f, n_docs_f = pickle.load(fh)
    idf_name_f, default_f = idf_from_df(df_name_f, n_docs_f)
    idf_addr_f, _ = idf_from_df(df_addr_f, n_docs_f)
    idf_name_b, default_b = idf_from_df(df_name_b, n_docs_b)
    idf_addr_b, _ = idf_from_df(df_addr_b, n_docs_b)

    with open(matcher_path, "rb") as fh:
        m = pickle.load(fh)
    booster = lgb.Booster(model_str=m["model"])
    best_iter = m.get("best_iteration") or 0

    emb = load_emb(emb_dir, emb_name) if emb_name else None
    blocker = Blocker(df_name_b, df_addr_b, df_cap=df_cap, max_posting=max_posting)

    def feature_fn(r1, r2, score, cos, is_s3):
        return pair_features(r1, r2, score, idf_name_f, idf_addr_f, default_f, cos, is_s3)

    # The rule score is itself a model feature - by a wide margin the most important one -
    # so it must be computed with the *training* IDF the model was fitted against. Only the
    # Blocker uses test statistics, because key rarity has to be judged against the pool
    # actually being searched. Passing test IDF here instead would shift the single feature
    # the model leans on hardest.
    result = run_shard(country, s1_path, s23_paths, blocker, idf_name_f, idf_addr_f,
                       default_f, Weights(), prefilter=prefilter, topk=topk,
                       s1_keep=chunk_keeper(chunk, n_chunks, 1), emb=emb,
                       feature_fn=feature_fn)

    sel_out, cand_out = {}, {}
    rows, owners = [], []
    for slot, bucket in result.candidates.items():
        eid = result.s1_ids[slot]
        cand_out[eid] = [t for _, t, _ in bucket]
        feats = [list(r) for _, _, r in bucket]
        add_rank_features(feats, [s for s, _, _ in bucket])
        start = len(rows)
        rows.extend(feats)
        owners.append((eid, start, len(feats)))
    for eid in result.s1_ids:
        cand_out.setdefault(eid, [])
        sel_out.setdefault(eid, [])

    if rows:
        X = np.asarray(rows, dtype=np.float32)
        probs = booster.predict(X, num_iteration=best_iter or None)
        for eid, start, n in owners:
            ids = cand_out[eid]
            sel_out[eid] = select_expected_f05(
                [(float(probs[start + i]), ids[i]) for i in range(n)])

    path = os.path.join(out_dir, f"sub_{country}_{chunk:03d}.pkl")
    with open(path, "wb") as fh:
        pickle.dump((sel_out, cand_out), fh, protocol=pickle.HIGHEST_PROTOCOL)
    st = dict(result.stats)
    st.update({"country": country, "chunk": chunk, "path": path, "wall": time.time() - t0,
               "entities": len(cand_out),
               "selected": sum(len(v) for v in sel_out.values())})
    return st


def run_submission(countries, s1_path, s23_paths, test_stats, train_stats_path,
                   matcher_path, out_dir, workers, chunks_per_country, prefilter, topk,
                   df_cap, max_posting, emb_dir, emb_name, log=None):
    os.makedirs(out_dir, exist_ok=True)
    tasks = [(c, k, chunks_per_country, s1_path, s23_paths, test_stats, train_stats_path,
              matcher_path, prefilter, topk, df_cap, max_posting, emb_dir, emb_name,
              out_dir)
             for c in countries for k in range(chunks_per_country)]
    if log:
        log(f"{len(tasks)} tasks ({len(countries)} countries x {chunks_per_country}) "
            f"on {workers} workers")

    stats = []
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for i, st in enumerate(pool.imap_unordered(_run_one, tasks), 1):
            stats.append(st)
            if log:
                log(f"  [{i}/{len(tasks)}] {st['country']} chunk {st['chunk']}: "
                    f"{st['entities']:,} entities, {st['candidates_per_s1']:.1f} cand/S1, "
                    f"{st['selected']:,} selected, {st['wall']:.0f}s")
    if log:
        log(f"all shards done in {time.time() - t0:.0f}s, merging ...")

    sel, cand = {}, {}
    for st in stats:
        with open(st["path"], "rb") as fh:
            s, c = pickle.load(fh)
        sel.update(s)
        cand.update(c)
        os.remove(st["path"])
    if log:
        log(f"merged {len(cand):,} entities")
    return sel, cand, stats
