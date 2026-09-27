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

from array import array
import multiprocessing as mp
import os
import pickle
import time


def _run_one(task):
    from .blocking import Blocker, idf_from_df
    from .embeddings import load as load_emb
    from .pairfeatures import (add_cross_source_features, add_rank_features,
                           pair_features, token_signature)
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

    use_xs = bool(m.get("cross_source"))

    def feature_fn(r1, r2, score, cos, is_s3):
        # array('f') rather than a list: a Python list of floats costs ~1,176 bytes (one
        # pointer plus one boxed float per column) against ~156 packed. Across workers
        # holding ~750k rows each that is the difference between 13 GB and under 2 GB.
        #
        # The token signature rides along when the model wants cross-source features: 108
        # bytes, against 1,126 for the token set and 1,686 for the whole record. Those
        # would need 8.3 GB and 12.4 GB respectively across five workers.
        row = array("f", pair_features(r1, r2, score, idf_name_f, idf_addr_f, default_f,
                                       cos, is_s3))
        return (row, token_signature(r2)) if use_xs else row

    # The rule score is itself a model feature - by a wide margin the most important one -
    # so it must be computed with the *training* IDF the model was fitted against. Only the
    # Blocker uses test statistics, because key rarity has to be judged against the pool
    # actually being searched. Passing test IDF here instead would shift the single feature
    # the model leans on hardest.
    result = run_shard(country, s1_path, s23_paths, blocker, idf_name_f, idf_addr_f,
                       default_f, Weights(), prefilter=prefilter, topk=topk,
                       s1_keep=chunk_keeper(chunk, n_chunks, 1), emb=emb,
                       feature_fn=feature_fn)

    # Emit per-candidate probabilities rather than a finished selection. Selection is done
    # in the parent, because resolving contention - a Source-2/3 record may belong to only
    # one Source-1 entity - is a global decision that a shard holding 1/12 of Source 1
    # cannot make. The last submission had 75,823 contested records and at least 179,788
    # provably-wrong links as a result.
    cand_out, prob_out = {}, {}
    rows, owners = [], []
    for slot, bucket in result.candidates.items():
        eid = result.s1_ids[slot]
        cand_out[eid] = [t for _, t, _ in bucket]
        if use_xs:
            feats = [p[0] for _, _, p in bucket]
            sigs = [p[1] for _, _, p in bucket]
            add_cross_source_features(feats, sigs, [t[1] == "3" for _, t, _ in bucket])
        else:
            feats = [r for _, _, r in bucket]  # stay packed; array('f') supports extend
        add_rank_features(feats, [s for s, _, _ in bucket])
        start = len(rows)
        rows.extend(feats)
        owners.append((eid, start, len(feats)))
    for eid in result.s1_ids:
        cand_out.setdefault(eid, [])
        prob_out.setdefault(eid, array("f"))

    n_sel = 0
    if rows:
        X = np.asarray(rows, dtype=np.float32)
        probs = booster.predict(X, num_iteration=best_iter or None)
        for eid, start, n in owners:
            prob_out[eid] = array("f", probs[start:start + n])
            n_sel += len(select_expected_f05(
                [(float(probs[start + i]), cand_out[eid][i]) for i in range(n)]))

    path = os.path.join(out_dir, f"sub_{country}_{chunk:03d}.pkl")
    with open(path, "wb") as fh:
        pickle.dump((prob_out, cand_out), fh, protocol=pickle.HIGHEST_PROTOCOL)
    st = dict(result.stats)
    st.update({"country": country, "chunk": chunk, "path": path, "wall": time.time() - t0,
               "entities": len(cand_out), "selected": n_sel})
    return st


def run_submission(countries, s1_path, s23_paths, test_stats, train_stats_path,
                   matcher_path, out_dir, workers, chunks_per_country, prefilter, topk,
                   df_cap, max_posting, emb_dir, emb_name, log=None, resume=True,
                   keep_parts=False):
    """Run every (country, chunk) shard and merge the results.

    Shards are independent and each writes its own file, so ``resume=True`` skips any whose
    output already exists. A full run is ~55 minutes; without this, an interruption costs all
    of it. Partial files are only deleted after a successful merge (unless ``keep_parts``),
    so stopping mid-run is always safe.
    """
    os.makedirs(out_dir, exist_ok=True)

    def part_path(c, k):
        return os.path.join(out_dir, f"sub_{c}_{k:03d}.pkl")

    all_pairs = [(c, k) for c in countries for k in range(chunks_per_country)]
    done = [(c, k) for c, k in all_pairs if resume and os.path.exists(part_path(c, k))]
    todo = [(c, k) for c, k in all_pairs if (c, k) not in set(done)]
    if log and done:
        log(f"resuming: {len(done)} shard(s) already on disk, {len(todo)} to run")

    tasks = [(c, k, chunks_per_country, s1_path, s23_paths, test_stats, train_stats_path,
              matcher_path, prefilter, topk, df_cap, max_posting, emb_dir, emb_name,
              out_dir)
             for c, k in todo]
    if log:
        log(f"{len(tasks)} task(s) on {workers} workers")

    stats = []
    t0 = time.time()
    if tasks:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for i, st in enumerate(pool.imap_unordered(_run_one, tasks), 1):
                stats.append(st)
                if log:
                    log(f"  [{i}/{len(tasks)}] {st['country']} chunk {st['chunk']}: "
                        f"{st['entities']:,} entities, {st['selected']:,} selected, "
                        f"{st['wall']:.0f}s")
        if log:
            log(f"shards finished in {time.time() - t0:.0f}s, merging ...")

    prob, cand = {}, {}
    merged = []
    for c, k in all_pairs:
        path = part_path(c, k)
        if not os.path.exists(path):
            if log:
                log(f"WARNING: missing shard {c}/{k}; its entities will have empty rows")
            continue
        with open(path, "rb") as fh:
            part_prob, part_cand = pickle.load(fh)
        prob.update(part_prob)
        cand.update(part_cand)
        merged.append(path)
    if not keep_parts:
        for path in merged:
            os.remove(path)
    if log:
        log(f"merged {len(merged)} shard(s), {len(cand):,} entities")

    sel = select_with_contention(prob, cand, log=log)
    return sel, cand, stats


def select_with_contention(prob, cand, log=None):
    """Global set selection that respects one-Source-1-per-record.

    The ground truth guarantees every matched Source-2/3 record belongs to exactly one
    Source-1 entity - verified across all 7,638,365 matched ids, zero exceptions - but a
    per-entity selection cannot enforce it, because the competing entities live in different
    shards.

    Iterated to a fixed point. Each pass awards every contested record to its most
    confident claimant and lets the losers re-select from what remains - rather than simply
    deleting the record, because losing one member changes how many the rest are worth
    emitting, and expected-F_0.5 must be recomputed on the set actually available.

    Iteration is required, not decorative: re-selection promotes new candidates, which can
    collide with other entities' choices. A single pass left 4,411 records contested out of
    an initial 86,441.
    """
    from .setselect import select_expected_f05

    t0 = time.time()
    pairs = {e: [(float(p), t) for p, t in zip(prob.get(e, ()), cand.get(e, ()))]
             for e in cand}
    first = {e: select_expected_f05(v) for e, v in pairs.items()}

    before = sum(len(v) for v in first.values())
    current = {e: list(v) for e, v in first.items()}
    banned = {}
    total_contested = 0

    # Iterate to a fixed point. Re-selection creates *new* contention: an entity that loses
    # a record may promote a different one that another entity also chose. A single pass
    # left 4,411 records still contested out of an initial 86,441.
    for it in range(12):
        contested = {t for t, n in _claim_counts(current).items() if n > 1}
        if it == 0:
            total_contested = len(contested)
            if log:
                log(f"contention: {len(contested):,} record(s) claimed by >1 entity")
        if not contested:
            break

        owner = {}
        for e, ids in current.items():
            pm = dict((t, p) for p, t in pairs[e])
            for t in ids:
                if t not in contested:
                    continue
                p = pm.get(t, 0.0)
                cur = owner.get(t)
                if cur is None or p > cur[0]:
                    owner[t] = (p, e)

        losers = set()
        for e, ids in current.items():
            lost = [t for t in ids if t in contested and owner[t][1] != e]
            if lost:
                banned.setdefault(e, set()).update(lost)
                losers.add(e)
        if not losers:
            break
        for e in losers:
            block = banned[e]
            current[e] = select_expected_f05([(p, t) for p, t in pairs[e]
                                              if t not in block])
        if log:
            log(f"  pass {it + 1}: {len(contested):,} contested, "
                f"{len(losers):,} entities re-selected")

    final = {e: set(v) for e, v in current.items()}
    if log:
        after = sum(len(v) for v in final.values())
        left = sum(1 for n in _claim_counts(final).values() if n > 1)
        log(f"contention: {total_contested:,} initial -> {left:,} remaining, "
            f"links {before:,} -> {after:,} ({after - before:+,}) in "
            f"{time.time() - t0:.0f}s")
    return final


def _claim_counts(selection):
    counts = {}
    for ids in selection.values():
        for t in ids:
            counts[t] = counts.get(t, 0) + 1
    return counts
