"""Run shards across processes.

**Why shard Source 1 rather than Source 2/3.** Each worker holds a slice of Source 1, builds
its own index, and streams all of Source 2/3 for the country past it. That means the
record-side work (normalising 10M records) is repeated per worker while the pair-side work
(the ~840M pair scores that dominate at full scale) divides cleanly. Wall time is therefore

    record_side + pair_side / W

which for the full test set is 13.2 + 159.7/W minutes: about 23 minutes at W=16 against
2.9 hours single-threaded. The alternative - partitioning Source 2/3 and giving every worker
the whole Source-1 index - divides both halves but needs ~2.5 GB of prepared records per
worker for the India shard, so it caps out around four workers and lands slower.

Re-streaming is cheap in practice: the training sources total ~1 GB and the test sources
~1 GB, so after the first pass they sit in the OS page cache.

Workers write partial results to disk and the parent merges them. Returning them through the
pool would mean pickling tens of millions of tuples back to the parent.
"""

import multiprocessing as mp
import os
import pickle
import time


def chunk_keeper(chunk_index, n_chunks, outer_every=1):
    """Predicate selecting the Source-1 entities belonging to one chunk.

    ``outer_every`` composes with a validation subsample, so parallel and single-process runs
    cover exactly the same entities.
    """
    if n_chunks <= 1 and outer_every <= 1:
        return None

    def keep(eid):
        try:
            n = int(eid.split("-", 1)[1])
        except (IndexError, ValueError):
            return False
        if outer_every > 1 and n % outer_every:
            return False
        return (n // max(1, outer_every)) % n_chunks == chunk_index if n_chunks > 1 else True
    return keep


def _run_one(task):
    """Worker entry point. Imports inside the function so spawn does not pay for them twice."""
    from .blocking import Blocker, idf_from_df
    from .embeddings import load as load_emb
    from .pipeline import as_id_map, run_shard
    from .scorer import Weights

    (country, chunk, n_chunks, outer_every, s1_path, s23_paths, stats_path,
     weights_kw, prefilter, topk, df_cap, max_posting, emb_dir, emb_name, out_dir) = task

    t0 = time.time()
    with open(stats_path, "rb") as fh:
        df_name, df_addr, n_docs = pickle.load(fh)
    idf_name, default_idf = idf_from_df(df_name, n_docs)
    idf_addr, _ = idf_from_df(df_addr, n_docs)

    emb = load_emb(emb_dir, emb_name) if emb_name else None
    blocker = Blocker(df_name, df_addr, df_cap=df_cap, max_posting=max_posting)
    result = run_shard(country, s1_path, s23_paths, blocker, idf_name, idf_addr, default_idf,
                       Weights(**weights_kw), prefilter=prefilter, topk=topk,
                       s1_keep=chunk_keeper(chunk, n_chunks, outer_every), emb=emb)

    path = os.path.join(out_dir, f"part_{country}_{chunk:03d}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(as_id_map(result), fh, protocol=pickle.HIGHEST_PROTOCOL)
    st = dict(result.stats)
    st.update({"country": country, "chunk": chunk, "path": path,
               "wall": time.time() - t0})
    return st


def run_parallel(countries, s1_path, s23_paths, stats_path, out_dir, weights_kw=None,
                 workers=None, chunks_per_country=None, outer_every=1, prefilter=0.34,
                 topk=40, df_cap=60, max_posting=200, emb_dir=None, emb_name=None,
                 log=None):
    """Run every (country, chunk) task across a process pool and merge the results.

    Returns (scored_map, country_of, stats_list).
    """
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    chunks_per_country = chunks_per_country or workers
    os.makedirs(out_dir, exist_ok=True)
    weights_kw = weights_kw or {}

    tasks = [
        (c, k, chunks_per_country, outer_every, s1_path, s23_paths, stats_path,
         weights_kw, prefilter, topk, df_cap, max_posting, emb_dir, emb_name, out_dir)
        for c in countries for k in range(chunks_per_country)
    ]
    if log:
        log(f"{len(tasks)} tasks ({len(countries)} countries x {chunks_per_country} chunks) "
            f"on {workers} workers")

    scored, country_of, stats = {}, {}, []
    t0 = time.time()
    ctx = mp.get_context("spawn")  # required on Windows
    with ctx.Pool(processes=workers) as pool:
        for i, st in enumerate(pool.imap_unordered(_run_one, tasks), 1):
            stats.append(st)
            if log:
                log(f"  [{i}/{len(tasks)}] {st['country']} chunk {st['chunk']}: "
                    f"{st['s1_records']:,} S1, {st['candidates_per_s1']:.1f} kept/S1, "
                    f"{st['wall']:.0f}s")
    if log:
        log(f"all shards done in {time.time() - t0:.0f}s, merging ...")

    for st in stats:
        with open(st["path"], "rb") as fh:
            part = pickle.load(fh)
        scored.update(part)
        for eid in part:
            country_of[eid] = st["country"]
        os.remove(st["path"])
    if log:
        log(f"merged {len(scored):,} entities")
    return scored, country_of, stats
