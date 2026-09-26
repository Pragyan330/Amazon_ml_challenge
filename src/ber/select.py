"""Turning scored candidates into the final predicted match set.

This is deliberately the simplest possible rule for the baseline: keep everything above a
threshold, optionally capped at the k best. The analysis says a single global threshold is
provably not the optimal policy for macro F_0.5 - the break-even probability for emitting
one more match is ``0.8 * current_F``, so the bar should rise as an entity accumulates
matches - but that belongs to a later stage. Establish the floor first.
"""


def select_threshold(scored, threshold, max_k=None):
    """Ids scoring at or above ``threshold``, best first, at most ``max_k`` of them.

    ``scored`` is a list of (score, id) sorted descending, as produced by the pipeline.
    """
    out = []
    # Bug fix: don't assume `scored` is pre-sorted. Sorting explicitly prevents 
    # early-break truncation when downstream models (like GBM) scramble the order.
    for score, tid in sorted(scored, key=lambda x: x[0], reverse=True):
        if score < threshold:
            break
        out.append(tid)
        if max_k is not None and len(out) >= max_k:
            break
    return out


def sweep_threshold(scored_map, truth, thresholds, max_k=None, metric=None):
    """Score every threshold in ``thresholds`` and return (results, best).

    ``results`` is a list of (threshold, macro_f05) and ``best`` is the winning pair.
    Kept separate from the pipeline so tuning never touches the candidate generation.
    """
    if metric is None:
        from .evaluate import macro_f_beta
        metric = macro_f_beta
    results = []
    for th in thresholds:
        pred = {eid: set(select_threshold(sc, th, max_k))
                for eid, sc in scored_map.items()}
        results.append((th, metric(pred, truth)))
    best = max(results, key=lambda r: r[1])
    return results, best
