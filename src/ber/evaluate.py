"""Macro F-beta scoring, matching the challenge's stated evaluation exactly.

The shipped ``utils/validate_submission.py`` only checks formatting and explicitly does not
compute a score, so we need our own to measure anything locally.

Derivation of the form used below. With P = tp/p and R = tp/t,

    F_beta = (1 + b^2) * P * R / (b^2 * P + R)
           = (1 + b^2) * tp / (b^2 * t + p)

Precision and recall cancel out, leaving a form that is linear in ``tp``. For beta = 0.5
that is ``1.25 * tp / (0.25 * t + p)``. Verified against the README's worked example
(tp=2, t=2, p=3 -> 0.714).
"""

BETA2 = 0.25
ONE_PLUS_BETA2 = 1.25


def f_beta_entity(tp, n_pred, n_true):
    """F_0.5 for a single Source-1 entity.

    Edge cases follow the challenge rules: an entity with no true matches scores 1.0 when
    we correctly predict nothing and 0.0 if we predict anything at all.
    """
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0 or tp == 0:
        return 0.0
    return ONE_PLUS_BETA2 * tp / (BETA2 * n_true + n_pred)


def macro_f_beta(predicted, truth):
    """Macro-average F_0.5 over every entity in ``truth``.

    ``predicted`` and ``truth`` are {source1_entity_id: set_of_matched_ids}. Entities
    absent from ``predicted`` count as an empty prediction, which is what the scorer would
    do to a submission missing rows.
    """
    if not truth:
        return 0.0
    total = 0.0
    for eid, true_set in truth.items():
        pred = predicted.get(eid) or ()
        tp = len(true_set.intersection(pred)) if true_set else 0
        total += f_beta_entity(tp, len(pred), len(true_set))
    return total / len(truth)


def breakdown(predicted, truth, group_of=None):
    """Diagnostics: overall score, plus micro precision/recall and per-group scores.

    ``group_of`` maps an entity id to a label (we use country) so we can see where the
    score is actually being lost.
    """
    per_group = {}
    total = 0.0
    tp_sum = pred_sum = true_sum = 0
    singleton_hits = singleton_total = 0
    for eid, true_set in truth.items():
        pred = predicted.get(eid) or set()
        tp = len(true_set & pred) if true_set else 0
        f = f_beta_entity(tp, len(pred), len(true_set))
        total += f
        tp_sum += tp
        pred_sum += len(pred)
        true_sum += len(true_set)
        if not true_set:
            singleton_total += 1
            singleton_hits += 1 if not pred else 0
        if group_of is not None:
            g = group_of.get(eid, "?")
            acc = per_group.setdefault(g, [0.0, 0])
            acc[0] += f
            acc[1] += 1
    n = len(truth)
    return {
        "macro_f05": total / n,
        "n_entities": n,
        "micro_precision": (tp_sum / pred_sum) if pred_sum else 0.0,
        "micro_recall": (tp_sum / true_sum) if true_sum else 0.0,
        "predicted_links": pred_sum,
        "true_links": true_sum,
        "singletons": singleton_total,
        "singletons_kept_empty": singleton_hits,
        "per_group": {g: (v[0] / v[1], v[1]) for g, v in sorted(per_group.items())},
    }
