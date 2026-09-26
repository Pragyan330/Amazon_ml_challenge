"""Per-entity set selection by maximising expected F_0.5.

A single global threshold cannot be optimal for this metric. Writing the score in its reduced
form, ``F = 1.25 * tp / (0.25 * t + p)``, the break-even probability for emitting one more
candidate when ``tp`` of ``p`` emitted are already correct is

    pi* = tp / (0.25 * t + p) = F_current / 1.25

so the bar *rises* as an entity accumulates matches: 0.50 for the second pick on a 4-match
entity, 0.67 for the third, 0.75 for the fourth. One number cannot express that, which is why
23.61% of true links currently sit inside the candidate set and get discarded.

``t`` is unknown at prediction time, so we treat each candidate's label as an independent
Bernoulli draw and choose the set size maximising expected F_0.5.

**This is computed exactly, not approximated.** With candidates sorted by probability,
selecting the top ``n`` gives ``tp ~ PoissonBinomial(p_1..p_n)`` and the unselected true
matches ``fn ~ PoissonBinomial(p_{n+1}..p_k)``, independent, so

    E[F | n] = 1.25 * sum_a sum_b  P(tp=a) P(fn=b) * a / (0.25 (a + b) + n)

Both distributions come from a linear-time DP and the double sum is a vector-matrix-vector
product. The tempting shortcut - ``1.25 * sum(p_i) / (0.25 * sum(all p) + n)`` - is a ratio
of expectations rather than an expectation of a ratio, and is biased by Jensen's inequality.
Worse, it is usually paired with the *exact* ``P(t=0) = prod(1 - p_i)`` for the empty set, so
the empty-versus-not comparison mixes an exact value against an approximate one - and that is
precisely the singleton decision, where a wrong call costs a full 1.0.

The empty set is handled exactly here too: ``E[F | n=0] = P(t = 0)``, since an entity with no
true matches scores 1.0 for an empty prediction and 0.0 otherwise.
"""

import numpy as np

BETA2 = 0.25
ONE_PLUS_BETA2 = 1.25
KMAX = 24  # candidates considered per entity; the true maximum match count is 11


def _poisson_binomial_prefix(p, k):
    """pref[n, a] = P(exactly a successes among the first n)."""
    pref = np.zeros((k + 1, k + 1))
    pref[0, 0] = 1.0
    for n in range(1, k + 1):
        pn = p[n - 1]
        prev = pref[n - 1]
        pref[n, 0] = prev[0] * (1.0 - pn)
        pref[n, 1:n + 1] = prev[1:n + 1] * (1.0 - pn) + prev[0:n] * pn
    return pref


def _poisson_binomial_suffix(p, k):
    """suf[n, b] = P(exactly b successes among p[n:])."""
    suf = np.zeros((k + 1, k + 1))
    suf[k, 0] = 1.0
    for n in range(k - 1, -1, -1):
        pn = p[n]
        nxt = suf[n + 1]
        m = k - n
        suf[n, 0] = nxt[0] * (1.0 - pn)
        suf[n, 1:m + 1] = nxt[1:m + 1] * (1.0 - pn) + nxt[0:m] * pn
    return suf


def expected_f05_curve(probs, kmax=KMAX):
    """Return (best_n, expected_scores) where expected_scores[n] = E[F_0.5 | select top n]."""
    p = np.sort(np.asarray(probs, dtype=float))[::-1]
    if p.size > kmax:
        p = p[:kmax]
    k = p.size
    if k == 0:
        return 0, np.array([1.0])

    pref = _poisson_binomial_prefix(p, k)
    suf = _poisson_binomial_suffix(p, k)

    out = np.empty(k + 1)
    out[0] = suf[0, 0]  # P(no true matches at all) - exact
    for n in range(1, k + 1):
        a = np.arange(n + 1)
        b = np.arange(k - n + 1)
        denom = BETA2 * (a[:, None] + b[None, :]) + n
        w = a[:, None] / denom
        out[n] = ONE_PLUS_BETA2 * (pref[n, :n + 1] @ w @ suf[n, :k - n + 1])
    return int(np.argmax(out)), out


def select_expected_f05(scored, kmax=KMAX):
    """Pick the ids maximising expected F_0.5.

    ``scored`` is a list of (probability, id); it is sorted here rather than assumed sorted,
    because a model rescoring the rule pipeline's output scrambles the original order.
    """
    if not scored:
        return []
    ranked = sorted(scored, key=lambda x: -x[0])
    if len(ranked) > kmax:
        ranked = ranked[:kmax]
    best_n, _ = expected_f05_curve([s for s, _ in ranked], kmax=kmax)
    return [tid for _, tid in ranked[:best_n]]


def expected_f05_ratio_approx(probs, kmax=KMAX):
    """The ratio-of-expectations shortcut, kept only to measure how far off it is."""
    p = np.sort(np.asarray(probs, dtype=float))[::-1]
    if p.size > kmax:
        p = p[:kmax]
    if p.size == 0:
        return 0
    e_t = p.sum()
    best_n, best_f = 0, float(np.prod(1.0 - p))
    run = 0.0
    for n in range(1, p.size + 1):
        run += p[n - 1]
        f = ONE_PLUS_BETA2 * run / (BETA2 * e_t + n)
        if f > best_f:
            best_f, best_n = f, n
    return best_n
