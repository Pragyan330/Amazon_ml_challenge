# Results

## Stage 1 — rule-based baseline

**Validation macro F_0.5 = 0.7672**

Run: `python scripts/validate.py --sample-every 20`

| | |
|---|---|
| Validation entities | 110,550 (1 in 20 of training Source 1) |
| True links | 383,501 |
| Singletons | 6,209 (5.62%) |
| Opposing pool | all 10,320,219 training S2/S3 records |
| Candidates kept | 32.31 per entity (from 485.1 considered, US) |
| **Candidate pair recall** | **94.430%** |
| **Oracle over that candidate set** | **0.9796** |
| Best threshold | 0.900 |
| Micro precision / recall | 0.9639 / 0.5909 |
| Predicted links | 235,099 (true 383,501) |
| Singletons correctly left empty | 5,626 / 6,209 = 90.61% |
| US / India | 0.7902 / 0.7329 |
| Runtime | 1,013s (US) + 715s (India), single-threaded |

The subsample is on the Source-1 side only; every sampled entity is scored against the
complete opposing pool, so precision reflects the real distractor density.

For scale: an all-empty submission scores 0.056, and the ceiling given this candidate set is
0.9796. So stage 1 captures roughly 78% of the available headroom.

## The dominant error mode: name-only evidence

The threshold sweep is badly non-monotonic between 0.80 and 0.91, and the winning threshold
sat exactly on a discontinuity. That turned out to be diagnostic rather than a nuisance.

Because the scorer renormalises by the evidence weight actually present, a *perfect* match
produces one of six discrete scores depending on which channels existed:

| Evidence present | Score of a perfect match |
|---|---|
| name only | **0.895600** |
| address only | 0.901000 |
| name + number | 0.919000 |
| address + number | 0.924400 |
| name + address | 0.976600 |
| all three | 1.000000 |

Counting candidates at each exact value:

| Score | Positives | Negatives | Ratio |
|---|---|---|---|
| 0.895600 (perfect name, no address) | 10,194 | **131,231** | 1 : 12.9 |
| 0.901000 (perfect address, no usable name) | 710 | 136 | 5.2 : 1 |

**A perfect name match with no address is ~13 times more likely to be a false merge than a
true match. A perfect address match with no usable name is 5 times more likely to be
correct.** That is the 39.5%-duplicate-name finding showing up as a concrete, measurable
failure: 131,231 of the 3.21M negatives sit in a single spike, and the optimal threshold of
0.900 earns its score almost entirely by sitting just above it.

The `missing_penalty` mechanism is therefore too crude. It is symmetric — a single knob
applied to a weight ratio — so it prices name-only and address-only evidence at 0.8956 vs
0.9010, nearly identical, when their reliability differs by two orders of magnitude.

**Fix: channel-specific reliability rather than one symmetric penalty.** Name-only evidence
should be worth very little on its own; address-only evidence should be worth nearly as much
as a full match. This is the single highest-value change to stage 1 and needs no model.

## Secondary observation: `max()` inflates negatives

Each channel takes the best of several views (IDF cosine, character n-gram Dice,
containment). That is optimistic by construction: a negative pair gets credit for whichever
view happens to be most generous.

The effect is visible in the separation. The throwaway blend used during exploration had
negatives at mean 0.129 / p90 0.238; this scorer has negatives at **mean 0.512 / p90 0.806**,
against positives at mean 0.906. The score is higher-performing overall (0.767 vs 0.743) but
far more fragile, which is why the threshold is a knife edge.

Worth testing whether a weighted *combination* of views, or a min/mean over them, separates
better than `max`.

## What the numbers say about priorities

| Observation | Implication |
|---|---|
| Oracle 0.9796 vs achieved 0.7672 | the matcher is the bottleneck, not blocking |
| Micro recall 0.5909 at the best threshold | 41% of true links are being discarded to protect precision — exactly what a per-entity selection rule should recover |
| 131,231 negatives in one score spike | fix the name-only channel before anything else |
| Candidate recall 94.43% vs 99.98% ceiling | ~5.5 points of recall left in blocking, cheap to chase |
| India 0.7329 vs US 0.7902 | transliteration and generic names; the 7.3% non-Latin tail contributes zero name evidence today |
| 485 pairs considered per entity, 31 µs each | ~7 hours single-threaded for the full test set; needs multiprocessing or tighter blocking |

## Threshold sweep

Fine sweep, computed from cached scores (`artifacts/val_scored.pkl`):

```
0.700  0.6010      0.840  0.7257      0.900  0.7672  <== best
0.760  0.6236      0.860  0.7206      0.905  0.7573
0.800  0.6417      0.880  0.7043      0.920  0.7298
0.810  0.7076      0.890  0.6948      0.950  0.6303
0.830  0.7222      0.895  0.6897      1.000  0.3074
```

Capping matches per entity does not help once the threshold is tuned (max_k=8 and no cap both
give 0.7672; max_k=3 gives 0.7566), which is expected — a cap is a crude stand-in for the
rising marginal bar that proper set selection provides directly.
