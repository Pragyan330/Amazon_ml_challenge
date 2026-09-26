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

---

# Stage 1b - stripped pipeline, Jaccard family, channel confidence

**Validation macro F_0.5 = 0.7981** (was 0.7672), threshold 0.725.

| | stage 1 | stage 1b |
|---|---|---|
| macro F_0.5 | 0.7672 | **0.7981** |
| best threshold | 0.900 (knife edge) | **0.725 (flat top 0.705-0.735)** |
| micro precision / recall | 0.9639 / 0.5909 | 0.9011 / 0.6876 |
| candidates kept per entity | 30.1 | 18.6 |
| candidate pos:neg ratio | 1 : 354 | **1 : 4.7** |
| candidate pair recall | 94.43% | 93.31% |
| oracle over candidates | 0.9803 | 0.9756 |
| singletons kept empty | 90.61% | 68.80% |
| US / India | 0.7902 / 0.7329 | 0.8262 / 0.7555 |
| runtime (US + India) | 1,728s | **909s** |

The score gain matters less than the shape change. Stage 1's optimum sat exactly on a
discontinuity, earning its score by excluding a 131,231-negative spike at 0.8956 by a
margin of 0.0044. Stage 1b has no such spike above the threshold - the largest is 57,552
at 0.450, far below it - and the optimum is flat across 0.705-0.735, so it should survive
the 23% density shift into the test set far better.

## What changed

**1. Normalisation happened twice per record.** `Blocker.keys()` and `prepare()` both ran
`fold()`, `name_core()` and `addr_tokens()` independently. A single `Rec` (see
`src/ber/record.py`) is now built once and serves both blocking and scoring, with character
n-grams built lazily since only pairs surviving the cheap gate need them.

**2. A cheap gate before the expensive features.** Plain token Jaccard on name and address
is computed from sets that already exist; if neither reaches 0.10 the pair is dropped before
any n-gram is built. Measured on 191,388 true pairs, that gate loses 0.021% of them.

**3. `max()` over views replaced by a weighted combination.** Taking the best of several
views let negatives claim credit from whichever view was most generous, which is what
degraded separation so badly (negatives at mean 0.512, p90 0.806).

**4. Channel-specific confidence** replaces the single symmetric `missing_penalty`, indexed
by which evidence channels actually carried signal. A perfect name match with no address now
scores 0.45 rather than 0.8956 - priced as the 13:1 false-merge risk it was measured to be.

**5. IDF-weighted (generalised) Jaccard** as the primary token measure:
`sum(idf over A&B) / sum(idf over A|B)`. Preferred to cosine because cosine normalises each
side independently and so rewards a single rare shared token even when the rest of the record
disagrees, while this form charges for every unmatched token.

## Measured cost

| | before | after | |
|---|---|---|---|
| `keys()` per record | 38.59 us | **3.50 us** | 11.0x |
| `build`/`prepare` per record | 40.97 us | 35.32 us | 1.16x |
| record-side total | 79.56 us | **38.82 us** | 2.05x |
| `score_pair` per pair | 11.40 us | **2.50 us** | 4.56x |
| full test projection, 1 thread | 2.88 h | **0.69 h** | 4.17x |

End-to-end validation runtime went 1,728s -> 909s (1.90x), consistent with the projection.

Note on the gate: it rejects 94.4% of a *random* pair mix but only 13-20% of the real
candidate set, because real candidates share a blocking key and therefore always have some
overlap. The pair-side speedup is real and measured; the 94.4% figure is not representative.

## Open items

- **Candidate recall fell 94.43% -> 93.31%**, dropping the oracle to 0.9756. Channel
  confidence pushes some true pairs below the 0.34 prefilter. Lowering the prefilter should
  recover this cheaply.
- **Singletons regressed, 90.61% -> 68.80% kept empty.** 1,937 polluted singletons cost
  roughly 0.018 of macro score. A consequence of the lower threshold; the per-entity
  expected-F_0.5 rule is the principled fix.
- **India still trails US** (0.7552 vs 0.8267). The multilingual encoder targets exactly
  this gap and is written but not yet run.

---

# Stage 1c - multilingual encoder

**Validation macro F_0.5 = 0.7998** (from 0.7981). LaBSE, Apache-2.0, 471M parameters.

| | stage 1b | stage 1c |
|---|---|---|
| macro F_0.5 | 0.7981 | **0.7998** |
| India | 0.7552 | 0.7603 |
| US | 0.8262 | 0.8262 |
| candidate pair recall | 93.31% | 93.43% |
| true links in candidate set | 357,862 | 358,289 (+427) |
| runtime (US shard) | 521s | 503s |

957 entities improved against 36 worsened - a 27:1 ratio, so the signal is correct - but the
scope is small. Non-Latin names are ~7% of Source-2/3 records, and for most of those the
address channel was already carrying the pair; the encoder only adds where the address is
*also* weak, which is a thin intersection. Embedding cost 2.8 minutes for 863,419 texts at
5,155 texts/s, and scoring runtime did not measurably change, so it is kept - but it is not a
lever.

## Encoder calibration

Measured on 4,000 real non-Latin true pairs before being trusted:

| | mean | p05 | p50 | p95 |
|---|---|---|---|---|
| POS (true pair) | 0.871 | 0.741 | 0.889 | 0.939 |
| NEG-hard (S1 sharing a name token) | 0.563 | 0.340 | 0.568 | 0.750 |
| NEG-rand (random same-country S1) | 0.399 | 0.135 | 0.425 | 0.599 |

AUC against hard negatives **0.9871**. Cosine >= 0.783 keeps 90% of true pairs while
admitting 1.93% of hard negatives. The encoder reads transliteration cleanly across scripts:
Gujarati/Devanagari/Bengali/Tamil/Kannada true pairs all land near 0.88-0.92.

That measurement caught a unit mismatch that would not have raised an error.
``embeddings.similarity()`` returned ``0.5*(cos+1)`` while the scorer's threshold was written
as a raw cosine, so the configured 0.55 actually meant a raw cosine of 0.10 - no filtering at
all - and read the other way it sat exactly on the hard-negative median, admitting half of
them as genuine name evidence. Both readings were wrong. Everything is now in raw cosine with
a calibrated band [0.70, 0.95].

The encoder channel also has a monotonicity guard: taking the better of using and ignoring it,
so extra evidence can never lower a score. Without it, a perfectly good 0.92 cosine dragged a
strong address match from 0.95 down to 0.86 by diluting the weighted mean.

# Where the remaining score actually is

Decomposition of 383,501 true links at threshold 0.72:

| | count | share of true links |
|---|---|---|
| Never entered the candidate set | 25,212 | 6.57% |
| **In candidates but scored below threshold** | **90,539** | **23.61%** |
| False positives emitted | 31,073 | - |
| Singletons polluted | 2,088 / 6,209 | costs 0.0189 of macro score |

Nearly a quarter of all true links are retrieved correctly and then discarded by a scalar
threshold. That is the bottleneck - not blocking (6.6%), not transliteration (0.2%), not the
feature set. Micro precision/recall at the optimum is 0.8960 / 0.6982: the threshold is set
high to protect precision and recall pays for it.

This is what the metric derivation predicts. The break-even probability for emitting one more
match is ``0.8 * F_current``, so the bar must rise per entity as matches accumulate, and one
global number cannot express that. Per-entity expected-F_0.5 selection targets the 23.61%
directly and needs no model, no GPU and no new data. It should also fix the singleton leak:
an entity whose candidates are all weak would choose the empty set on its own.

Oracle over the current candidate set is 0.9756 against 0.7998 achieved.

# Parallelism

``src/ber/parallel.py`` shards Source 1 across processes; each worker builds its own index and
streams all of Source 2/3 past it. Record-side work is duplicated per worker while pair-side
work divides, so wall time is ``record_side + pair_side / W`` - for the full test set
13.2 + 159.7/W minutes, about 23 minutes at W=16 against 2.9 hours single-threaded. That is
~7.5x, not the 12x first estimated: the estimate ignored the duplicated normalisation.

Sharding Source 2/3 instead would divide both halves but needs ~2.5 GB of prepared records per
worker for the India shard, capping out near four workers and landing slower.

Partitioning is unit-tested for disjointness, completeness and balance across 1/4/7/16 chunks,
composed with the validation subsample.

---

# Stage 2+3 - learned matcher and expected-F_0.5 set selection

**Validation macro F_0.5 = 0.9249** (rule baseline 0.7994 on the same split).

LightGBM over 35 pairwise features, reranking the rule pipeline's candidate set. Measured on
22,110 held-out entities; train/calib/test splits are entity-disjoint and only the test split
is ever reported.

| | macro F_0.5 |
|---|---|
| Rule baseline, threshold 0.725 | 0.7994 |
| GBM + best fixed threshold (0.50) | 0.8727 |
| **GBM + expected-F_0.5, exact** | **0.9249** |
| GBM + expected-F_0.5, uncalibrated probabilities | 0.9251 |
| GBM + expected-F_0.5, ratio approximation | 0.9243 |
| Oracle over the same candidate set | 0.9758 |

Model: 1,386 rounds, calib AUC 0.99797, 94s to train. Selection runs in 4.5s for 22,110
entities.

| | rule | GBM + expected-F_0.5 |
|---|---|---|
| micro precision | 0.9011 | 0.9767 |
| micro recall | 0.6876 | 0.8510 |
| singletons kept empty | 68.80% | **83.97%** |
| India | 0.7552 | 0.9064 |
| US | 0.8267 | 0.9374 |

## What actually produced the gain

**Set selection, not the model.** The GBM with a fixed threshold gives 0.8727; swapping the
threshold for per-entity expected-F_0.5 adds a further **+0.0522**. That is the single largest
improvement in the project so far and it needed no extra features, no extra data and 4.5
seconds of compute. It is exactly what the error decomposition predicted: 23.61% of true links
were sitting inside the candidate set and being discarded by a scalar threshold.

It also repaired the singleton leak on its own - 68.80% to 83.97% kept empty - because an
entity whose candidates are all weak now chooses the empty set rather than being dragged over
a global threshold.

## Two predictions that did not survive contact

**Calibration turned out to be unnecessary.** The three-way split existed specifically to
calibrate honestly, and the answer is that LightGBM's binary-logloss output was already
calibrated: weighted calibration error 0.0006 raw against 0.0001 after isotonic, and the
*uncalibrated* probabilities actually scored marginally higher (0.9251 vs 0.9249, within
noise). The isotonic step is dead weight here. The methodology still matters - fitting a
calibrator in-sample would have made this impossible to detect - but the correction it applies
is nil.

**The exact selector beats the ratio approximation by 0.0006**, not the margin the theory
suggested. The approximation disagrees with the exact computation on 4.2% of entities, but
those disagreements are concentrated on entities where both choices score almost the same.
The exact version is verified against Monte Carlo (max error 0.0013 against ~0.002 MC noise)
and costs nothing, so it stays - but it is not where the points are.

## Feature importance (gain)

| feature | gain |
|---|---|
| base_score (rule score) | 5,680,602 |
| rank (position within entity) | 654,288 |
| num_score | 382,504 |
| n_gram_dice | 343,110 |
| a_cont | 247,520 |
| a_gram_dice | 156,931 |
| len_ratio_addr | 147,527 |
| emb_cos (LaBSE) | 130,362 |

The rule score dominates, so the hand-built scorer is doing real work rather than being
replaced. `rank` is second, which vindicates the per-entity competition features: matching is
a contest within an entity and an absolute similarity cannot express that. The encoder cosine
earns its place at eighth.

## The ceiling is now candidate recall

At 0.9249 against an oracle of 0.9758, the matcher captures 94.8% of what this candidate set
allows. **The leaderboard leader is at 0.98, which this candidate set cannot reach even with a
perfect matcher.** The binding constraint has moved from the matcher to blocking: 6.57% of
true links never enter the candidate set.

The measured blocking union ceiling (name-4gram OR address-token) is 99.98%, so that recall is
reachable. It is being lost in the 0.34 prefilter and the df-capped keys, both of which were
tightened for speed. That is the next target.
