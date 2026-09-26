# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [FILL IN]
**Team Members:** [FILL IN]
**Submission Date:** [FILL IN]

---

## 1. Executive Summary

A three-stage pipeline: country-partitioned blocking on rarest tokens, a LightGBM matcher
over 35 pairwise similarity features, and per-entity set selection that maximises expected
F_0.5 exactly rather than thresholding a score. The two contributions we would highlight are
the **set-selection stage**, which alone added +0.052 macro F_0.5 over the best possible
fixed threshold on the same model, and an **evidence-weighted scorer** that prices each
similarity channel by its measured reliability instead of averaging them.

Held-out validation macro F_0.5: **0.9249**, against 0.0558 for an all-empty submission.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from the training data that determined the design. Sample sizes are stated; pair
statistics come from a uniform 1-in-40 sample (55,171 entities / 191,388 true pairs).

**Structural constraints.**

- **Country agrees on 100.000% of true pairs** (zero exceptions in 191,388). Country is a
  hard partition, so a cross-country candidate is never generated.
- **Every matched Source-2/3 record belongs to exactly one Source-1 entity** — verified
  across all 7,638,365 matched ids. The task is a many-to-one assignment.
- **5.58% of entities are singletons** (123,247), mean 3.461 matches, max 11. An all-empty
  submission therefore scores 0.0558.
- **26% of Source-2/3 records match nothing** (1,340,997 in S2, 1,340,857 in S3). These
  distractors are the main precision hazard.
- **Singletons are statistically invisible**: 5.583% for US against 5.588% for India, with
  identical name and address lengths to matched entities. No classifier on Source-1 features
  alone can find them; "no match" is only inferable from weak candidate evidence. This is
  why set selection, not a singleton detector, is the right mechanism.

**Noise.** How a true match's name relates to the Source-1 name:

| Relationship | Share |
|---|---|
| Shares at least one name token | 85.05% |
| Non-Latin script (transliterated) | 7.27% |
| No shared token, char-3gram >= 0.5 (typo / concatenation) | 5.65% |
| Name replaced entirely, no overlap | 1.76% |

Exact name equality is 4.59% raw, 25.74% normalised; 14.4% of true pairs share **zero** name
tokens. Ten non-Latin scripts appear (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati,
Malayalam, Odia, Gurmukhi, plus Latin-accented), and Source-1 names are never non-Latin, so
these are always romanised-against-native-script. Other corruptions: domain-style names
5.22%, `@handle`/`#hashtag` 0.83%, `d/b/a` and `M/s` prefixes 1.14%, leading punctuation
2.39%, mojibake, token transposition, injected legal tokens.

Addresses are more reliable: 95.6% of true pairs share an address token, mean token Jaccard
0.611, but 3.3% of Source-2/3 addresses are empty.

**The central difficulty is name ambiguity, not noise.** 39.5% of Source-1 entities share an
identical normalised name with another entity in the same country; 48.1% share a
legal-suffix-stripped core. The largest group is 527 US businesses named "Meridian", and
22.1% of entities sit in a group of 20 or more. For roughly half the dataset the name carries
almost no discriminative power and **address is the primary key**. Ranking same-name
candidates by address token Jaccard resolves 95.55% of those cases correctly.

**Source formatting signatures.** Source 2 is an ALLCAPS-abbreviated near-copy (89.9% of US
addresses uppercase, street types abbreviated); Source 3 expands state names, reorders
components and is measurably noisier (address token Jaccard to Source 1: 0.726 for S2 against
0.558 for S3). Postal codes are largely stripped — present in 10.8% of US and 0.28% of India
Source-1 addresses — so postcode blocking is not viable.

**Distribution shift into the test set.** France is 15% of the test set and absent from
training. It follows the same corruption process with French vocabulary, so a
country-agnostic character-level feature set transfers, but anything hard-coded to US/India
patterns breaks on 15% of entities. Less obviously, the test set carries **23% more
Source-2/3 records per Source-1 entity** (5.754 against 4.677, consistently across all three
countries), so either the distractor rate rises to ~40% or matches per entity rise to ~4.26.
Either way a threshold tuned on training is miscalibrated — which is a further argument for
selection that adapts per entity.

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned classifier + decision-theoretic set selection

**Core Innovation:** Per-entity set selection by exact expected-F_0.5 maximisation, replacing
the usual global threshold. Writing the metric in its reduced form
`F = 1.25 * tp / (0.25 * t + p)` shows the break-even probability for emitting one more
candidate is `F_current / 1.25` — the bar *rises* as an entity accumulates matches. A scalar
threshold cannot express this. Measured on our own pipeline, 23.61% of all true links were
sitting inside the candidate set and being discarded by the threshold.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used.** Within a country partition, keys are built from each record's
*rarest* tokens, sorted by document frequency with an alphabetical tiebreak so both sides of
a pair choose the same tokens independently:

| Family | Form | Purpose |
|---|---|---|
| `a` | (country, address token) | up to 3 rarest with df <= cap |
| `n` | (country, name token) | up to 2 rarest with df <= cap |
| `x` | (country, name token, address token) | generic name at a distinct address |
| `y` | (country, 2 rarest address tokens) | records whose name is unusable |
| `z` | (country, 2 rarest name tokens) | records with an empty address |
| `d` | (country, street number, rarest address token) | numeric agreement |

Single-token keys fire only for genuinely rare tokens; composite keys always fire, because
pairing two tokens is selective by construction even when each is individually common. That
is what covers the 39.5% duplicate-name problem. Any key whose posting list exceeds a cap is
dropped as non-discriminative.

**How we ensured true matches were not lost.** We measured the recall ceiling of each key
family directly on true pairs:

| Scheme | All | US | India |
|---|---|---|---|
| Name: any shared token | 85.6% | 92.0% | 76.1% |
| Name: any shared 4-gram | 90.8% | 97.7% | 80.7% |
| Address: any shared token | 95.6% | 95.3% | 96.0% |
| **Name-4gram OR address-token** | **99.98%** | 99.98% | 99.97% |

No single family suffices — name alone caps at 85.6% — so a name path and an address path
are always active simultaneously. The address path is what recovers the 1.76% of true matches
whose name was replaced outright and the 7.27% written in a non-Latin script.

**Scaling.** Numeric tokens appear only inside composite keys. Keying on them directly
produced a candidate explosion (values like `1` and `100` are ubiquitous) that reached 13.9 GB
of RAM before we caught it. Every standalone key additionally carries a document-frequency
cap.

**Candidate pairs generated.** 38,196,413 across 1,732,544 Source-1 entities, i.e. **22.05
per entity**, against a full cross-product of 1.73M x 9.97M. That is a reduction ratio of
2.2e-06, or one pair retained per 452,000 possible. The candidate set is the last filtering
stage before the matcher: exactly the pairs the model runs inference over, and a strict
superset of the final matches. 653,923,969 pairs were considered during blocking before this
filter.

**Recall/size trade-off.** We measured it rather than guessing. Holding the matcher fixed,
the achieved score is flat from ~22 candidates per entity down to ~4.5, because everything
between the prefilter and the decision threshold was never going to be emitted:

| Rule | cand/entity | pair recall | achieved F_0.5 |
|---|---|---|---|
| score >= 0.34 | 18.60 | 93.43% | 0.8001 |
| score >= 0.55 | 4.61 | 87.26% | 0.8001 |
| top-6 & >= 0.50 | 4.50 | 86.66% | 0.8002 |

---

## 4. Matching Model

**Features used** (35 total; each channel also carries an explicit "evidence present" flag so
the model distinguishes *no address to compare* from *addresses disagree*):

- **Name:** token Jaccard, Dice, containment, IDF-weighted generalised Jaccard, character
  trigram Dice and Jaccard, and the same on legal-and-generic-stripped tokens.
- **Address:** the same family, with character n-grams taken over *sorted* tokens so Source
  3's component reordering still matches.
- **Numeric:** street/plot number agreement, plus a separate flag distinguishing "no numbers"
  from "numbers disagree".
- **Provenance and shape:** source (S2 against S3, which corrupt differently), target script
  is Latin, token counts and length ratios.
- **Per-entity competition:** rank within the entity, gap to the best candidate, ratio to the
  best, candidate count, deviation from the entity mean. Matching is a contest within an
  entity — a 0.6 score means something different as the best candidate than as the fifth —
  and no absolute feature can express that. `rank` is the second most important feature by
  gain.
- **Rule score:** the hand-built evidence-weighted blend, which is the single most important
  feature by a factor of eight.

The rule score itself uses **evidence-weighted averaging**: name, address and number
contribute only when that evidence exists, and the result is renormalised by the weight
actually used, then scaled by a confidence term indexed by *which* channels were present.
This was driven by a measurement — a perfect name match with no address is 12.9 times more
likely to be a false merge than a true match (10,194 positives against 131,231 negatives),
while a perfect address match with no usable name is 5.2 times more likely to be correct. A
symmetric missing-data penalty prices these almost identically; ours does not.

**Model type:** LightGBM binary classifier (MIT), 1,386 boosting rounds with early stopping,
63 leaves, learning rate 0.05. Validation AUC 0.99797. Training takes 94 seconds.

**Threshold selection method:** none — the model's probabilities feed set selection directly.
For each entity, with candidates sorted by probability, selecting the top *n* gives
`tp ~ PoissonBinomial(p_1..p_n)` and unselected true matches `fn ~ PoissonBinomial(p_n+1..p_k)`,
so

```
E[F | n] = 1.25 * sum_a sum_b  P(tp=a) P(fn=b) * a / (0.25 (a+b) + n)
```

is computed **exactly** by convolution, including `E[F | n=0] = P(t=0)` for the empty set. We
verified it against Monte Carlo (max error 0.0013 against ~0.002 MC noise). The common
shortcut — a ratio of expectations — is biased by Jensen's inequality and is usually paired
with an exact `P(t=0)`, so the empty-versus-not comparison mixes an exact value against an
approximate one, precisely at the singleton decision where an error costs a full 1.0.

---

## 5. Results & Error Analysis

**F_0.5 Score (macro):** **0.9249** on 22,110 held-out entities. Splits are entity-disjoint
(train / calibration / test) and only the test split is reported.

| Configuration | macro F_0.5 |
|---|---|
| All-empty submission | 0.0558 |
| Rule scorer, best fixed threshold | 0.7994 |
| LightGBM, best fixed threshold | 0.8727 |
| **LightGBM + expected-F_0.5 selection** | **0.9249** |
| Oracle over the same candidate set | 0.9758 |

Micro precision/recall 0.9767 / 0.8510. Singletons correctly left empty: 83.97%, up from
68.80% under a fixed threshold — set selection repaired this without a dedicated mechanism,
because an entity whose candidates are all weak now chooses the empty set. Per country:
US 0.9374, India 0.9064.

**Common false positives (wrong merges).** Distractor records that share a name with a
Source-1 entity and sit at a plausibly similar address. 26.2% of distractors share a
legal-stripped core name with some Source-1 entity, and 2.4% reach address Jaccard >= 0.7
against a same-name entity — genuinely ambiguous on the evidence available. The residual
cases where address similarity prefers a wrong entity are dominated by Indian records where
the corrupted address retains only city and state.

**Common false negatives (missed matches).** Decomposing at the operating point: 6.57% of
true links never enter the candidate set, and the remainder are ranked below the selected
set. Within the first group, the recurring pattern is a short name combined with an empty
address (`C & G Homes LLC` against `C & G` with no address), where no evidence exists in
either channel.

**Calibration.** We built a separate calibration split to fit isotonic regression, and found
it unnecessary: LightGBM's output was already calibrated (weighted calibration error 0.0006
raw against 0.0001 after isotonic) and uncalibrated probabilities scored marginally higher.
We report this because the three-way split is what made it detectable — fitting a calibrator
in-sample would have produced a flattering and meaningless correction.

---

## 6. Conclusion

Treating set construction as a decision problem rather than a thresholding problem was worth
more than any modelling change: +0.052 macro F_0.5 over the best achievable fixed threshold on
an identical model, with no extra features and four seconds of compute. The second lesson was
that measuring the reliability of each evidence channel — rather than assuming symmetry —
exposed a single score value holding 131,231 false positives, which no amount of threshold
tuning would have surfaced. The remaining gap to the oracle is now dominated by blocking
recall rather than by matching quality.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains the runnable pipeline; `README.md` there gives
exact end-to-end commands. Entry points, in order:

| Script | Purpose |
|---|---|
| `src/scripts/validate.py` | Blocking + rule scoring on a held-out training split |
| `src/scripts/train_matcher.py` | Trains the LightGBM matcher, evaluates set selection |
| `src/scripts/submit.py` | Generates both output files for the test set (resumable) |
| `src/scripts/embed_names.py` | Optional multilingual encoder precompute |

Core modules under `src/ber/`: `normalize`, `record`, `blocking`, `scorer`, `pairfeatures`,
`setselect`, `evaluate`, `dataio`, `pipeline`, `submitworker`.

### B. Additional Results

**Metric mechanics.** With P = tp/p and R = tp/t, precision and recall cancel:
`F_beta = (1 + b^2) * tp / (b^2 * t + p)`. Differentiating gives
`(dF/dP)/(dF/dR) = R^2 / (b^2 P^2)`, which at P = R equals `1/b^2 = 4` — a unit of precision
is worth four units of recall at the balanced point, which is a stronger asymmetry than the
"2x" framing suggests and shaped every threshold decision in this work.

**Reference points**, computed against the real match-count distribution:

| Behaviour | macro F_0.5 |
|---|---|
| All-empty submission | 0.056 |
| Perfect precision, only 1 match per entity | 0.696 |
| All true matches + 1 false positive per entity | 0.752 |
| Miss 1 true match, never a false positive | 0.871 |
| Perfect precision, up to 3 matches per entity | 0.948 |

**Optional multilingual encoder.** LaBSE (Apache-2.0, 471M) separates transliterated pairs
well — AUC 0.9871 against hard negatives, with true pairs at mean cosine 0.871 against 0.563
for same-name-token negatives — but contributes only +0.0019 macro F_0.5 overall, because the
address channel already carries most non-Latin pairs. It is disabled in the submitted
configuration; see the code README, section 5.
