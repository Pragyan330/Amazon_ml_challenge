# Dataset analysis

Findings from the training data that the pipeline design is based on. Every number here was
measured, not assumed; sample sizes are stated where sampling was used.

## Shape

| File | Rows |
|---|---|
| `train_source1.tsv` | 2,206,821 |
| `train_source2.tsv` | 5,034,616 |
| `train_source3.tsv` | 5,285,603 |
| `train_ground_truth.tsv` | 2,206,821 |
| `test_source1.tsv` | 1,732,544 |
| `test_source2.tsv` | 4,887,273 |
| `test_source3.tsv` | 5,082,316 |

Data quality is high: exactly 4 columns on every row, no empty names, no dangling ground
truth references, ground truth covers Source 1 exactly 1:1, and no train/test id overlap.

**Trap:** S2 and S3 numeric ids collide in 26,801 cases. Never strip the `S2-`/`S3-` prefix
when using an id as a key.

## Structural constraints

These are the highest-value findings; each one removes work or removes error.

1. **Country agrees on 100.000% of true pairs** (191,388 sampled pairs, zero exceptions).
   Country is a hard partition — never generate a cross-country candidate.
2. **Every matched S2/S3 record belongs to exactly one S1 entity** — verified across all
   7,638,365 matched ids, zero exceptions. The task is a *many-to-one assignment*, so two
   S1 entities claiming the same record is always an error that can be detected and fixed
   without any model.
3. **Only 5.58% of entities are singletons** (123,247). Mean 3.461 matches per entity, max
   11 (at most 5 from S2, at most 6 from S3). An all-empty submission therefore scores
   0.0558 — there is no cheap sandbag.
4. **26% of S2/S3 records match nothing** — 1,340,997 in S2, 1,340,857 in S3. These
   distractors are the main precision hazard.
5. **Singletons are statistically invisible.** Singleton rate is 5.583% for US and 5.588%
   for India, with identical name and address lengths to matched entities. No classifier on
   S1 features alone can find them; "no match" can only be inferred from weak candidates.

## Noise

Measured over 191,388 true pairs. How a true match's name relates to the S1 name:

| Relationship | Share |
|---|---|
| Shares at least one name token | 85.05% |
| Non-Latin script (transliterated) | 7.27% |
| No shared token, char-3gram >= 0.5 (typo / concatenation) | 5.65% |
| Name replaced entirely, no overlap at all | 1.76% |
| Weak 3-gram overlap only | 0.26% |

Exact name equality is 4.59% raw, 25.74% after normalisation. 14.4% of true pairs share
**zero** name tokens, so a name-only pipeline is capped around 85% recall.

**Ten non-Latin scripts appear**, not just Devanagari: Devanagari 4.15%, Telugu 0.64%,
Kannada 0.56%, Tamil 0.51%, Bengali 0.46%, Gujarati 0.46%, Malayalam 0.29%, Odia 0.10%,
Gurmukhi 0.10%, plus Latin-accented 6.60%. S1 names are *never* non-Latin, so this is always
romanised-vs-Indic and unreachable by string similarity.

Other corruptions: domain-style names 5.22% (`lumynoriental.com`), `@handle` 0.40%,
`#hashtag` 0.43%, `d/b/a` and `M/s` prefixes 1.14%, leading punctuation junk 2.39%
(`--`, `<<`, `***`), mojibake (`Dréxkor`), token transposition
(`Baptist Columbus Inc Association`), injected legal tokens
(`Sterling Eastern Inc` -> `STERLING INC SERVICES`).

Addresses are more reliable: 95.6% of true pairs share at least one address token, only 4.4%
share none, mean token Jaccard 0.611. But 3.3% of S2/S3 addresses are empty (4.39% of
true-match records).

## Source formatting signatures

| | S1 | S2 | S3 |
|---|---|---|---|
| US address ALLCAPS | 0% | 89.9% | 0% |
| US address ends in state abbreviation | 86.4% | 83.4% | 4.1% (full state names) |
| Street types | spelled out | abbreviated | mixed |
| Address token Jaccard to S1 | — | 0.726 | 0.558 |

S2 is an ALLCAPS-abbreviated near-copy; **S3 is the noisier source** — it expands state
names, reorders components (`Tennessee, Hixson, Kensley Ln`) and substitutes county for city.

Postal codes are mostly stripped: present in only 10.8% of US and 0.28% of India S1
addresses, and essentially absent for France. **Postcode blocking is not viable.**

## The real difficulty: name ambiguity

**39.5% of S1 entities share an identical normalised name with another S1 entity in the same
country**; 48.1% share a legal-suffix-stripped core name. Largest group: 527 US businesses
named "Meridian". Only 51.5% of entities have a unique core name and 22.1% sit in a group of
20 or more.

So for about half the entities the name carries almost no discriminative power and **address
is the primary key**. Any scorer weighted toward name similarity will false-merge.

Address does resolve it. Ranking same-name S1 candidates by address token Jaccard, among
true pairs whose S1 name is not unique:

| Outcome | Share |
|---|---|
| Address picks the right S1 uniquely | 95.55% |
| Target address empty, unresolvable | 4.03% |
| Address picks a wrong S1 | 0.32% |
| Tied at top | 0.10% |

## Cross-source coherence (unexploited)

Blended name+address similarity between an S2 and an S3 record matching the **same** S1
averages **0.447**; for records in different clusters within the same country it averages
**0.015**, and **0.00%** exceed 0.5 (vs 42.7% within-cluster).

Mutual agreement between two candidates is close to diagnostic. This is the strongest signal
in the dataset and argues for a collective/graph formulation rather than independent pairwise
scoring. Not yet used by the pipeline.

## Blocking recall ceilings

Measured on true pairs — does *any* shared key exist:

| Scheme | All | US | India |
|---|---|---|---|
| Name: any shared token | 85.6% | 92.0% | 76.1% |
| Name: any shared 4-gram | 90.8% | 97.7% | 80.7% |
| Address: any shared token | 95.6% | 95.3% | 96.0% |
| Address: 3 rarest tokens | 90.9% | 93.9% | 86.3% |
| **Name-4gram OR address-token** | **99.98%** | 99.98% | 99.97% |

Both a name path and an address path must always be active. Note 4-grams fail on names
shorter than 4 characters, so a short-name fallback is required.

**Cost warning, learned the hard way:** keying on bare numeric address tokens causes a
candidate explosion (common values like `1` and `100`) — an early version reached 13.9 GB of
RAM before being killed. Numbers belong only in composite keys, and every standalone key
needs a document-frequency cap.

## Train to test distribution shift

**France is 15% of the test set and absent from training** (259,452 S1 entities). Country mix
moves from 60% US / 40% India to 46.8% India / 38.3% US / 15.0% France.

France follows the *same* noise process with French vocabulary: S2 abbreviates
(`63 R. DE DIEPPE, LILLE, Hauts-de-France`), S3 expands and drops the region (30% have only
two fields), same typos. 96% of French addresses are exactly three comma fields with no
postcode. A country-agnostic, character-level feature set transfers; anything hard-coded to
US/India patterns breaks on 15% of the test set.

**Less obvious: the test set is 23% denser.** S2+S3 records per S1 entity is 4.677 in train
but 5.754 in test, consistently across all three countries (US 4.674 -> 5.756, India
4.680 -> 5.824, France 5.531). If matches per entity stays at 3.461, the distractor rate
rises from 26% to ~40%; if the distractor rate holds, matches per entity rises to 4.26.
Either way **a threshold tuned on train will be miscalibrated on test**, in opposite
directions depending on which it is. Worth probing with early leaderboard submissions.

## Reference scores

Computed against the real match-count distribution:

| Behaviour | Macro F_0.5 |
|---|---|
| All-empty submission | 0.056 |
| Perfect precision, only 1 match per entity | 0.696 |
| All true matches + 1 false positive per entity | 0.752 |
| Miss 1 true match, never a false positive | 0.871 |
| Perfect precision, up to 3 matches per entity | 0.948 |

## Metric mechanics

With P = tp/p and R = tp/t, precision and recall cancel:

```
F_beta = (1 + b^2) * P * R / (b^2 * P + R) = (1 + b^2) * tp / (b^2 * t + p)
F_0.5  = 1.25 * tp / (0.25 * t + p)
```

Two consequences worth designing around:

**The marginal trade-off is 4:1, not 2:1.** Differentiating gives
`(dF/dP) / (dF/dR) = R^2 / (b^2 * P^2)`, which at P = R is `1 / b^2 = 4`. A unit of precision
is worth four units of recall, so lean harder toward precision than the problem statement's
"2x" wording suggests.

**The optimal threshold is not constant.** Having already emitted `p` candidates of which
`tp` are correct, adding one more with match probability pi is worth it iff

```
pi > tp / (0.25 * t + p) = F_current / 1.25 = 0.8 * F_current
```

The bar rises as an entity accumulates matches: for an entity with 4 true matches it is 0.50
for the second pick, 0.67 for the third, 0.75 for the fourth. A single global threshold
cannot express this, which is why set selection is its own stage. Since `t` is unknown in
practice, the usable form is to sort candidates by probability and pick the set size
maximising expected F_0.5, marginalising over the number of true matches — which also handles
singletons automatically, because when every probability is low the empty set wins. This
requires **calibrated** probabilities, not just a good ranking.
