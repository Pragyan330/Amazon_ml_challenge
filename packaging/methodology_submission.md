# Business Entity Resolution: methodology

Team: <<< REPLACE WITH TEAM NAME >>>
Members: <<< REPLACE WITH MEMBER NAMES >>>
Date: <<< REPLACE WITH DATE >>>

## 1. Methodology used

The pipeline has three stages. Blocking cuts the search space to a small candidate set per
Source-1 entity. A gradient-boosted classifier scores each surviving pair. A selection stage
then decides, per entity, how many of those candidates to emit.

We began by measuring the data before writing any matcher, because several properties turned
out to decide the design:

Country agrees on 100.000% of true pairs. We checked all 191,388 pairs in a uniform 1-in-40
sample and found no exception, so blocking partitions by country and never compares across
it.

Every matched Source-2/3 record belongs to exactly one Source-1 entity. This holds across all
7,638,365 matched ids in the training ground truth. Matching is therefore a many-to-one
assignment, and two entities claiming the same record is always an error that can be detected
without a model.

39.5% of Source-1 entities share an identical normalised name with another entity in the same
country, and 48.1% share a legal-suffix-stripped core. The largest group is 527 US businesses
called "Meridian". For about half the dataset the name carries little discriminating power
and the address does the work. Ranking same-name candidates by address token Jaccard picks
the right entity 95.55% of the time.

Singletons are statistically invisible. 5.58% of entities have no match, and their rate is
5.583% for US against 5.588% for India with identical name and address lengths to matched
entities. No classifier reading only the Source-1 record can find them, so "no match" has to
come from the absence of good candidates. That pushed us toward a selection stage that can
choose the empty set on its own rather than a separate singleton detector.

The evaluation metric also shaped the design. Writing F_beta with P = tp/p and R = tp/t,
precision and recall cancel:

    F_beta = (1 + b^2) * tp / (b^2 * t + p),  so  F_0.5 = 1.25 * tp / (0.25 * t + p)

Differentiating gives (dF/dP)/(dF/dR) = R^2 / (b^2 * P^2), which equals 4 at P = R. A unit of
precision is worth four units of recall at the balanced point, a stronger asymmetry than the
usual "2x" description suggests, and it set how conservatively we tuned every decision.

Results on a held-out split of 88,331 entities, scored against the full 10.3M Source-2/3
pool:

| Configuration | macro F_0.5 |
|---|---|
| All-empty submission | 0.0558 |
| Rule scorer, best fixed threshold | 0.7986 |
| LightGBM, best fixed threshold | 0.8761 |
| LightGBM with expected-F_0.5 selection | 0.9288 |
| Oracle over the same candidate set | 0.9757 |

Micro precision and recall at the operating point are 0.9775 and 0.8658. By country, US
scores 0.9479 and India 0.9001.

## 2. Candidate generation and blocking strategy

Blocking sets the recall ceiling for everything downstream, so we measured each key family
against the ground truth before choosing:

| Scheme | All | US | India |
|---|---|---|---|
| Name, any shared token | 85.6% | 92.0% | 76.1% |
| Name, any shared 4-gram | 90.8% | 97.7% | 80.7% |
| Address, any shared token | 95.6% | 95.3% | 96.0% |
| Address, 3 rarest tokens | 90.9% | 93.9% | 86.3% |
| Name-4gram OR address-token | 99.98% | 99.98% | 99.97% |

No single family is enough. Name overlap alone caps at 85.6%, so a name path and an address
path are always active together. The address path is what recovers the 1.76% of true matches
whose name has been replaced outright and the 7.27% written in one of ten non-Latin scripts,
where string similarity against a romanised Source-1 name is structurally zero.

Within a country, keys are built from each record's rarest tokens, ordered by document
frequency with an alphabetical tiebreak so both sides of a pair pick the same tokens
independently. Six families are used:

Single-token keys fire only for genuinely rare tokens: up to three address tokens and two
name tokens whose document frequency sits below a cap. Composite keys always fire, because
pairing two tokens is selective by construction even when each is common on its own. We use
name token with address token, which handles a generic business name at a distinct address;
two address tokens, which reaches records whose name is unusable; two name tokens, which
reaches records with an empty address; and a street number paired with the rarest address
token.

Document frequencies come from a 1-in-10 sample of the Source-2/3 pool being searched, about
one million documents, which is enough for stable estimates. Any key whose posting list grows
past a cap is dropped as non-discriminating.

Numeric tokens appear only inside composite keys. An early version keyed on them directly and
common values such as "1" and "100" produced a candidate explosion that reached 13.9 GB of
RAM before we caught it.

On the test set this produced 38,196,413 candidates across 1,732,544 entities, or 22.05 per
entity, from 789,478,266 pairs examined during blocking. Against the full cross product of
1.73M by 9.97M that is a reduction ratio of 2.2e-06, or one pair retained per 452,000
possible. Measured candidate pair recall is 93.3% on a 441,655-entity validation split and
93.45% on a 110,550-entity one. The figure is a property of the method, not of how much
data we sampled.

We also measured what a smaller candidate set would cost, since the organisers rank on it:

| Candidate rule | per entity | oracle | final F_0.5 |
|---|---|---|---|
| Current, prefilter 0.34 | 18.54 | 0.9758 | 0.9197 |
| Top 8 | 6.92 | 0.9705 | 0.9076 |
| Rule score >= 0.50 | 6.72 | 0.9607 | 0.9039 |
| Top 6 and rule score >= 0.50 | 4.50 | 0.9562 | 0.8999 |

Against an earlier rule-based scorer with a fixed threshold, cutting to 4.5 per entity was
free, because that threshold never emitted the lower-ranked candidates anyway. Once the
learned matcher and the selection stage were in place those candidates began to be used, and
the same cut costs between 0.012 and 0.020. We kept the larger set.

## 3. Model architecture and feature engineering

The matcher is a LightGBM binary classifier, MIT licensed, trained on 4.86M labelled pairs
drawn from 441,655 Source-1 entities. It ran 2,983 boosting rounds before early stopping, at
127 leaves and a learning rate of 0.04, reaching 0.99843 AUC on the calibration split.

Splits are entity-disjoint and three-way: train fits the booster, calibration drives early
stopping and the probability calibrator, and test is the only split any reported number comes
from. We split on entities rather than pairs because one entity's candidates are not
independent of each other.

### Features

Thirty-nine features per pair, in four groups.

Similarity measures cover name and address separately: token Jaccard, Dice, containment,
IDF-weighted generalised Jaccard, and character trigram Jaccard and Dice, plus the same
family computed on tokens with legal suffixes and generic category words removed. We prefer
IDF-weighted Jaccard to cosine because cosine normalises each side independently and so
rewards a single rare shared token even when the rest of the record disagrees, while the
Jaccard form charges for every unmatched token. Address character n-grams are taken over
sorted tokens, so Source 3's habit of reordering components still matches.

Presence flags record which evidence existed. Each channel carries an explicit flag so the
model can distinguish "no address to compare" from "addresses disagree", and street numbers
carry a separate flag separating "no numbers present" from "numbers present and different".

Per-entity competition features describe how a candidate ranks against the other candidates
for the same entity: rank, gap to the best candidate, ratio to the best, candidate count, and
deviation from the entity mean. Matching is a contest within an entity, where a 0.6 score
means something different as the best candidate than as the fifth, and no absolute feature
can express that. Rank is the second most important feature by gain.

Cross-source corroboration compares each candidate against its rivals for the same entity,
split by source. In the training data a Source-2 and a Source-3 record matching the same
entity have blended name and address similarity averaging 0.447, while records matching
different entities in the same country average 0.015. None of the cross-cluster pairs exceed
0.5, against 42.7% of the within-cluster ones. A purely pairwise scorer cannot see this,
because it only ever compares a candidate against the Source-1 record. Adding four features
built on it was worth +0.0070 macro F_0.5. They are computed from a compact signature of
sorted 32-bit token hashes, 108 bytes against 1,686 for a full normalised record, which keeps
the Jaccard exact. A MinHash sketch would have been smaller but approximate, and the feature
has to mean the same thing during training and at inference.

The rule score feeds in as a feature and is the most important one by a factor of eight. It
is a weighted blend of the three channels with two properties worth describing. Channels
contribute only when their evidence exists, and the result is renormalised by the weight
actually used. The blend is then scaled by a confidence term indexed by which channels were
present, which we set from measurement: a perfect name match with no address is 12.9 times
more likely to be a false merge than a true match (10,194 positives against 131,231
negatives), while a perfect address match with no usable name is 5.2 times more likely to be
correct. A single symmetric missing-data penalty prices those two cases almost identically.

### Set selection

The model's probabilities feed a selection stage rather than a threshold. From the reduced
form of the metric, the break-even probability for emitting one more candidate when tp of p
emitted are already correct is tp / (0.25 * t + p), which equals F_current / 1.25. The bar
rises as an entity accumulates matches, so no single global number can express the right
policy.

Since t is unknown at prediction time, we treat each candidate's label as an independent
Bernoulli draw and choose the set size maximising expected F_0.5. With candidates sorted by
probability, selecting the top n gives tp ~ PoissonBinomial(p_1..p_n) and the unselected true
matches fn ~ PoissonBinomial(p_n+1..p_k), independent, so

    E[F | n] = 1.25 * sum_a sum_b  P(tp=a) P(fn=b) * a / (0.25 (a+b) + n)

We compute this exactly by convolution, including E[F | n=0] = P(t=0) for the empty set, and
verified it against Monte Carlo to a maximum error of 0.0013 against roughly 0.002 sampling
noise. The common shortcut, a ratio of expectations, is biased by Jensen's inequality and is
usually paired with an exact P(t=0), so the empty-versus-not comparison mixes an approximate
value against an exact one at exactly the singleton decision, where an error costs a full 1.0.

This stage was worth +0.0527 over the best achievable fixed threshold on an identical model,
the largest single improvement in the project, and it needed no extra features and about four
seconds of compute. It also repaired singleton handling on its own: entities correctly left
empty rose from 68.80% to 83.64%, because an entity whose candidates are all weak now chooses
the empty set instead of being dragged over a global threshold.

### Global assignment

Because each Source-2/3 record may belong to only one Source-1 entity, selection is resolved
globally after all shards finish. Each contested record is awarded to its most confident
claimant, and the entities that lose one re-select from what remains rather than simply
having it deleted, since losing a member changes how many of the rest are worth emitting.

This iterates to a fixed point. Re-selection promotes new candidates that can collide with
other entities' choices, so a single pass is not enough: on the test set, 59,328 records were
contested initially and five passes were needed to reach zero, removing 111,927 links. An
earlier single-pass version left 4,411 contested.

The effect cannot be measured on a subsampled validation split at all. With 1-in-20 of
Source 1, two entities competing for the same record almost never both appear, and we
observed 8 contested records there against 59,328 at full scale.

## 4. Other relevant information

### Final output

1,732,544 rows, one per test Source-1 entity. 5,613,117 links, 3.24 per entity, with 85,389
entities predicted empty (4.9%, against a true singleton rate of 5.58% in training). No
Source-2/3 record is claimed by more than one entity. Both files pass the provided validator
including the optional ID-existence check.

### Distribution shift we designed around

France is 15% of the test set and absent from training. It follows the same corruption
process with French vocabulary, so we kept every feature country-agnostic and character-level
and avoided hard-coding anything to US or India patterns, which would have failed on 15% of
entities. Our validation figures carry no evidence about France, and we have treated that as
the main uncertainty in our own estimates.

The test set also carries 23% more Source-2/3 records per Source-1 entity than training
(5.754 against 4.677, consistently across all three countries), so either the distractor rate
rises to about 40% or matches per entity rise to about 4.26. Either way a threshold tuned on
training is miscalibrated on test, which is a further argument for per-entity selection over
a global threshold.

### Measurement practice

Where a measurement and an assumption disagreed, we followed the measurement, and several
design choices came from being wrong first. Two examples that changed the pipeline:

We ran a model trained with multilingual encoder features without those features at
inference, reasoning that 93% of training pairs already carried zeros there. That cost 0.026
macro F_0.5, not the 0.002 we expected, because the presence flag acted as a routing signal
for non-Latin names and its loss fell almost entirely on India. The encoder is disabled in
the submitted configuration and the model is trained without those features, so training and
inference match.

Our first scorer took the maximum over several similarity views per channel. That is
optimistic by construction, since a negative pair gets credit from whichever view is most
generous, and it degraded separation badly: negatives drifted to a mean of 0.512 against
positives at 0.906, and the optimal threshold ended up sitting on a discontinuity where a
single score value held 131,231 false positives. Weighted combination replaced it.

### Compute and reproducibility

The pipeline runs on CPU and needs only numpy and lightgbm. Blocking and scoring shard across
processes by Source-1 slice, with each worker streaming the Source-2/3 pool for its country
once. The full test run takes about 105 minutes on five workers on a 16-core machine with
23 GB of RAM, and is resumable at shard granularity. Memory is bounded by construction:
feature rows are written into preallocated float32 arrays, and records are built in groups so
neither side is ever held whole.

The submitted code reproduces both output files end to end from the provided data, with the
exact commands in the package README.

### Fair play and licences

The pipeline reads only the provided TSV files and never accesses the network at runtime. We
used no external database, API, geocoding service or any other outside source to resolve
entities. The abbreviation and state-name tables in the normalisation module (RD to road, TN
to tennessee) are static string-normalisation dictionaries and contain no business records.

LightGBM is MIT licensed. LaBSE, which is supported in the code but disabled in the submitted
configuration, is Apache-2.0 and has 471M parameters. Both satisfy the MIT or Apache-2.0
requirement and the 8B parameter limit.

### Known limitations

The candidate set is the ceiling. Our oracle over it is 0.9757, so 6.55% of true links never
reach the matcher and no improvement to the model can recover them. The measured union
ceiling for name-4gram OR address-token is 99.98%, so that recall exists and is being lost in
the prefilter and the document-frequency caps. That is where we would go next.

India trails US by about 0.048, concentrated in transliterated names and in generic business
names at addresses that degrade to city and state only.
