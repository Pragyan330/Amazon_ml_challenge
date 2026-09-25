# Business Entity Resolution — Amazon ML Challenge 2026

Matching noisy business records from three independent sources to a deduplicated reference
source. For every Source-1 entity we must emit the set of Source-2/Source-3 records that
refer to the same real-world business. Scored on **macro F_0.5**, which penalises a wrong
merge roughly four times harder than a missed link.

Challenge statement: `student_resource/README.md`.
Measured properties of the data: [`docs/dataset_analysis.md`](docs/dataset_analysis.md).

## Where we are

Built in stages, simplest first, measuring at each step.

| Stage | Status | Validation macro F_0.5 |
|---|---|---|
| 1. Rule-based blocking + scoring | done | 0.7672 |
| 1b. Stripped pipeline + Jaccard family + channel confidence | done | 0.7981 |
| 1c. Multilingual encoder (LaBSE) for the transliterated tail | done | **0.7998** |
| 2. Gradient-boosted matcher | not started | — |
| 3. Expected-F_0.5 set selection + collective S2<->S3 pass | **next** — targets the 23.61% of true links retrieved but discarded | — |


Measured results and error analysis: [`docs/results.md`](docs/results.md).

Reference points for reading any score: an all-empty submission scores **0.056**, finding
every true match but adding one false positive per entity scores **0.752**, and missing one
true match while never adding a false positive scores **0.871**.

## Layout

```
src/ber/
  normalize.py    Unicode folding, abbreviation expansion, script detection
  similarity.py   Jaccard / Dice / containment / IDF-cosine primitives
  blocking.py     corpus statistics and the Source-1 inverted index
  scorer.py       rule-based pair score (stage 1 matcher)
  select.py       scored candidates -> final predicted set
  evaluate.py     macro F_0.5, matching the challenge definition
  dataio.py       TSV reading and submission writing
  pipeline.py     per-country shard runner
scripts/
  validate.py     held-out validation run with a threshold sweep
docs/
  dataset_analysis.md
output/           generated submission files (gitignored)
artifacts/        cached corpus statistics and scored candidates (gitignored)
```

The dataset itself is **not** committed — it is 2.5 GB. Unzip the provided archive so that
`student_resource/dataset/{train,test}/` exists.

## Running it

Stage 1 needs only the Python standard library (developed against Python 3.14).

```bash
# validation: holds out 1 in 20 training Source-1 entities, scores them against
# all 10.3M training Source-2/3 records, and sweeps the decision threshold
python scripts/validate.py --sample-every 20
```

The first run spends ~20s building corpus token statistics and caches them under
`artifacts/`; later runs reuse that.

## How stage 1 works

**Blocking.** Country is a hard partition — all 191,388 sampled true pairs agree on country
with zero exceptions. Within a country, keys are built from the *rarest* tokens in each
record (keying on `road` or `delhi` produces useless blocks), and any key whose posting list
grows past a cap is dropped as non-discriminative. Both a name path and an address path are
always active: name-token overlap alone reaches only 85.6% recall and address-token overlap
95.6%, but their union reaches 99.98%. The address path is what recovers the 1.76% of true
matches whose name was replaced outright and the 7.3% written in a non-Latin script.

**Scoring.** A weighted blend of name similarity, address similarity and street-number
agreement, with two wrinkles that matter:

- *Evidence-weighted averaging.* Each channel contributes only when that evidence exists. A
  Devanagari name carries no usable signal against a romanised S1 name, and 4.4% of
  true-match records have an empty address; a fixed weighted sum would systematically
  under-score both. We renormalise by the weight actually used.
- *A missing-evidence discount.* Renormalising alone would let one strong signal look as good
  as agreement on everything, so partial evidence is damped.

**Selection.** A single global threshold, tuned by sweep. This is knowingly suboptimal — the
break-even probability for emitting one more match is `0.8 * current_F`, so the bar should
rise as an entity accumulates matches — but establishing the floor comes first. See the
metric derivation in [`docs/dataset_analysis.md`](docs/dataset_analysis.md).

## Fair play

The challenge forbids external databases, APIs and geocoding services for resolving
entities. This pipeline reads only the provided TSVs and never reaches the network. The
abbreviation and state-name tables in `normalize.py` are static string-normalisation
dictionaries (`RD` -> `road`, `TN` -> `tennessee`), not lookups of business identity.
