# Business Entity Resolution — reproduction guide

Everything needed to regenerate `output/matching_results.tsv` and `output/candidate_pairs.tsv`
from the provided training and test data. No network access is used at any point.

## 1. Environment

Developed and measured on Python 3.14.6, Windows 11, 16 cores / 23 GB RAM, RTX 5050 (8 GB).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

The matcher (`lightgbm`) and the pipeline (`numpy`) are the only runtime requirements. The
multilingual encoder is optional and used only for a precompute step — see §5.

Set `BER_WORK` to a directory with ~10 GB free for cached statistics and intermediates
(defaults to `D:\ml_challenge`):

```bash
export BER_WORK=/path/with/space          # Windows: set BER_WORK=D:\ml_challenge
```

## 2. Data layout

Unzip the provided archive so the following exist, relative to the repository root:

```
student_resource/dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
student_resource/dataset/test/{test_source1,test_source2,test_source3}.tsv
```

## 3. Reproduce end to end

```bash
# (a) blocking + rule scoring on a held-out training split, and the threshold sweep.
#     Writes $BER_WORK/artifacts/val_scored.pkl and caches corpus statistics.
#     ~15 min single-threaded.
python src/scripts/validate.py --sample-every 20

# (b) train the matcher on those candidates and evaluate set selection.
#     Writes $BER_WORK/artifacts/matcher.pkl. ~5 min.
python src/scripts/train_matcher.py --scored val_scored.pkl --embeddings ""

# (c) generate both submission files for the test set. ~60 min on 5 workers.
python src/scripts/submit.py --workers 5 --chunks-per-country 12 \
    --matcher matcher.pkl --embeddings ""

# (d) check the output against the challenge rules
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

Step (c) is resumable: each shard writes its own part file and a rerun skips shards already
on disk. `--chunks-per-country` must not change between a run and its resume, because chunk
*k* means `entity_id % n_chunks == k` — a different chunk count would make existing part
files cover different entities while still counting as done.

## 4. Pipeline stages

| Stage | Module | What it does |
|---|---|---|
| Normalisation | `ber/normalize.py` | Unicode folding, static abbreviation and state-name expansion, detection of ten non-Latin scripts |
| Record build | `ber/record.py` | One normalisation pass per record, shared by blocking and scoring; character n-grams built lazily |
| Blocking | `ber/blocking.py` | Country-partitioned inverted index on rarest tokens, with document-frequency and posting-length caps |
| Rule score | `ber/scorer.py` | Evidence-weighted blend of name, address and street-number similarity |
| Features | `ber/pairfeatures.py` | 35 pairwise features including per-entity rank |
| Matcher | `ber/scripts/train_matcher.py` | LightGBM binary classifier (MIT) |
| Set selection | `ber/setselect.py` | Exact expected-F0.5 maximisation per entity |
| Output | `ber/dataio.py` | Submission TSV writing |

## 5. Optional: multilingual encoder

7.3% of true pairs have a Source-2/3 name in a non-Latin script, where string similarity is
structurally zero. LaBSE (Apache-2.0, 471M) cosine is supported as an extra feature:

```bash
python src/scripts/embed_names.py --split train --s1-every 20   # ~3 min on GPU
python src/scripts/embed_names.py --split test                  # ~8 min on GPU
# then pass --embeddings train_s1every20_names / test_names to (b) and (c)
```

It is **disabled in the submitted configuration**. It is worth +0.0019 macro F_0.5 on
validation, and the per-worker id map plus random reads across a 4 GB memmap cost more
memory than the gain justifies on a 23 GB machine. Only Source-1 and non-Latin Source-2/3
names are ever embedded, so ~93% of pairs already carry a zero encoder feature and disabling
it introduces no train/inference mismatch.

## 6. Fair play

The challenge forbids external databases, APIs and geocoding services for resolving
entities. This pipeline reads only the provided TSVs and never accesses the network at
runtime. The abbreviation and state-name tables in `normalize.py` (`RD` -> `road`,
`TN` -> `tennessee`) are static string-normalisation dictionaries, not lookups of business
identity, and contain no business records of any kind.

Model licences: LightGBM is MIT; LaBSE, if enabled, is Apache-2.0 and 471M parameters. Both
satisfy the MIT/Apache-2.0 and <=8B parameter requirement.
