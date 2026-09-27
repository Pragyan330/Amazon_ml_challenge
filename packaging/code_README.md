# Business Entity Resolution — reproduction guide

Everything needed to regenerate `output/matching_results.tsv` and `output/candidate_pairs.tsv`
from the provided training and test data. The commands in section 3 are the exact ones that
produced the submitted files. Nothing here accesses the network at runtime.

## 1. Environment

Developed and measured on Python 3.14.6, Windows 11, 16 cores / 23 GB RAM.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Only `numpy` and `lightgbm` are needed for the submitted configuration. Set `BER_WORK` to a
directory with ~15 GB free for cached statistics and intermediates (default
`D:\ml_challenge`):

```bash
export BER_WORK=/path/with/space          # Windows: set BER_WORK=D:\ml_challenge
```

## 2. Data layout

Unzip the provided archive so these exist relative to the repository root:

```
student_resource/dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
student_resource/dataset/test/{test_source1,test_source2,test_source3}.tsv
```

## 3. Reproduce end to end

```bash
# (a) blocking + rule scoring on a held-out 1-in-5 training split.
#     Writes $BER_WORK/artifacts/val_scored_s5.pkl and caches corpus statistics.
#     ~32 min single-threaded; produces 441,655 entities / 8.1M candidate pairs.
python src/scripts/validate.py --sample-every 5 \
    --out $BER_WORK/artifacts/val_scored_s5.pkl

# (b) train the matcher on those candidates and evaluate set selection.
#     Writes $BER_WORK/artifacts/matcher_v2.pkl. ~25 min.
python src/scripts/train_matcher.py \
    --scored val_scored_s5.pkl --cross-source --groups 20 \
    --rounds 4000 --leaves 127 --lr 0.04 --threads 10 \
    --out-model matcher_v2.pkl

# (c) generate both submission files for the test set. ~105 min on 5 workers.
python src/scripts/submit.py --workers 5 --chunks-per-country 12 \
    --matcher matcher_v2.pkl --embeddings ""

# (d) check the output against the challenge rules
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test --check-ids

# (e) rebuild this submission package
python src/scripts/build_submission.py --team <team_name>
```

Expected from (b): validation macro F_0.5 **0.9288** on an 88,331-entity held-out split
(India 0.9001, US 0.9479), early stopping around 2,983 rounds.

Expected from (c): 1,732,544 rows, ~5.61M links, ~3.24 per entity, ~4.9% empty, zero records
claimed by more than one entity.

### Notes on the flags

* `--cross-source` is **required** to match the submitted model; without it the features the
  model was trained on are absent and the score drops to ~0.9196.
* `--groups` bounds memory by building records in batches rather than all at once. Lower
  values are faster but use more RAM; 20 peaks around 4.5 GB.
* `--threads 10` leaves cores free so the machine stays usable. `--workers 5` in step (c) is
  chosen the same way; more workers is faster but each holds its own index and statistics.
* Step (c) is resumable — each shard writes its own part file and a rerun skips shards
  already on disk. **`--chunks-per-country` must not change between a run and its resume**,
  because chunk *k* means `entity_id % n_chunks == k`; a different count would make existing
  part files cover different entities while still counting as done.

## 4. Pipeline stages

| Stage | Module | What it does |
|---|---|---|
| Normalisation | `ber/normalize.py` | Unicode folding, static abbreviation and state-name expansion, detection of ten non-Latin scripts |
| Record build | `ber/record.py` | One normalisation pass per record, shared by blocking and scoring; character n-grams built lazily |
| Blocking | `ber/blocking.py` | Country-partitioned inverted index on rarest tokens, with document-frequency and posting-length caps |
| Rule score | `ber/scorer.py` | Evidence-weighted blend of name, address and street-number similarity |
| Features | `ber/pairfeatures.py` | 39 pairwise features: similarity family, per-entity rank, cross-source corroboration |
| Matcher | `scripts/train_matcher.py` | LightGBM binary classifier (MIT) |
| Set selection | `ber/setselect.py` | Exact expected-F_0.5 maximisation per entity |
| Global assignment | `ber/submitworker.py` | Enforces one Source-1 owner per Source-2/3 record, iterated to a fixed point |
| Output | `ber/dataio.py` | Submission TSV writing |

## 5. Optional: multilingual encoder

7.3% of true pairs have a Source-2/3 name in a non-Latin script, where string similarity is
structurally zero. LaBSE (Apache-2.0, 471M) cosine is supported as an extra feature:

```bash
python src/scripts/embed_names.py --split train --s1-every 5
python src/scripts/embed_names.py --split test
# then pass --embeddings to (b) and (c)
```

**Disabled in the submitted configuration.** It is worth +0.0019 macro F_0.5 once the model
is trained with it, and its per-worker id map plus random reads across a 4 GB memmap cost
more memory than that justifies on a 23 GB machine.

If you enable it, the model must be *retrained* with the encoder features present. Training
with them and inferring without them is a silent train/serve mismatch that costs 0.026, not
0.002, because the presence flag acts as a routing signal for non-Latin names.

## 6. Fair play

The challenge forbids external databases, APIs and geocoding services for resolving
entities. This pipeline reads only the provided TSVs and never accesses the network at
runtime. The abbreviation and state-name tables in `normalize.py` (`RD` -> `road`,
`TN` -> `tennessee`) are static string-normalisation dictionaries, not lookups of business
identity, and contain no business records of any kind.

Model licences: LightGBM is MIT. LaBSE, if enabled, is Apache-2.0 and 471M parameters. Both
satisfy the MIT/Apache-2.0 and <=8B parameter requirement.
