# Business Entity Resolution — Amazon ML Challenge 2026

Two-stage pipeline: **blocking** (candidate generation) → **matching** (LightGBM pairwise classifier),
tuned for F0.5 (precision-heavy).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Drop the provided competition files into:
```
data/train/train_source1.tsv
data/train/train_source2.tsv
data/train/train_source3.tsv
data/train/train_ground_truth.tsv
data/test/test_source1.tsv
data/test/test_source2.tsv
data/test/test_source3.tsv
```

Drop the organizer-provided `utils/validate_submission.py` into `utils/`.

## Run

```bash
# EDA (optional, in notebooks/eda.ipynb)
jupyter notebook notebooks/eda.ipynb

# Train the matcher + tune threshold on a held-out validation split
python -m src.train

# Generate predictions on the test set
python -m src.predict

# Validate output format before every leaderboard upload
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir data/test
```

## Repo layout

```
data/            # not committed - train/ and test/ TSVs go here
output/          # matching_results.tsv + candidate_pairs.tsv (generated)
src/             # pipeline source
  config.py      # all paths/hyperparams - single source of truth
  preprocess.py  # name/address normalization
  blocking.py    # candidate generation
  features.py    # pairwise similarity features
  train.py       # train + threshold-tune the matcher
  predict.py     # run pipeline on test, write submission files
  evaluate.py    # local F0.5 macro-average scorer
models/          # trained matcher artifact (not committed)
notebooks/       # EDA
code/business_entity_resolution/  # final zip package mirror (see plan doc)
utils/           # organizer's validate_submission.py goes here
```

See `IMPLEMENTATION_PLAN.md` for the phase-by-phase build plan.
