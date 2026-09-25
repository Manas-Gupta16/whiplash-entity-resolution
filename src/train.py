"""
Train the pairwise matcher on train_ground_truth.tsv.

Pipeline:
  1. Load + normalize S1/S2/S3 train sources (preprocess.py)
  2. Split S1 entities into train/val (config.VALIDATION_FRACTION) - split by
     entity, never by row, to avoid leakage
  3. Run blocking on the train split to get candidate pairs (blocking.py)
  4. Label each candidate pair: positive if in ground truth, negative otherwise
     (these blocking-derived negatives are your hard negatives - far better
     than random negatives)
  5. Build feature matrix (features.py)
  6. Train LightGBM classifier (config.LGBM_PARAMS)
  7. Run blocking + predict on the val split, sweep decision thresholds,
     pick the one maximizing macro F0.5 (evaluate.py) -> save as config.MATCH_THRESHOLD
  8. Save model artifact to config.MATCHER_PATH

TODO: implement end to end. Log blocking recall, feature importances,
and val F0.5 at the chosen threshold - all three belong in the methodology doc.
"""
from . import config

def main():
    raise NotImplementedError("Fill in per the plan doc, Phase 3-4")

if __name__ == "__main__":
    main()
