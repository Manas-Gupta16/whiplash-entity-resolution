"""
Central configuration. Every other module imports paths/params from here —
never hardcode a path or a magic number inline elsewhere.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- Data paths ----
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

# ---- Output paths ----
OUTPUT_DIR = ROOT / "output"
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# ---- Model artifact ----
MODELS_DIR = ROOT / "models"
MATCHER_PATH = MODELS_DIR / "matcher.pkl"

# ---- Validation split ----
# Split by S1 entity_id, never by row, to avoid leakage.
VALIDATION_FRACTION = 0.2
RANDOM_SEED = 42

# ---- Blocking ----
# Max candidates kept per S1 entity after blocking/union, before the matcher scores them.
MAX_CANDIDATES_PER_ENTITY = 40
TFIDF_NGRAM_RANGE = (2, 4)  # character n-grams, good for typos/transliteration
TFIDF_TOP_K = 25            # nearest neighbors pulled from TF-IDF/cosine blocking
MIN_SHARED_TOKENS = 1       # for inverted-index token blocking

# ---- Matching model ----
LGBM_PARAMS = dict(
    objective="binary",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
)

# Decision threshold on predicted match probability. Tune this on the
# validation split to maximize F0.5 — do not leave at 0.5 by default,
# F0.5 is precision-heavy so the optimal cutoff is usually higher.
MATCH_THRESHOLD = 0.5  # placeholder — overwritten by threshold search in train.py
