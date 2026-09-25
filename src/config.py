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
# RANKING UPDATE: candidate_pairs.tsv now counts toward final ranking, separate
# from the matching_results.tsv leaderboard score. A blocking approach that
# produces a SMALLER average candidate set per S1 entity — while still hitting
# the recall target — ranks higher. This shifts the objective from "stay under
# the cap" to "actively minimize candidates per entity without losing recall."
# Treat MAX_CANDIDATES_PER_ENTITY as an upper bound, not a target to fill —
# track and try to reduce the ACTUAL average/median candidate count too.
#
# Dataset scale note: train_source1 ~2.2M rows, train_source2 ~5.0M, train_source3
# ~5.3M (test is similar order of magnitude). A global sklearn NearestNeighbors /
# TF-IDF fit over millions of vectors is NOT viable here (too slow, too much
# memory). Blocking is therefore inverted-index-first: cheap dict lookups that
# scale linearly, with similarity scoring (rapidfuzz) applied only WITHIN the
# small per-entity candidate set produced by the index — never as a global search.

# Max candidates kept per S1 entity after blocking/union, before the matcher scores them.
MAX_CANDIDATES_PER_ENTITY = 40

# Tokens (or n-grams) appearing in more than this fraction of the candidate pool
# are excluded from the inverted index entirely — common words like "inc", "store",
# "the" would otherwise create blocks with hundreds of thousands of entities and
# make blocking effectively useless (and slow). Tune this after looking at the
# actual token document-frequency distribution in Phase 1/2 EDA.
MAX_TOKEN_DF_RATIO = 0.01  # a token in >1% of the pool is treated as a stopword

# Character n-gram length used for the secondary (typo/transliteration-tolerant)
# blocking index. Same max-df exclusion rule applies to n-grams too.
NGRAM_LENGTH = 4
MAX_NGRAM_DF_RATIO = 0.01

# Row-processing batch size for anything iterating S1 entities — keeps memory
# bounded and gives a natural place to log/checkpoint progress on multi-million-row
# runs. Used by generate_candidates() and the feature/predict pipelines.
CANDIDATE_BATCH_SIZE = 50_000

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
