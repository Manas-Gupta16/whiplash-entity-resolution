"""
Candidate generation ("blocking"). This determines the recall ceiling of the
whole pipeline — a true match dropped here can never be recovered later.

SCALE NOTE — read this before touching anything below:
train_source1 ~2.2M rows, train_source2 ~5.0M, train_source3 ~5.3M (test is
similar order of magnitude). A global nearest-neighbor / TF-IDF search over
millions of vectors (e.g. sklearn.NearestNeighbors.fit() on the whole pool) is
NOT viable — too slow and too much memory. This module is therefore built
around cheap, linear-time inverted-index lookups:

  1. Token-based inverted index on normalized business_name (dict lookups, O(1)
     average per token) — the primary blocking signal.
  2. Character n-gram inverted index (same mechanism) as a secondary net for
     typos/transliteration that share no exact tokens.
  3. Similarity scoring (rapidfuzz) is applied ONLY within the small per-entity
     candidate set an index lookup returns — never as a search over the full pool.

Tokens/n-grams that appear in a very large fraction of the pool (e.g. "inc",
"store", "the") are excluded from the index entirely (config.MAX_TOKEN_DF_RATIO /
MAX_NGRAM_DF_RATIO) — otherwise a single common token creates a block with
hundreds of thousands of candidates and blocking stops doing its job.

IMPORTANT: never filter/group by a hardcoded country set — test includes
France, which never appears in train. Country is used only as an optional
soft tie-breaker in reranking, never as a hard filter.
"""
from collections import defaultdict
import pandas as pd
from rapidfuzz import fuzz, process

from . import config


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int) -> set:
    text = text.replace(" ", "")
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def build_inverted_index(
    df: pd.DataFrame,
    text_col: str,
    tokenizer,
    max_df_ratio: float,
) -> dict:
    """
    Generic inverted index: token -> set(entity_id).
    `tokenizer` turns a normalized string into an iterable of tokens
    (word tokens for the name index, char n-grams for the n-gram index).
    Tokens whose document frequency exceeds max_df_ratio * len(df) are
    dropped after the pass — they're too common to be useful for blocking
    and would otherwise blow up candidate set sizes.
    """
    index = defaultdict(set)
    for entity_id, text in zip(df["entity_id"], df[text_col]):
        for tok in tokenizer(text):
            if tok:
                index[tok].add(entity_id)

    max_df = max_df_ratio * len(df)
    dropped = [tok for tok, ids in index.items() if len(ids) > max_df]
    for tok in dropped:
        del index[tok]

    return index


def build_token_index(df: pd.DataFrame, text_col: str = "business_name_norm") -> dict:
    """Word-token inverted index, with overly common tokens excluded."""
    def tokenize(text):
        return (t for t in text.split() if len(t) >= 2)

    return build_inverted_index(df, text_col, tokenize, config.MAX_TOKEN_DF_RATIO)


def build_ngram_index(df: pd.DataFrame, text_col: str = "business_name_norm") -> dict:
    """Character n-gram inverted index — catches typos/transliteration that
    share no exact word tokens."""
    def tokenize(text):
        return _char_ngrams(text, config.NGRAM_LENGTH)

    return build_inverted_index(df, text_col, tokenize, config.MAX_NGRAM_DF_RATIO)


# ---------------------------------------------------------------------------
# Per-entity candidate lookup
# ---------------------------------------------------------------------------

def token_candidates(name_norm: str, token_index: dict) -> set:
    """Union of entity_ids sharing any surviving (non-stopword) token."""
    candidates = set()
    for tok in name_norm.split():
        if tok in token_index:
            candidates |= token_index[tok]
    return candidates


def ngram_candidates(name_norm: str, ngram_index: dict) -> set:
    """Union of entity_ids sharing any surviving character n-gram."""
    candidates = set()
    for gram in _char_ngrams(name_norm, config.NGRAM_LENGTH):
        if gram in ngram_index:
            candidates |= ngram_index[gram]
    return candidates


# ---------------------------------------------------------------------------
# Reranking / trimming — only ever runs over a small (already-blocked) set
# ---------------------------------------------------------------------------

def rerank_candidates(name_norm: str, candidate_ids: set, pool_lookup: dict, top_k: int) -> list:
    """
    Trim a candidate set down to top_k by name similarity, using rapidfuzz
    (fast C implementation) over this small subset only — this is safe at
    any pool size because candidate_ids is already small after indexing.
    pool_lookup: entity_id -> normalized name string.
    Returns a list of entity_ids, most similar first.
    """
    if len(candidate_ids) <= top_k:
        return list(candidate_ids)

    choices = {cid: pool_lookup[cid] for cid in candidate_ids if cid in pool_lookup}
    results = process.extract(
        name_norm, choices, scorer=fuzz.token_sort_ratio, limit=top_k
    )
    # process.extract returns (match_string, score, key) tuples
    return [key for _, _score, key in results]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_candidates(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      source1_entity_id, candidate_entity_ids (list[str])
    This is the exact set later fed to the matcher, and written to candidate_pairs.tsv.

    Processes s1_df in batches (config.CANDIDATE_BATCH_SIZE) so progress can be
    logged/checkpointed on multi-million-row runs — do not attempt to hold
    intermediate per-entity candidate lists for the full 2M+ rows in memory at
    once without batching if you extend this further.

    Steps:
      1. Build a combined S2+S3 pool with normalized name/address (preprocess.py
         should already have been run on all three inputs before calling this).
      2. Build token index + n-gram index over the pool.
      3. For each S1 entity: union(token_candidates, ngram_candidates).
      4. If the union exceeds config.MAX_CANDIDATES_PER_ENTITY, rerank with
         rapidfuzz and keep the top K.
      5. Track blocking recall on the labeled (train) split during development —
         this function itself doesn't need ground truth, recall measurement
         happens in evaluate.py / train.py against train_ground_truth.tsv.

    TODO: implement steps 1–4 following the helper functions already defined
    above in this file (build_token_index, build_ngram_index, token_candidates,
    ngram_candidates, rerank_candidates). Use itertuples() rather than iterrows()
    when looping over s1_df for speed at this scale.
    """
    raise NotImplementedError("Fill in per the plan doc, Phase 2 (scale-aware version)")
