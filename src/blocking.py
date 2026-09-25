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

    Uses vectorized pandas (explode + groupby) instead of
    a pure Python loop — 20-50x faster at 10M+ row scale.
    """
    total = len(df)
    print(f"  Vectorized index build for {total:,} rows...")

    # Apply tokenizer to produce unique tokens per row, then explode
    tokens_series = df[text_col].map(lambda t: set(tokenizer(t)))
    exploded = (
        pd.DataFrame({"entity_id": df["entity_id"].values, "token": tokens_series})
        .explode("token")
    )
    exploded = exploded[exploded["token"].notna() & (exploded["token"] != "")]

    # Drop overly common tokens (document frequency > max_df_ratio * total)
    max_df = max_df_ratio * total
    token_counts = exploded["token"].value_counts()
    valid_tokens = token_counts[token_counts <= max_df].index
    exploded = exploded[exploded["token"].isin(valid_tokens)]

    # Build dict: token -> set of entity_ids
    index = exploded.groupby("token")["entity_id"].agg(set).to_dict()
    print(f"  Done. {len(index):,} tokens in index (after df filter).")
    return index


def build_token_index(df: pd.DataFrame, text_col: str = "business_name_norm") -> dict:
    """Word-token inverted index, with overly common tokens excluded."""
    def tokenize(text):
        return [t for t in str(text).split() if len(t) >= 2]
    return build_inverted_index(df, text_col, tokenize, config.MAX_TOKEN_DF_RATIO)


def build_ngram_index(df: pd.DataFrame, text_col: str = "business_name_norm") -> dict:
    """Character n-gram inverted index — catches typos/transliteration that
    share no exact word tokens. NOTE: at 10M+ rows this still takes several
    minutes even vectorized — use_ngrams=False by default."""
    def tokenize(text):
        return list(_char_ngrams(str(text), config.NGRAM_LENGTH))
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

def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    token_index: dict = None,
    ngram_index: dict = None,
    pool_lookup: dict = None,
    use_ngrams: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      source1_entity_id, candidate_entity_ids (list[str])

    use_ngrams: set True to add the n-gram index as a second blocking signal.
    At full scale (10M+ rows), building the n-gram index takes 30-60 min in
    pure Python — leave False until the token-only run is validated and
    indices are cached to disk. Token blocking alone achieves strong recall
    once normalization is good, and rapidfuzz reranking covers most remaining
    typos within the returned candidate set.
    """
    if token_index is None or pool_lookup is None:
        print("Building combined pool...")
        pool_df = pd.concat([s2_df, s3_df], ignore_index=True)

        print("Building token index (name)...")
        token_index = build_token_index(pool_df, "business_name_norm")

        if use_ngrams:
            print("Building ngram index (name) — this is slow at full scale...")
            ngram_index = build_ngram_index(pool_df, "business_name_norm")

        print("Building pool lookup dictionary...")
        pool_lookup = dict(zip(pool_df["entity_id"], pool_df["business_name_norm"]))
    
    results = []
    
    print("Generating candidates for S1 entities...")
    ent_idx = s1_df.columns.get_loc("entity_id") + 1
    name_norm_idx = s1_df.columns.get_loc("business_name_norm") + 1
    
    total = len(s1_df)
    batch_size = config.CANDIDATE_BATCH_SIZE
    
    for i, row in enumerate(s1_df.itertuples()):
        s1_id = row[ent_idx]
        name_norm = row[name_norm_idx]

        cands = token_candidates(name_norm, token_index)
        if ngram_index is not None:
            cands |= ngram_candidates(name_norm, ngram_index)
        
        if len(cands) > config.MAX_CANDIDATES_PER_ENTITY:
            cands_list = rerank_candidates(name_norm, cands, pool_lookup, config.MAX_CANDIDATES_PER_ENTITY)
        else:
            cands_list = list(cands)
            
        results.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": cands_list
        })
        
        if (i + 1) % batch_size == 0:
            print(f"Processed {i + 1} / {total} entities...")
            
    print(f"Processed {total} / {total} entities. Done.")
    
    return pd.DataFrame(results)
