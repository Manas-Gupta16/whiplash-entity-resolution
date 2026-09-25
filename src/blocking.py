"""
Candidate generation ("blocking"). This determines the recall ceiling of the
whole pipeline — a true match dropped here can never be recovered later.

Strategy: union of several cheap blocking signals, then cap per S1 entity.
  1. Token-based inverted index on normalized business_name
  2. Character n-gram TF-IDF + nearest neighbors (catches typos/transliteration)
  3. (optional) phonetic code blocking for heavy name distortion

IMPORTANT: never filter/group by a hardcoded country set — test includes
France, which never appears in train. Treat country as an open string field.
"""
from collections import defaultdict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from . import config


def build_token_index(df: pd.DataFrame, text_col: str = "business_name_norm") -> dict:
    """entity_id -> tokens, and token -> set(entity_id), for inverted-index lookup."""
    token_to_ids = defaultdict(set)
    for entity_id, text in zip(df["entity_id"], df[text_col]):
        for tok in set(text.split()):
            if len(tok) < 2:
                continue
            token_to_ids[tok].add(entity_id)
    return token_to_ids


def token_candidates(s1_row, token_to_ids: dict) -> set:
    """Candidates sharing >= config.MIN_SHARED_TOKENS tokens with the S1 name."""
    tokens = set(s1_row["business_name_norm"].split())
    candidates = set()
    for tok in tokens:
        candidates |= token_to_ids.get(tok, set())
    return candidates


def build_tfidf_index(df: pd.DataFrame, text_col: str = "business_name_norm"):
    """Fit a char n-gram TF-IDF vectorizer + NearestNeighbors index over df."""
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=config.TFIDF_NGRAM_RANGE, min_df=1
    )
    matrix = vectorizer.fit_transform(df[text_col])
    nn = NearestNeighbors(n_neighbors=min(config.TFIDF_TOP_K, len(df)), metric="cosine")
    nn.fit(matrix)
    return vectorizer, nn, matrix


def tfidf_candidates(s1_text: str, vectorizer, nn, df: pd.DataFrame) -> list:
    """Top-k nearest entity_ids by cosine similarity on char n-grams."""
    vec = vectorizer.transform([s1_text])
    _, indices = nn.kneighbors(vec)
    return df.iloc[indices[0]]["entity_id"].tolist()


def generate_candidates(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point. Returns a DataFrame with columns:
      source1_entity_id, candidate_entity_ids (list[str])
    This is the exact set later fed to the matcher, and written to candidate_pairs.tsv.

    TODO:
      - build token + tfidf indices per source (S2, S3 combined pool)
      - union token_candidates + tfidf_candidates for each S1 row
      - cap at config.MAX_CANDIDATES_PER_ENTITY (keep highest-similarity ones if trimming)
      - measure blocking recall against train_ground_truth during development
    """
    raise NotImplementedError("Fill in per the plan doc, Phase 2")
