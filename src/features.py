"""
Pairwise similarity features between an S1 record and one candidate record.
Used both to rank/trim blocking output and as the matcher's input vector —
keep this the single source of truth so both stages stay consistent.
"""
import rapidfuzz.fuzz as fuzz
import jellyfish
import pandas as pd


def jaccard_tokens(a: str, b: str) -> float:
    set_a, set_b = set(a.split()), set(b.split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def pair_features(row_s1: pd.Series, row_cand: pd.Series) -> dict:
    """
    Build the feature dict for one (S1, candidate) pair.
    row_s1 / row_cand must already have *_norm columns from preprocess.py.

    TODO: extend with address-component overlap features once address
    parsing is added in preprocess.py, and with a country-match flag
    (treat as arbitrary string equality, not an enumerated category).
    """
    name_a, name_b = row_s1["business_name_norm"], row_cand["business_name_norm"]
    addr_a, addr_b = row_s1["business_address_norm"], row_cand["business_address_norm"]

    return {
        "name_jaccard": jaccard_tokens(name_a, name_b),
        "name_levenshtein_ratio": fuzz.ratio(name_a, name_b) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(name_a, name_b) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(name_a, name_b) / 100.0,
        "name_jaro_winkler": jellyfish.jaro_winkler_similarity(name_a, name_b),
        "name_len_diff": abs(len(name_a) - len(name_b)),
        "addr_jaccard": jaccard_tokens(addr_a, addr_b),
        "addr_levenshtein_ratio": fuzz.ratio(addr_a, addr_b) / 100.0,
        "addr_token_sort_ratio": fuzz.token_sort_ratio(addr_a, addr_b) / 100.0,
        "country_match": int(row_s1.get("country", "") == row_cand.get("country", "")),
    }


def build_feature_matrix(pairs_df: pd.DataFrame, s1_lookup: dict, cand_lookup: dict) -> pd.DataFrame:
    """
    pairs_df: columns source1_entity_id, candidate_entity_id (long format, one row per pair)
    s1_lookup / cand_lookup: entity_id -> row (Series), for fast access
    Returns a DataFrame of features, one row per pair, aligned to pairs_df's index.

    TODO: implement using pair_features() above, ideally vectorized/batched
    for speed once candidate volume is known from blocking.
    """
    raise NotImplementedError("Fill in per the plan doc, Phase 3")
