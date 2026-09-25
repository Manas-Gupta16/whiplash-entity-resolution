"""
Normalization utilities for business names and addresses.
Keep BOTH raw and normalized text available downstream — some features
(typo-distance) work better on raw strings, others (token overlap) on normalized.
"""
import re
import pandas as pd

# TODO: expand this list from real EDA on train_ground_truth matched pairs.
LEGAL_SUFFIX_MAP = {
    "corporation": "corp",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "company": "co",
    "&": "and",
}

# TODO: expand from EDA — India vs US address abbreviation patterns differ.
ADDRESS_ABBR_MAP = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
}


def load_source(path) -> pd.DataFrame:
    """Load a source TSV with explicit sep='\t'. Never omit sep."""
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, expand/standardize legal suffixes."""
    text = name.lower().strip()
    text = re.sub(r"[^\w\s&]", " ", text)
    for long_form, short_form in LEGAL_SUFFIX_MAP.items():
        text = re.sub(rf"\b{long_form}\b", short_form, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_address(address: str) -> str:
    """Lowercase, strip punctuation, expand common abbreviations.
    TODO: split into components (city/state/pin/landmark) where regex allows —
    do this per-country pattern, discovered from EDA, not hardcoded to {US, India}.
    """
    text = address.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    for long_form, short_form in ADDRESS_ABBR_MAP.items():
        text = re.sub(rf"\b{long_form}\b", short_form, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["business_name_norm"] = df["business_name"].map(normalize_name)
    df["business_address_norm"] = df["business_address"].map(normalize_address)
    return df
