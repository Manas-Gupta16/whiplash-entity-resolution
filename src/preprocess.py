"""
Normalization utilities for business names and addresses.
Keep BOTH raw and normalized text available downstream — some features
(typo-distance) work better on raw strings, others (token overlap) on normalized.
"""
import re
import pandas as pd

# Expanded from EDA on train_ground_truth matched pairs.
LEGAL_SUFFIX_MAP = {
    "corporation": "corp",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "company": "co",
    "&": "and",
    "l.l.c.": "llc",
    "l.l.p.": "llp",
    "pllc": "llc"
}

# Expanded from EDA — India vs US address abbreviation patterns differ.
ADDRESS_ABBR_MAP = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "circle": "cir",
    "drive": "dr",
    "terrace": "ter",
    "court": "ct",
    "lane": "ln",
    "maharashtra": "mh",
    "texas": "tx",
    "tennessee": "tn",
    "maryland": "md",
    "illinois": "il",
    "north carolina": "nc",
    "arizona": "az",
    "washington": "wa",
    "new york": "ny",
    "california": "ca",
    "arkansas": "ar",
    "wisconsin": "wi",
    "andhra pradesh": "ap",
    "uttar pradesh": "up",
    "telangana": "tg"
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
    
    # Vectorized name normalization
    name_norm = df["business_name"].str.lower().str.strip()
    name_norm = name_norm.str.replace(r"[^\w\s&]", " ", regex=True)
    for long_form, short_form in LEGAL_SUFFIX_MAP.items():
        name_norm = name_norm.str.replace(rf"\b{long_form}\b", short_form, regex=True)
    name_norm = name_norm.str.replace(r"\s+", " ", regex=True).str.strip()
    df["business_name_norm"] = name_norm
    
    # Vectorized address normalization
    addr_norm = df["business_address"].str.lower().str.strip()
    addr_norm = addr_norm.str.replace(r"[^\w\s]", " ", regex=True)
    for long_form, short_form in ADDRESS_ABBR_MAP.items():
        addr_norm = addr_norm.str.replace(rf"\b{long_form}\b", short_form, regex=True)
    addr_norm = addr_norm.str.replace(r"\s+", " ", regex=True).str.strip()
    df["business_address_norm"] = addr_norm
    
    return df


def get_normalization_cache_key() -> str:
    """
    Generate an MD5 fingerprint of current normalization dictionaries.
    If LEGAL_SUFFIX_MAP or ADDRESS_ABBR_MAP are modified, this key changes,
    ensuring that outdated/stale caches are automatically invalidated.
    """
    import hashlib
    import json
    data = {
        "legal": sorted(LEGAL_SUFFIX_MAP.items()),
        "addr": sorted(ADDRESS_ABBR_MAP.items()),
        "version": "1.0"
    }
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()[:10]

