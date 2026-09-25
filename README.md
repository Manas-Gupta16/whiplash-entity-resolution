# Business Entity Resolution - Amazon ML Challenge 2026

## Overview
This repository contains the solution for the Amazon ML Challenge 2026: Business Entity Resolution. The objective is to accurately link noisy, inconsistent business records across three independent data sources. Source 1 serves as the deduplicated reference dataset, and the goal is to identify all corresponding matches in Source 2 and Source 3 for every entity in Source 1. 

The evaluation metric is the macro-averaged F0.5 score, which heavily penalizes false positives (incorrect merges).

## Architecture
Given the scale of the dataset (tens of millions of records), an exhaustive pairwise comparison is computationally infeasible. Our pipeline is structured as a two-stage process:

1. **Candidate Generation (Blocking)**
   - Utilizes inverted indices (token and character n-gram based) with maximum document frequency cutoffs to exclude common stopwords.
   - Casts a wide but computationally inexpensive net to retrieve a small subset of plausible candidates for each Source 1 entity.
   - Trims excessively large candidate sets using rapid string similarity heuristics.

2. **Pairwise Classification**
   - Extracts detailed engineered features (Jaccard, Levenshtein, Jaro-Winkler, and token-sort distances) from the generated candidate pairs.
   - Employs a Gradient Boosted Decision Tree (LightGBM) to classify each pair as a match or non-match.
   - Applies an optimized decision threshold tuned specifically for the F0.5 metric to maximize precision while preserving recall.

## Repository Structure
```text
business_entity_resolution/
├── data/
│   ├── train/            # Training datasets (sources 1, 2, 3, and ground truth)
│   └── test/             # Test datasets (sources 1, 2, 3)
├── models/               # Serialized model artifacts (LightGBM)
├── notebooks/            # Exploratory Data Analysis and submission logs
├── output/               # Final generated TSV files (matching_results.tsv, candidate_pairs.tsv)
├── src/                  # Core pipeline modules
│   ├── config.py         # Global hyperparameter and path configurations
│   ├── preprocess.py     # String normalization and abbreviation mapping
│   ├── blocking.py       # Inverted index construction and candidate generation
│   ├── features.py       # Pairwise string similarity feature engineering
│   ├── train.py          # Model training and threshold tuning
│   ├── predict.py        # Inference pipeline on the test set
│   └── utils.py          # Shared utility functions
├── utils/                # Organizer-provided validation scripts
└── requirements.txt      # Dependency specifications
```

## Setup Instructions

1. **Environment Initialization**
   Ensure Python 3.8+ is installed. Create a virtual environment and install the required dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Data Placement**
   Place the competition `.tsv` files into their respective directories:
   - Training files into `data/train/`
   - Test files into `data/test/`

## Execution Pipeline

*Note: The pipeline execution scripts are actively under development.*

1. **Candidate Generation:** Run the blocking module to generate `candidate_pairs.tsv`.
2. **Feature Engineering & Training:** Run the training module to engineer features on the training candidate pairs, train the LightGBM classifier, and determine the optimal F0.5 threshold.
3. **Inference:** Run the prediction module to score the test candidates and generate the final `matching_results.tsv`.
4. **Validation:** Validate the outputs prior to submission using the provided organizer script:
   ```bash
   python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/test
   ```
