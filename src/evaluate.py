"""
Local F0.5 scorer, matching the competition's macro-average definition exactly.
Use this against your own held-out validation split (never against test,
which has no ground truth) before every leaderboard submission.
"""
from typing import Dict, Set


def f_beta_per_entity(predicted: Set[str], truth: Set[str], beta: float = 0.5) -> float:
    """
    Score one S1 entity.
    - Correctly predicting an empty set for a true singleton -> 1.0
    - Predicting any match for a true singleton -> 0.0
    """
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0  # recall = 0 -> F_beta = 0

    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth) if truth else 0.0
    if precision == 0.0 and recall == 0.0:
        return 0.0

    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def macro_f_beta(
    predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]], beta: float = 0.5
) -> float:
    """
    predictions / ground_truth: source1_entity_id -> set(matched_entity_ids)
    Every S1 entity in ground_truth must have an entry in predictions
    (missing entries should be treated as empty set = predicted singleton).
    """
    scores = []
    for s1_id, truth in ground_truth.items():
        predicted = predictions.get(s1_id, set())
        scores.append(f_beta_per_entity(predicted, truth, beta))
    return sum(scores) / len(scores) if scores else 0.0
