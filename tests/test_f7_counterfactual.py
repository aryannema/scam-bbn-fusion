"""
Tests for F7 (src/f7_counterfactual.py), counterfactual explanations.
"""

import pandas as pd
import pytest

from src.config import COMBINED_CSV, FRAUD_THRESHOLD, MAX_COUNTERFACTUAL_CHANGES, PROCESSED_DIR
from src.f1_bbn import _probability_of_fraud, evidence_from_features, fit_network
from src.f7_counterfactual import (
    alternatives,
    changes_of_size,
    describe,
    find_counterfactuals,
    verdict_of,
)


def test_verdict_uses_the_one_shared_threshold():
    """F1, F7 and F8 must agree on what counts as fraud."""
    assert verdict_of(FRAUD_THRESHOLD) == "FRAUD"
    assert verdict_of(FRAUD_THRESHOLD - 0.01) == "SAFE"


def test_alternatives_exclude_the_current_value():
    """A counterfactual has to change something, so staying put is not an option."""
    assert set(alternatives("low")) == {"medium", "high"}
    assert "low" not in alternatives("low")


def test_candidate_changes_are_the_right_size_and_complete():
    """
    With 2 variables and 2 alternatives each there are exactly 4 single changes.

    Counting them is how we know the brute force really is exhaustive.
    """
    evidence = {"A": "low", "B": "low"}
    singles = list(changes_of_size(evidence, 1))
    assert len(singles) == 4
    assert all(len(change) == 1 for change in singles)
    pairs = list(changes_of_size(evidence, 2))
    assert len(pairs) == 4                     # 2 x 2 combinations of both variables
    assert all(len(change) == 2 for change in pairs)


def test_sentence_reads_like_english():
    """The sentence is the deliverable, so its wording is part of the contract."""
    sentence = describe({"SenderTrust": "high"}, 0.93, 0.31)
    assert sentence.startswith("If how trustworthy the sender account looks had been HIGH")
    assert "marked SAFE" in sentence
    assert "0.93 -> 0.31" in sentence


@pytest.fixture(scope="module")
def engine_and_rows():
    """A fitted network plus a flagged message from the real test split, or skip."""
    if not COMBINED_CSV.exists():
        pytest.skip("run `python -m src.data_builder` first")
    from src.f1_bbn import score_many
    from src.f3_fused_features import build_fused_table
    messages = pd.read_csv(COMBINED_CSV)
    senders = pd.read_csv(PROCESSED_DIR / "senders.csv")
    table = build_fused_table(messages, senders)
    is_train = (messages["split_leaksafe"] == "train").to_numpy()
    _, engine, edges = fit_network(table[is_train], messages["label"][is_train])
    scores = score_many(engine, table[~is_train], edges)
    flagged = table[~is_train].iloc[scores.argmax()]
    return engine, evidence_from_features(flagged, edges)


def test_counterfactuals_actually_flip_the_verdict(engine_and_rows):
    """
    Every returned counterfactual must really change the outcome.

    This re-runs inference on the changed evidence rather than trusting the search.
    """
    engine, evidence = engine_and_rows
    original = verdict_of(_probability_of_fraud(engine, evidence))
    results = find_counterfactuals(engine, evidence)
    assert results, "no counterfactual found for the most-flagged message"
    for item in results:
        changed = dict(evidence)
        changed.update(item["change"])
        assert verdict_of(_probability_of_fraud(engine, changed)) != original


def test_counterfactuals_are_minimal(engine_and_rows):
    """
    THE WHOLE POINT. "If everything were different it would be safe" is useless.

    All returned changes must be the same, smallest size - and if a one-variable
    change exists, no two-variable change may be reported.
    """
    engine, evidence = engine_and_rows
    results = find_counterfactuals(engine, evidence)
    sizes = {item["n_changes"] for item in results}
    assert len(sizes) == 1
    smallest = sizes.pop()
    if smallest > 1:
        # prove no single change could have done it
        for change in changes_of_size(evidence, 1):
            candidate = dict(evidence)
            candidate.update(change)
            original = verdict_of(_probability_of_fraud(engine, evidence))
            assert verdict_of(_probability_of_fraud(engine, candidate)) == original


def test_changes_never_exceed_the_configured_limit(engine_and_rows):
    """Beyond three changes a counterfactual stops being actionable."""
    engine, evidence = engine_and_rows
    for item in find_counterfactuals(engine, evidence):
        assert item["n_changes"] <= MAX_COUNTERFACTUAL_CHANGES


def test_a_safe_message_gets_counterfactuals_towards_fraud(engine_and_rows):
    """
    The search is symmetric: it flips the verdict, whichever way it points.

    For an already-safe message the counterfactual explains what would have made it
    suspicious, which is just as useful.
    """
    engine, _ = engine_and_rows
    safe = {"UrgencyCue": "low", "CredentialAsk": "low", "LinkRisk": "low",
            "SenderTrust": "high", "NetworkCentrality": "low"}
    assert verdict_of(_probability_of_fraud(engine, safe)) == "SAFE"
    results = find_counterfactuals(engine, safe)
    assert all(item["verdict"] == "FRAUD" for item in results)
