"""
Tests for F1 (src/f1_bbn.py), the expert Bayesian network and its explanation chain.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import BIN_LABELS, COMBINED_CSV, FRAUD_THRESHOLD, PROCESSED_DIR
from src.f1_bbn import (
    EXPERT_EDGES,
    NODE_SOURCES,
    build_network,
    discretize,
    evidence_from_features,
    fit_network,
    pattern_anomaly_from_parents,
    rank_evidence,
    score_message,
    score_many,
    sender_trust_score,
)


# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------
def test_discretize_splits_into_three_named_bins():
    """A discrete network cannot read a raw number, so every column becomes low/med/high."""
    labels, edges = discretize(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert set(labels) <= set(BIN_LABELS)
    assert len(edges) == 2                      # two cut points make three bins


def test_discretize_reuses_training_edges():
    """
    The cut points must come from the TRAINING data and be reused on test data.

    Re-cutting the test set by its own quantiles would leak the test distribution into
    the model: the same raw value would land in a different bin depending on what else
    happened to be in the test set.
    """
    _, edges = discretize(pd.Series([0, 10, 20, 30, 40, 50]))
    labels, reused = discretize(pd.Series([0, 50]), edges)
    assert reused == edges
    assert labels.tolist() == ["low", "high"]


def test_sender_trust_is_higher_for_an_old_clean_account():
    """
    SenderTrust is the one node where HIGH is reassuring.

    Getting this direction wrong would invert every explanation sentence.
    """
    frame = pd.DataFrame({"account_age_days": [2000, 5], "past_reports": [0, 9]})
    trust = sender_trust_score(frame)
    assert trust.iloc[0] > trust.iloc[1]


def test_pattern_anomaly_counts_how_many_parents_are_high():
    """The hidden middle node is derived from its three causes, not measured."""
    frame = pd.DataFrame({
        "UrgencyCue": ["high", "low", "high"],
        "CredentialAsk": ["high", "low", "low"],
        "LinkRisk": ["high", "low", "low"],
    })
    assert pattern_anomaly_from_parents(frame).tolist() == ["high", "low", "medium"]


# ---------------------------------------------------------------------------
# Network structure
# ---------------------------------------------------------------------------
def test_network_has_the_expert_structure_we_drew():
    """The diagram in the file docstring must match the code."""
    network = build_network()
    assert set(network.edges()) == set(EXPERT_EDGES)
    assert set(network.nodes()) == {"UrgencyCue", "CredentialAsk", "LinkRisk",
                                   "PatternAnomaly", "SenderTrust",
                                   "NetworkCentrality", "Fraud"}


def test_the_three_wording_nodes_reach_fraud_only_through_pattern_anomaly():
    """
    This is the modelling claim of the structure.

    Urgency does not cause fraud directly; it makes the WORDING anomalous, and that is
    what counts towards fraud.
    """
    network = build_network()
    for node in ["UrgencyCue", "CredentialAsk", "LinkRisk"]:
        assert list(network.successors(node)) == ["PatternAnomaly"]


# ---------------------------------------------------------------------------
# Fitting and scoring, on the real data
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fitted():
    """Fit the network once on the real training split and reuse it for every test."""
    if not COMBINED_CSV.exists():
        pytest.skip("run `python -m src.data_builder` first")
    from src.f3_fused_features import build_fused_table
    messages = pd.read_csv(COMBINED_CSV)
    senders = pd.read_csv(PROCESSED_DIR / "senders.csv")
    table = build_fused_table(messages, senders)
    is_train = (messages["split_leaksafe"] == "train").to_numpy()
    network, engine, edges = fit_network(table[is_train], messages["label"][is_train])
    return messages, table, is_train, network, engine, edges


def test_every_node_gets_a_probability_table(fitted):
    """A network missing a CPD cannot answer any query."""
    _, _, _, network, _, _ = fitted
    assert network.check_model()
    assert len(network.get_cpds()) == len(network.nodes())


def test_probability_is_a_probability(fitted):
    """A number outside 0..1 means the inference call was misused."""
    _, _, _, _, engine, _ = fitted
    result = score_message(engine, {"SenderTrust": "low"})
    assert 0.0 <= result["probability"] <= 1.0
    assert 0.0 <= result["prior"] <= 1.0


def test_low_sender_trust_raises_the_probability(fitted):
    """The direction the model learned must match how scams actually work."""
    _, _, _, _, engine, _ = fitted
    low = score_message(engine, {"SenderTrust": "low"})["probability"]
    high = score_message(engine, {"SenderTrust": "high"})["probability"]
    assert low > high


def test_explanation_starts_at_the_prior_and_adds_one_piece_at_a_time(fitted):
    """
    The chain IS the computation, so its shape is part of the contract.

    One line for the prior, then exactly one line per piece of evidence.
    """
    _, _, _, _, engine, _ = fitted
    evidence = {"SenderTrust": "low", "NetworkCentrality": "high", "LinkRisk": "high"}
    result = score_message(engine, evidence)
    assert result["explanation"][0].startswith("Prior")
    assert len(result["explanation"]) == len(evidence) + 1
    assert len(result["steps"]) == len(evidence) + 1


def test_explanation_leads_with_the_strongest_reason(fitted):
    """
    A human explains the biggest reason first, so the chain is ordered by effect.

    "Strongest" is measured by trying each piece of evidence on its own.
    """
    _, _, _, _, engine, _ = fitted
    evidence = {"SenderTrust": "low", "UrgencyCue": "low"}
    ranked = rank_evidence(engine, evidence)
    prior = score_message(engine, {})["probability"]
    shifts = [abs(alone - prior) for _, _, alone in ranked]
    assert shifts == sorted(shifts, reverse=True)


def test_the_final_probability_matches_the_last_step(fitted):
    """
    The chain must END at the score we report.

    If these disagreed, the explanation would not be an explanation of the verdict.
    """
    _, _, _, _, engine, _ = fitted
    evidence = {"SenderTrust": "low", "NetworkCentrality": "high"}
    result = score_message(engine, evidence)
    assert np.isclose(result["probability"], result["steps"][-1][2])


def test_verdict_follows_the_single_configured_threshold(fitted):
    """One threshold, from config, used by F1, F7 and F8 alike."""
    _, _, _, _, engine, _ = fitted
    for evidence in [{"SenderTrust": "low"}, {"SenderTrust": "high"}, {}]:
        result = score_message(engine, evidence)
        expected = "FRAUD" if result["probability"] >= FRAUD_THRESHOLD else "SAFE"
        assert result["verdict"] == expected


def test_evidence_from_features_fills_every_node(fitted):
    """Every evidence node needs a value, or inference silently uses fewer clues."""
    _, table, _, _, _, edges = fitted
    evidence = evidence_from_features(table.iloc[0], edges)
    assert set(evidence) == set(NODE_SOURCES)
    assert all(value in BIN_LABELS for value in evidence.values())


def test_score_many_matches_scoring_one_at_a_time(fitted):
    """
    score_many caches by evidence combination, so it must agree with the slow path.

    There are only 3^5 = 243 possible evidence combinations, which is why caching is
    safe and worth it.
    """
    _, table, _, _, engine, edges = fitted
    rows = table.head(25)
    fast = score_many(engine, rows, edges)
    slow = [score_message(engine, evidence_from_features(row, edges))["probability"]
            for _, row in rows.iterrows()]
    assert np.allclose(fast, slow)


def test_the_network_beats_guessing_on_held_out_data(fitted):
    """A network with no skill would make the whole explanation pointless."""
    from sklearn.metrics import roc_auc_score
    messages, table, is_train, _, engine, edges = fitted
    probability = score_many(engine, table[~is_train], edges)
    assert roc_auc_score(messages["label"][~is_train], probability) > 0.8
