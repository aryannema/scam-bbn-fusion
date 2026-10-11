"""
Tests for F5 (src/f5_structure_learning.py), data-driven structure learning.
"""

import pandas as pd
import pytest

from src.config import COMBINED_CSV, MAX_INDEGREE, OUTPUTS_DIR, PROCESSED_DIR
from src.f1_bbn import EXPERT_EDGES
from src.f5_structure_learning import (
    build_training_frame,
    compare_edges,
    fit_structure,
    forbidden_edges,
    holdout_log_likelihood,
    learn_structure,
    save_graph_png,
)


def test_blacklist_forbids_every_edge_out_of_fraud():
    """
    "Being a fraud makes your account younger" is backwards in time.

    We forbid the direction, not the variables, so the search still has to discover
    for itself which ones matter.
    """
    forbidden = forbidden_edges(["Fraud", "SenderTrust", "LinkRisk"])
    assert ("Fraud", "SenderTrust") in forbidden
    assert ("Fraud", "LinkRisk") in forbidden
    assert all(parent == "Fraud" for parent, _ in forbidden)
    assert ("Fraud", "Fraud") not in forbidden


def test_compare_edges_names_the_actual_differences():
    """"The graphs differ" is not a result; which edges differ is."""
    expert = [("A", "B"), ("B", "C")]
    learned = [("A", "B"), ("C", "B"), ("A", "C")]
    difference = compare_edges(expert, learned)
    assert difference["shared"] == [("A", "B")]
    assert difference["reversed"] == [("C", "B")]
    assert difference["added_by_data"] == [("A", "C")]
    assert difference["removed_by_data"] == []


@pytest.fixture(scope="module")
def frames():
    """Discretized train and test frames from the real data, or skip."""
    if not COMBINED_CSV.exists():
        pytest.skip("run `python -m src.data_builder` first")
    from src.f3_fused_features import build_fused_table
    messages = pd.read_csv(COMBINED_CSV)
    senders = pd.read_csv(PROCESSED_DIR / "senders.csv")
    table = build_fused_table(messages, senders)
    is_train = (messages["split_leaksafe"] == "train").to_numpy()
    train, edges = build_training_frame(table[is_train], messages["label"][is_train])
    return train, edges


def test_learning_respects_the_blacklist_and_the_parent_cap(frames):
    """
    Two hard constraints on the search, both checked on the real learned graph.

    If either leaked, the learned structure would not be comparable to the expert one.
    """
    train, _ = frames
    learned = learn_structure(train, "bic-d")
    assert all(parent != "Fraud" for parent, _ in learned), "an edge points out of Fraud"
    parents = pd.Series([child for _, child in learned]).value_counts()
    assert parents.max() <= MAX_INDEGREE


def test_learned_structure_is_a_dag_covering_our_variables(frames):
    """A cyclic 'network' cannot be used for inference at all."""
    import networkx as nx
    train, _ = frames
    learned = learn_structure(train, "bic-d")
    graph = nx.DiGraph(learned)
    assert nx.is_directed_acyclic_graph(graph)
    assert "Fraud" in graph.nodes()


def test_both_scores_produce_a_usable_structure(frames):
    """We report BIC and BDeu side by side, so both must actually fit."""
    train, _ = frames
    for score in ["bic-d", "bdeu"]:
        edges = learn_structure(train, score)
        network, engine = fit_structure(edges, train)
        assert network.check_model()


def test_fit_structure_keeps_variables_that_got_no_edges(frames):
    """Search can leave a variable isolated; it must still be part of the model."""
    train, _ = frames
    network, _ = fit_structure([("SenderTrust", "Fraud")], train)
    assert set(network.nodes()) >= set(train.columns)
    assert network.check_model()


def test_log_likelihood_is_negative_and_finite(frames):
    """
    A log-probability is always below zero, and must never be -inf.

    -inf would mean the model assigned probability exactly 0 to something that
    happened, which is why the CPDs are fitted with a BDeu prior.
    """
    train, _ = frames
    network, _ = fit_structure(EXPERT_EDGES, train)
    score = holdout_log_likelihood(network, train.head(200))
    assert score < 0
    assert score > -50


def test_graph_png_is_written(tmp_path):
    """The two pictures side by side are how this result gets presented."""
    path = save_graph_png([("A", "B")], tmp_path / "g.png", "test")
    assert path.exists() and path.stat().st_size > 0
