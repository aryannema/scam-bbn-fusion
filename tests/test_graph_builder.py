"""
Tests for src/graph_builder.py (the sender graph).

A small hand-built network is used so the expected answers can be worked out on paper.
"""

import pandas as pd
import pytest

from src.graph_builder import (
    build_graph,
    build_sender_features,
    degree_features,
    neighbourhood,
    sharing_features,
    template_fingerprint,
    timing_features,
)


@pytest.fixture
def tiny():
    """
    Five messages: one spraying sender and one steady sender.

    SPRAY sends to three different people; STEADY sends to the same person twice.
    """
    return pd.DataFrame({
        "sender_id": ["SPRAY", "SPRAY", "SPRAY", "STEADY", "STEADY"],
        "recipient_id": ["R1", "R2", "R3", "R9", "R9"],
        "text": ["aapka kyc band ho jayega", "aapka kyc band ho jayega now",
                 "aapka kyc band ho jayega today", "kal movie chalein", "khana kha liya"],
        "timestamp": ["2026-01-01 10:00", "2026-01-01 10:05", "2026-01-01 10:10",
                      "2026-01-01 10:00", "2026-01-20 10:00"],
    })


def test_graph_has_one_edge_per_distinct_pair(tiny):
    """Messaging the same person twice thickens one edge, it does not add a second."""
    graph = build_graph(tiny)
    assert graph.number_of_edges() == 4           # 3 for SPRAY, 1 for STEADY
    assert graph["STEADY"]["R9"]["weight"] == 2


def test_unique_recipient_ratio_separates_spraying_from_chatting(tiny):
    """
    The clearest campaign signal: a different stranger every time, or the same person?
    """
    features = degree_features(tiny, build_graph(tiny))
    assert features.loc["SPRAY", "unique_recipient_ratio"] == 1.0    # 3 people, 3 messages
    assert features.loc["STEADY", "unique_recipient_ratio"] == 0.5   # 1 person, 2 messages


def test_burstiness_is_higher_for_a_short_blast(tiny):
    """SPRAY sent everything in 10 minutes; STEADY spread it over 19 days."""
    features = timing_features(tiny)
    assert features.loc["SPRAY", "burstiness"] > features.loc["STEADY", "burstiness"]


def test_fingerprint_is_the_opening_words_of_the_cleaned_text():
    """Two messages from one campaign start the same way, then diverge."""
    assert template_fingerprint("Aapka SBI KYC band ho jayega! Turant...") == \
        "aapka sbi kyc band ho"
    assert template_fingerprint("AAPKA sbi kyc band ho") == "aapka sbi kyc band ho"


def test_forwarding_ratio_needs_a_second_sender():
    """
    One sender repeating itself is not forwarding.

    The ring signal is the SAME opening used by DIFFERENT accounts, so a single
    sender repeating its own template must score 0.
    """
    one = pd.DataFrame({"sender_id": ["A", "A"],
                        "text": ["aapka kyc band ho jayega", "aapka kyc band ho jayega x"]})
    assert sharing_features(one).loc["A", "forwarding_ratio"] == 0.0
    two = pd.DataFrame({"sender_id": ["A", "B"],
                        "text": ["aapka kyc band ho jayega", "aapka kyc band ho jayega x"]})
    assert sharing_features(two).loc["A", "forwarding_ratio"] == 1.0


def test_feature_table_has_a_row_per_sender_and_no_gaps(tiny):
    """F3 joins this table onto every message, so missing values would spread."""
    features = build_sender_features(tiny)
    assert set(features.index) == {"SPRAY", "STEADY"}
    assert not features.isna().any().any()


def test_neighbourhood_returns_just_the_sender_and_its_contacts(tiny):
    """The Streamlit app draws this, so it must stay small."""
    graph = build_graph(tiny)
    around = neighbourhood(graph, "STEADY")
    assert set(around.nodes) == {"STEADY", "R9"}
    assert neighbourhood(graph, "NOBODY").number_of_nodes() == 0


# ---------------------------------------------------------------------------
# The real graph
# ---------------------------------------------------------------------------
@pytest.fixture
def real_features():
    """Graph features for the real dataset, or skip if it has not been built."""
    from src.config import COMBINED_CSV, PROCESSED_DIR
    if not COMBINED_CSV.exists():
        pytest.skip("run `python -m src.data_builder` first")
    messages = pd.read_csv(COMBINED_CSV)
    senders = pd.read_csv(PROCESSED_DIR / "senders.csv")
    return build_sender_features(messages).join(senders.set_index("sender_id"), how="inner")


def test_clustering_is_not_all_zero(real_features):
    """
    REGRESSION TEST. Clustering must actually measure something.

    In a strictly sender->recipient graph there are no triangles at all, so every
    clustering value was exactly 0 and the feature was dead weight. Normal accounts
    now message each other inside social circles, which creates the triangles.
    """
    assert real_features["clustering"].max() > 0.1
    assert real_features.loc[~real_features["behaves_scammy"], "clustering"].mean() > \
        real_features.loc[real_features["behaves_scammy"], "clustering"].mean()


def test_pagerank_is_not_uniform(real_features):
    """
    REGRESSION TEST. PageRank needs incoming edges to say anything.

    When recipients never sent anything, every sender had in-degree 0 and PageRank was
    the same constant for all of them.
    """
    assert real_features["pagerank"].std() > 0


def test_no_single_graph_feature_gives_the_answer_away(real_features):
    """
    The simulation must not hand the label to one column.

    An earlier version let message volume separate the classes almost perfectly
    (out_degree reached F1 = 1.00 on its own), which made every result meaningless.
    """
    from sklearn.metrics import roc_auc_score
    truth = real_features["is_scammer"].astype(int)
    for column in ["messages_sent", "out_degree", "account_age_days", "past_reports"]:
        auc = roc_auc_score(truth, real_features[column])
        assert 0.02 < min(auc, 1 - auc) or abs(auc - 0.5) < 0.48, \
            f"{column} is a near-perfect separator (AUC {auc:.3f})"
