"""
Tests for F3 (src/f3_fused_features.py), the fused feature set and the key comparison.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import COMBINED_CSV, PROCESSED_DIR
from src.f3_fused_features import (
    build_fused_table,
    compare_feature_sets,
    feature_groups,
    model_ranking,
    select_features,
    train_and_score,
)


@pytest.fixture
def real():
    """The real dataset, or skip if it has not been built on this machine."""
    if not COMBINED_CSV.exists():
        pytest.skip("run `python -m src.data_builder` first")
    # A RANDOM sample, not the first N rows: the CSV is grouped by source, so head()
    # returns UCI messages only and some tests then see a single class.
    messages = pd.read_csv(COMBINED_CSV).sample(2000, random_state=0).reset_index(drop=True)
    senders = pd.read_csv(PROCESSED_DIR / "senders.csv")
    table = build_fused_table(messages, senders)
    return messages, table


def test_fused_table_has_one_row_per_message(real):
    """Row order must match the messages, or features attach to the wrong labels."""
    messages, table = real
    assert len(table) == len(messages)
    assert not table.isna().any().any()


def test_feature_groups_split_text_from_behaviour(real):
    """The three-way comparison is only fair if the split is explicit."""
    _, table = real
    text_columns, graph_columns = feature_groups(table)
    assert "scam_lexicon_hits" in text_columns
    assert "phonetic_only_hits" in text_columns
    assert "unique_recipient_ratio" in graph_columns
    assert "past_reports" in graph_columns
    assert set(text_columns) & set(graph_columns) == set()      # no column in both
    assert len(text_columns) + len(graph_columns) == table.shape[1]


def test_selection_is_fitted_on_training_rows_only(real):
    """
    Choosing features using test rows is a quiet form of leakage.

    This test just pins the interface that makes the right thing easy to do.
    """
    messages, table = real
    is_train = (messages["split_leaksafe"] == "train").to_numpy()
    selected, scores = select_features(table[is_train], messages["label"][is_train], k=8)
    assert len(selected) == 8
    assert len(scores) == table.shape[1]
    assert scores.is_monotonic_decreasing


def test_model_ranking_covers_every_feature(real):
    """Reported next to mutual information, so it must cover the same columns."""
    messages, table = real
    ranking = model_ranking(table, messages["label"])
    assert set(ranking.index) == set(table.columns)
    assert np.isclose(ranking.sum(), 1.0, atol=0.01)


def test_scores_are_in_range(real):
    """A metric outside 0..1 means the labels and predictions got misaligned."""
    messages, table = real
    is_train = (messages["split_leaksafe"] == "train").to_numpy()
    result = train_and_score(table.to_numpy(), messages["label"].to_numpy(), is_train)
    for name, value in result.items():
        assert 0.0 <= value <= 1.0, f"{name} = {value}"


def test_the_three_way_comparison_runs_and_is_labelled(real):
    """This table is the first key result in the paper, so its shape is pinned."""
    messages, table = real
    is_train = (messages["split_leaksafe"] == "train").to_numpy()
    comparison = compare_feature_sets(table, messages["label"].to_numpy(), is_train)
    assert list(comparison.index) == ["text only (F2+F6)", "graph only (F3)",
                                      "fused (text+graph)"]
    assert "f1" in comparison.columns
    assert "false_positive_rate" in comparison.columns
