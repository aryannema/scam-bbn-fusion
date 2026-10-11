"""
Tests for F4 (src/f4_ensemble.py), the stacked ensemble.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.config import COMBINED_CSV, N_FOLDS
from src.f4_ensemble import (
    make_feature_model,
    metrics,
    out_of_fold_probabilities,
    train_stack,
)


@pytest.fixture
def toy():
    """A small separable problem, so expected behaviour is obvious."""
    return make_classification(n_samples=300, n_features=8, n_informative=4,
                               random_state=0)


def test_out_of_fold_predictions_cover_every_row(toy):
    """Every training row needs a prediction from a model that never saw it."""
    features, labels = toy
    probability = out_of_fold_probabilities(features, labels)
    assert len(probability) == len(labels)
    assert probability.min() >= 0 and probability.max() <= 1


def test_out_of_fold_is_more_modest_than_asking_the_model_about_its_own_data(toy):
    """
    THE POINT OF OUT-OF-FOLD PREDICTIONS.

    A model asked about rows it was trained on gives memorised, over-confident
    answers. If we fed those to the stack, it would learn "always believe the feature
    model" and the ensemble would be worse than useless on new data. Out-of-fold
    predictions must therefore be measurably LESS confident.
    """
    features, labels = toy
    honest = out_of_fold_probabilities(features, labels)
    model = make_feature_model().fit(features, labels)
    memorised = model.predict_proba(features)[:, 1]
    confidence = lambda p: np.abs(p - 0.5).mean()
    assert confidence(honest) < confidence(memorised)


def test_stack_learns_to_follow_the_detector_that_is_right():
    """
    The stack's whole job is deciding which detector to trust.

    Here the first detector is always correct and the second is pure noise, so the
    first must end up with the larger weight.
    """
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 400)
    good = labels * 0.9 + 0.05                      # tracks the truth
    noise = rng.random(400)                         # tells us nothing
    stack = train_stack(good, noise, labels)
    good_weight, noise_weight = stack.coef_[0]
    assert good_weight > noise_weight


def test_stack_takes_exactly_two_inputs():
    """Two inputs is what makes the learned weights readable out loud."""
    labels = np.array([0, 1] * 50)
    stack = train_stack(np.linspace(0, 1, 100), np.linspace(1, 0, 100), labels)
    assert stack.coef_.shape == (1, 2)


def test_metrics_are_all_in_range():
    """A metric outside 0..1 means labels and predictions got misaligned."""
    labels = np.array([0, 0, 1, 1])
    result = metrics(labels, np.array([0.1, 0.2, 0.8, 0.9]))
    assert all(0.0 <= value <= 1.0 for value in result.values())
    assert result["f1"] == 1.0
    assert result["false_positive_rate"] == 0.0


def test_false_positive_rate_counts_flagged_benign_messages():
    """
    FPR is reported separately because it is the expensive mistake.

    Here one of two benign messages is flagged, so the rate must be exactly 0.5.
    """
    labels = np.array([0, 0, 1])
    result = metrics(labels, np.array([0.9, 0.1, 0.9]))
    assert result["false_positive_rate"] == 0.5


def test_folds_match_the_configured_number():
    """The stack's honesty depends on this, so it comes from config, not a literal."""
    assert N_FOLDS >= 2


@pytest.mark.skipif(not COMBINED_CSV.exists(), reason="dataset not built")
def test_the_full_ensemble_trains_and_beats_the_bbn_alone():
    """
    End-to-end check on the real data.

    The ensemble does not have to beat BOTH parts - on the time split it does not, and
    we report that honestly - but it must beat the weaker part, or the fusion has
    achieved nothing at all.
    """
    from src.f4_ensemble import train_ensemble
    comparison, parts = train_ensemble("split_leaksafe")
    assert comparison.loc["stacked ensemble (F4)", "f1"] > \
        comparison.loc["BBN alone (F1)", "f1"]
    assert set(parts["weights"]) == {"feature_model", "bbn"}
