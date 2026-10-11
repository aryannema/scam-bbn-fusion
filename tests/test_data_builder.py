"""
Tests for src/data_builder.py (the dataset and the simulated network).

These tests never touch the network. They build a tiny dataset in memory with a fixed
seed, so they run in under a second and never depend on a website being up.

The last group of tests reads the real committed CSV and checks the leak-safety
promises we make in the paper. Those are the tests that matter most: if one of them
fails, our reported scores are not trustworthy.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    COMBINED_CSV,
    FRAC_BENIGN_SENDERS_LOOK_ODD,
    FRAC_SCAM_SENDERS_LOOK_NORMAL,
    RANDOM_SEED,
)
from src.data_builder import (
    BENIGN_TEMPLATES,
    SCAM_TEMPLATES,
    _sender_number,
    apply_spelling_variants,
    assign_senders,
    build_rings,
    fill_placeholders,
    generate_synthetic_messages,
    make_leaksafe_split,
    make_time_split,
    simulate_senders,
)


@pytest.fixture
def rng():
    """A fresh seeded random generator, so every test is repeatable."""
    return np.random.default_rng(RANDOM_SEED)


@pytest.fixture
def small_dataset(rng):
    """A tiny end-to-end dataset (400 messages) with senders and timestamps attached."""
    messages = generate_synthetic_messages(400, rng)
    senders = simulate_senders(messages, rng)
    return assign_senders(messages, senders, rng), senders


# ---------------------------------------------------------------------------
# Message generation
# ---------------------------------------------------------------------------
def test_every_placeholder_gets_filled(rng):
    """No message may reach the dataset still containing {curly} markers."""
    for templates in list(SCAM_TEMPLATES.values()) + list(BENIGN_TEMPLATES.values()):
        for template in templates:
            filled = fill_placeholders(template, rng)
            assert "{" not in filled and "}" not in filled


def test_spelling_variants_keep_the_sentence_intact(rng):
    """Respelling swaps words, so the word count must not change."""
    text = "turant verify karein kyc band account"
    varied = apply_spelling_variants(text, rng)
    assert len(varied.split()) == len(text.split())


def test_generated_messages_are_unique_and_balanced(rng):
    """
    Synthetic text must contain no duplicates and be close to 50/50 scam/benign.

    REGRESSION TEST: an earlier version produced 2,500 rows of which only 191 were
    benign, because the benign templates had no placeholders and collapsed into ~24
    identical strings that deduplication then removed.
    """
    frame = generate_synthetic_messages(500, rng)
    assert len(frame) == 500
    assert frame["text"].duplicated().sum() == 0
    share_scam = frame["label"].mean()
    assert 0.45 <= share_scam <= 0.55


# ---------------------------------------------------------------------------
# The simulated network
# ---------------------------------------------------------------------------
def test_each_ring_runs_exactly_one_campaign(small_dataset):
    """
    A ring must map to exactly one template family.

    This is what lets "hold out a ring" and "hold out a wording" be the same action,
    which is the foundation of the leak-safe split.
    """
    messages, _ = small_dataset
    per_ring = messages.groupby("sender_ring")["template_family"].nunique()
    assert (per_ring == 1).all()


def test_every_family_gets_at_least_two_rings(small_dataset):
    """No family may be all-or-nothing, or we could not split it at all."""
    messages, _ = small_dataset
    rings = build_rings(messages)
    assert all(len(ring_list) >= 2 for ring_list in rings.values())


def test_noise_fractions_are_exact(small_dataset):
    """
    Exactly 12% of scammers hide and 5% of innocent senders look odd.

    Without this noise the graph features would give the answer away, and the task
    would be far easier than the real one.
    """
    _, senders = small_dataset
    scammers = senders[senders["is_scammer"]]
    innocents = senders[~senders["is_scammer"]]
    hiding = (~scammers["behaves_scammy"]).sum()
    odd = innocents["behaves_scammy"].sum()
    assert hiding == round(len(scammers) * FRAC_SCAM_SENDERS_LOOK_NORMAL)
    assert odd == round(len(innocents) * FRAC_BENIGN_SENDERS_LOOK_ODD)


def test_history_follows_behaviour_not_hidden_truth(small_dataset):
    """
    Complaints and account age must track how a sender BEHAVES.

    A scammer who lies low should look like an old, unreported account, because that
    is exactly what an abuse team would see.
    """
    _, senders = small_dataset
    scammy = senders[senders["behaves_scammy"]]
    quiet = senders[~senders["behaves_scammy"]]
    assert scammy["past_reports"].mean() > quiet["past_reports"].mean()
    assert scammy["account_age_days"].mean() < quiet["account_age_days"].mean()


def test_scam_senders_are_busier_than_normal_senders(small_dataset):
    """A campaign is a few senders sending a lot; this is the signal F3 looks for."""
    messages, senders = small_dataset
    counts = messages.groupby("sender_id").size().rename("sent")
    joined = senders.set_index("sender_id").join(counts).dropna(subset=["sent"])
    busy = joined.loc[joined["is_scammer"], "sent"].mean()
    quiet = joined.loc[~joined["is_scammer"], "sent"].mean()
    assert busy > quiet


def test_fan_out_is_informative_but_not_a_giveaway(small_dataset):
    """
    REGRESSION TEST. The unique-recipient ratio must NOT separate perfectly.

    An earlier version sent every scam message to a brand-new stranger, so the ratio
    was exactly 1.00 for every scammy sender and 0 for nobody else - F3 would have
    been handed the answer. We now expect a real difference WITH overlap.
    """
    messages, senders = small_dataset
    stats = messages.groupby("sender_id").agg(sent=("text", "count"),
                                              seen=("recipient_id", "nunique"))
    joined = senders.set_index("sender_id").join(stats).dropna(subset=["sent"])
    joined["ratio"] = joined["seen"] / joined["sent"]
    scammy = joined.loc[joined["behaves_scammy"], "ratio"]
    normal = joined.loc[~joined["behaves_scammy"], "ratio"]
    assert scammy.mean() > normal.mean()          # there IS a signal
    assert scammy.min() < normal.max()            # but the two groups overlap


def test_sender_number_is_stable_across_processes():
    """
    REGRESSION TEST. We must not use Python's hash() on sender ids.

    Python randomises string hashing per process, so an earlier version produced a
    DIFFERENT dataset on every run even though RANDOM_SEED was fixed.
    """
    assert _sender_number("S00042") == 42
    assert _sender_number("S00000") == 0


# ---------------------------------------------------------------------------
# The splits
# ---------------------------------------------------------------------------
def test_leaksafe_split_separates_families_rings_and_senders(small_dataset, rng):
    """Nothing that shares wording or a sender may appear on both sides."""
    messages, _ = small_dataset
    messages["split_leaksafe"] = make_leaksafe_split(messages, rng)
    for column in ["template_family", "sender_ring", "sender_id"]:
        spanning = messages.groupby(column)["split_leaksafe"].nunique()
        assert (spanning == 1).all(), f"{column} appears in both train and test"


def test_leaksafe_split_keeps_both_labels_on_both_sides(small_dataset, rng):
    """A test set with only one label would make precision/recall meaningless."""
    messages, _ = small_dataset
    messages["split_leaksafe"] = make_leaksafe_split(messages, rng)
    for part in ["train", "test"]:
        labels = set(messages.loc[messages["split_leaksafe"] == part, "label"])
        assert labels == {0, 1}


def test_time_split_puts_the_future_in_test(small_dataset):
    """Train must end before test begins; that is the point of a time split."""
    messages, _ = small_dataset
    messages["split_time"] = make_time_split(messages)
    latest_train = messages.loc[messages["split_time"] == "train", "timestamp"].max()
    earliest_test = messages.loc[messages["split_time"] == "test", "timestamp"].min()
    assert earliest_test > latest_train


# ---------------------------------------------------------------------------
# The real committed dataset
# ---------------------------------------------------------------------------
@pytest.fixture
def built_dataset():
    """The real CSV, or skip if it has not been built on this machine yet."""
    if not COMBINED_CSV.exists():
        pytest.skip("run `python -m src.data_builder` first")
    return pd.read_csv(COMBINED_CSV)


def test_real_dataset_has_the_promised_columns(built_dataset):
    """The columns the project brief asks for must all be present."""
    required = {"message_id", "text", "sender_id", "recipient_id", "timestamp",
                "label", "language_mix", "source"}
    assert required.issubset(built_dataset.columns)


def test_real_dataset_has_no_duplicate_text(built_dataset):
    """
    Duplicate text is the simplest form of leakage.

    This matters here because the Hugging Face seed is 97.7% a re-upload of UCI; if
    the dedup step in load_hinglish_seed ever breaks, this test catches it.
    """
    assert built_dataset["text"].duplicated().sum() == 0


def test_real_dataset_no_text_in_both_splits(built_dataset):
    """No message may be used for both training and testing."""
    train = set(built_dataset.loc[built_dataset["split_leaksafe"] == "train", "text"])
    test = set(built_dataset.loc[built_dataset["split_leaksafe"] == "test", "text"])
    assert len(train & test) == 0


def test_real_dataset_synthetic_families_do_not_span_splits(built_dataset):
    """Template wording must never be shared across the train/test boundary."""
    synthetic = built_dataset[built_dataset["source"] == "synthetic_hinglish"]
    spanning = synthetic.groupby("template_family")["split_leaksafe"].nunique()
    assert (spanning == 1).all()


def test_real_dataset_contains_all_three_sources(built_dataset):
    """UCI, the deduplicated Hinglish seed, and our synthetic messages."""
    assert set(built_dataset["source"]) == {"uci_sms", "hf_hinglish_seed", "synthetic_hinglish"}
