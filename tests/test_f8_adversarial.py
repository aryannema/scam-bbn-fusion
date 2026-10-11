"""
Tests for F8 (src/f8_adversarial.py), adversarial robustness.
"""

import numpy as np
import pytest

from src.config import COMBINED_CSV, RANDOM_SEED
from src.f2_hinglish_features import extract_text_features
from src.f8_adversarial import (
    ATTACKS,
    attack_many,
    features_for,
    homoglyph_attack,
    leetspeak_attack,
    respell_attack,
    typo_attack,
    weak_normalizer,
    zero_width_attack,
)
from src.preprocess import normalize


@pytest.fixture
def rng():
    return np.random.default_rng(RANDOM_SEED)


def test_every_attack_changes_the_text_but_keeps_it_readable(rng):
    """An attack nobody can read is not an attack - the victim has to be fooled too."""
    original = "Aapka SBI KYC band ho jayega. Turant verify karein"
    for name, attack in ATTACKS.items():
        attacked = attack(original, rng)
        assert attacked != original, f"{name} changed nothing"
        assert len(attacked) >= len(original) * 0.8, f"{name} destroyed the message"


def test_homoglyphs_are_invisible_to_a_human_but_not_to_a_computer(rng):
    """"bank" with a Cyrillic a is a different string entirely."""
    attacked = homoglyph_attack("bank account", rng)
    assert attacked != "bank account"
    assert any(ord(character) > 127 for character in attacked)


def test_normalize_undoes_the_homoglyph_attack(rng):
    """
    This is the measurable claim behind defence (b).

    If normalization did not restore the original text, it would be adding no value.
    """
    assert normalize(homoglyph_attack("bank account", rng)) == "bank account"


def test_normalize_undoes_leetspeak_and_zero_width(rng):
    """Two more disguises the normalizer is responsible for."""
    assert "bank" in normalize(leetspeak_attack("bank", rng))
    assert normalize(zero_width_attack("kyc", rng)) == "kyc"


def test_respelling_needs_no_trickery_at_all(rng):
    """
    THE ATTACK F6 EXISTS FOR.

    Nothing here is a hidden character or a digit swap - "turrant" is simply another
    legitimate way to spell the same Hindi word, so normalization cannot fix it and
    only sound-based matching can.
    """
    attacked = respell_attack("turant verify karein", rng)
    assert attacked == "turrant varify karein"
    assert all(ord(character) < 128 for character in attacked)
    assert normalize(attacked) != "turant verify karein"   # normalizing does NOT help


def test_phonetic_defence_recovers_a_respelled_scam_word():
    """F6 must see through the respelling that defeats the normalizer."""
    from src.f6_phonetic import phonetic_hits
    total, missed_by_exact = phonetic_hits("turrant varify karein")
    assert total >= 2 and missed_by_exact >= 2


def test_typo_attack_leaves_very_short_text_alone(rng):
    """Swapping letters in a 3-character string would just destroy it."""
    assert typo_attack("abc", rng) == "abc"


def test_weak_normalizer_is_a_real_baseline():
    """
    "No defence" must genuinely leave the disguise in place.

    If the baseline secretly cleaned the text, defences (b) and (c) would look
    worthless and the whole comparison would be meaningless.
    """
    assert weak_normalizer("B@NK") == "b@nk"       # the @ survives
    assert "bank" not in weak_normalizer("B@NK")
    assert weak_normalizer(None) == ""


def test_the_three_defences_produce_different_feature_sets():
    """Each defence must change what the model can see, or it is not a defence."""
    texts = ["Aapka KYC b@nd ho jayega turrant"]
    none = features_for(texts, "none")
    plain = features_for(texts, "normalize")
    phonetic = features_for(texts, "normalize+phonetic")
    assert phonetic.shape[1] > plain.shape[1]       # phonetic adds columns
    assert plain.shape[1] == none.shape[1]          # same columns, different values
    assert none["scam_lexicon_hits"].iloc[0] < plain["scam_lexicon_hits"].iloc[0]


def test_attacks_are_reproducible():
    """A fixed seed means the numbers in the report can be regenerated exactly."""
    texts = ["Aapka SBI KYC band ho jayega turant verify"]
    first = attack_many(texts, homoglyph_attack)
    second = attack_many(texts, homoglyph_attack)
    assert first == second


@pytest.mark.skipif(not COMBINED_CSV.exists(), reason="dataset not built")
def test_the_full_evaluation_runs_and_is_shaped_as_reported():
    """The robustness table is a paper result, so its shape is pinned."""
    from src.f8_adversarial import run_attack_evaluation
    table = run_attack_evaluation()
    assert "clean (no attack)" in table.index
    for name in ATTACKS:
        assert name in table.index
    for defence in ["none", "normalize", "normalize+phonetic"]:
        assert defence in table.columns
        assert table[defence].between(0, 1).all()
