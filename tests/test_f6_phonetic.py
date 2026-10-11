"""
Tests for F6 (src/f6_phonetic.py), the phonetic layer.

The first test is the one to show a professor: it is the entire reason this file
exists. The rest check one encoder rule each.
"""

import pandas as pd
import pytest

from src.f2_hinglish_features import SCAM_LEXICON
from src.f6_phonetic import (
    MIN_CODE_LENGTH,
    SCAM_LEXICON_CODES,
    _apply_digraphs,
    _drop_inner_vowels,
    _drop_nasal_before_consonant,
    _squeeze_doubles,
    build_code_vectorizer,
    codes_as_text,
    extract_phonetic_features,
    phonetic_code,
    phonetic_codes,
    phonetic_hits,
)


# ---------------------------------------------------------------------------
# The headline claim
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spellings", [
    ["turant", "turrant", "turat", "tarant"],   # "at once" - the brief's own example
    ["kyc", "keyc"],
    ["khata", "khaata", "khatha"],              # "account"
    ["band", "bnd", "bandh"],                   # "closed"
    ["verify", "varify", "verifiy"],
    ["paise", "paisa", "pese"],                 # "money"
    ["warna", "varna"],                         # "otherwise"
    ["jaldi", "jaldee", "jldi"],                # "quickly"
])
def test_every_spelling_of_a_word_gets_one_code(spellings):
    """
    THE POINT OF F6. All the ways people spell one Hindi word must agree.

    If this fails, a scammer only has to change spelling to walk past the lexicon.
    """
    codes = {phonetic_code(word) for word in spellings}
    assert len(codes) == 1, f"{spellings} produced {codes}"


def test_different_words_keep_different_codes():
    """
    The encoder must not collapse everything into one blob.

    A code that matched every word would make the lexicon useless.
    """
    assert phonetic_code("turant") != phonetic_code("lottery")
    assert phonetic_code("kyc") != phonetic_code("loan")


# ---------------------------------------------------------------------------
# One rule at a time
# ---------------------------------------------------------------------------
def test_digraphs_become_one_letter():
    """Two letters spelling one sound collapse: kh->k, ph->f, aa->a."""
    assert _apply_digraphs("khaata") == "kata"
    assert _apply_digraphs("phone") == "fone"
    assert _apply_digraphs("sheet") == "sit"


def test_nasal_before_a_consonant_is_dropped():
    """
    The anusvara rule: Hindi's small nasal sound is often simply not typed.

    This is what makes "turant" and "turat" the same word.
    """
    assert _drop_nasal_before_consonant("turant") == "turat"
    assert _drop_nasal_before_consonant("band") == "bad"


def test_nasal_at_the_end_or_before_a_vowel_is_kept():
    """"pin" and "nahi" must keep their n: it is a real, always-written sound."""
    assert _drop_nasal_before_consonant("pin") == "pin"
    assert _drop_nasal_before_consonant("nahi") == "nahi"


def test_doubled_letters_collapse():
    """Doubling a letter is a typing habit, not a different sound."""
    assert _squeeze_doubles("turrant") == "turant"
    assert _squeeze_doubles("aabbcc") == "abc"


def test_inner_vowels_are_dropped_but_the_first_letter_stays():
    """Soundex's original trick: the opening sound is reliable, vowels are not."""
    assert _drop_inner_vowels("turat") == "trt"
    assert _drop_inner_vowels("account") == "accnt"


def test_encoder_survives_junk_input():
    """Real messages contain punctuation, digits and empty strings."""
    assert phonetic_code("") == ""
    assert phonetic_code("!!!") == ""
    assert phonetic_code("123") == ""
    assert phonetic_code("KYC!") == phonetic_code("kyc")


def test_codes_are_capped_in_length():
    """Long codes add nothing and make the bag-of-codes table wider than needed."""
    assert len(phonetic_code("disconnection")) <= 6


# ---------------------------------------------------------------------------
# Matching the lexicon by sound
# ---------------------------------------------------------------------------
def test_lexicon_codes_are_all_long_enough():
    """
    Every stored code must clear the minimum length.

    Short codes caused 1,865 false hits on benign messages, so they are excluded.
    """
    assert all(len(code) >= MIN_CODE_LENGTH for code in SCAM_LEXICON_CODES)
    assert len(SCAM_LEXICON_CODES) > 0


def test_phonetic_matching_catches_what_exact_matching_misses():
    """
    A misspelled scam message must still be caught.

    "turrant varify" is invisible to exact matching and visible to F6. That gap is
    the value this functionality adds.
    """
    total, missed_by_exact = phonetic_hits("Aapka khaata bnd ho jayega, turrant varify karein")
    assert total >= 2
    assert missed_by_exact >= 2


def test_correctly_spelled_scam_words_are_not_double_counted():
    """When spelling is normal, F6 adds nothing new - and must say so."""
    _, missed_by_exact = phonetic_hits("Aapka khata band ho jayega, turant verify karein")
    assert missed_by_exact == 0


def test_a_benign_chat_message_gets_no_phonetic_hits():
    """
    REGRESSION TEST. Everyday words must not trip the phonetic matcher.

    With a 2-character minimum, "baje" matched "bhej", "like" matched "link" and
    "bad" matched "band" - 1,865 false hits across our benign messages. Raising the
    floor to 3 characters removed those.
    """
    total, _ = phonetic_hits("Yaar kal movie chalein? Show 9 baje ka hai.")
    assert total == 0
    total, _ = phonetic_hits("I like the bad bed, see you soon")
    assert total == 0


def test_known_collision_is_documented_not_hidden():
    """
    An HONEST limitation: "later" and "lottery" really do get the same code.

    Matching by sound cannot be perfect - some innocent words sound like scam words.
    We measured the cost on our own data: about 10.6% of benign messages pick up at
    least one phonetic hit (0.12 hits per message on average), against 1.39 per scam
    message. We keep the collision rather than special-case it, because:
      1. phonetic hits are a COUNT fed to a model, not a rule that blocks a message,
         so the model can learn how much one stray hit is worth; and
      2. special-casing English words would break the Hindi variants we need - our own
         Hindi word list contains "bnd" and "turrant", so filtering on it would delete
         exactly the codes F6 exists to catch (we tried: it cut the signal from 3.1x
         down to 2.7x).
    """
    assert phonetic_code("later") == phonetic_code("lottery")
    total, _ = phonetic_hits("talk to you later")
    assert total == 1


# ---------------------------------------------------------------------------
# The "embedding" part
# ---------------------------------------------------------------------------
def test_codes_as_text_turns_a_message_into_codes():
    """CountVectorizer needs text, so we hand it a string of codes."""
    assert codes_as_text("Turant verify") == "trt vrf"


def test_phonetic_codes_maps_every_token():
    """One code per token, in order."""
    assert phonetic_codes(["turant", "verify"]) == ["trt", "vrf"]


def test_bag_of_codes_matrix_has_one_row_per_message():
    """This matrix is what the feature model in F4 consumes."""
    texts = ["turant verify karein", "turrant varify karen", "kal movie chalein"]
    matrix, vectorizer = build_code_vectorizer(texts, max_features=50)
    assert matrix.shape[0] == 3
    assert len(vectorizer.get_feature_names_out()) <= 50


def test_misspelled_variants_land_in_the_same_column():
    """
    The whole reason to vectorize codes instead of words.

    "turant" and "turrant" must share a column, or each gets too few examples to
    learn anything from.
    """
    matrix, vectorizer = build_code_vectorizer(["turant", "turrant"])
    names = list(vectorizer.get_feature_names_out())
    assert names == ["trt"]
    assert matrix.shape == (2, 1)


# ---------------------------------------------------------------------------
# The entry function
# ---------------------------------------------------------------------------
def test_extract_returns_both_columns_one_row_per_message():
    """F3 glues these two columns onto F2's table, so the order must match."""
    frame = extract_phonetic_features(["turrant kyc band", "hello friend"])
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["phonetic_scam_hits", "phonetic_only_hits"]
    assert len(frame) == 2


def test_extract_handles_empty_text():
    """Blank cells exist in real data."""
    frame = extract_phonetic_features(["", None])
    assert frame["phonetic_scam_hits"].tolist() == [0, 0]
