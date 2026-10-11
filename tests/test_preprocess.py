"""
Tests for src/preprocess.py (the shared text normalizer).

Each test names the trick it defends against, so when one fails you know which
scammer technique just got through.
"""

from src.preprocess import (
    collapse_repeats,
    fold_homoglyphs,
    normalize,
    strip_zero_width,
    tokenize,
    undo_leetspeak_token,
)


def test_homoglyphs_become_latin():
    """A Cyrillic 'а' and 'е' must end up as plain Latin letters."""
    disguised = "bаnk аlеrt"          # Cyrillic a and e
    assert fold_homoglyphs(disguised) == "bank alert"


def test_zero_width_characters_are_removed():
    """Invisible characters hidden inside a word must disappear."""
    assert strip_zero_width("k​y‍c") == "kyc"


def test_leetspeak_is_undone_in_words():
    """Digit-for-letter swaps inside a real word must be reversed."""
    assert undo_leetspeak_token("b@nk") == "bank"
    assert undo_leetspeak_token("acc0unt") == "account"
    assert undo_leetspeak_token("0tp") == "otp"


def test_pure_numbers_are_never_touched():
    """
    REGRESSION TEST. "10000" must stay "10000".

    If we de-leetspeak a token with no letters, an amount like 10000 becomes "ioooo"
    and every money feature in F2 is destroyed.
    """
    assert undo_leetspeak_token("10000") == "10000"
    assert undo_leetspeak_token("499") == "499"
    assert undo_leetspeak_token("482913") == "482913"


def test_long_letter_runs_collapse():
    """Padding like 'urgeeeent' must shrink back to 'urgent'."""
    assert collapse_repeats("urgeeeent") == "urgent"
    assert collapse_repeats("wow!!!!") == "wow!"


def test_digit_runs_never_collapse():
    """
    REGRESSION TEST. "10,000" contains three zeros in a row.

    An earlier version collapsed them and turned ₹10,000 into ₹10,0, silently
    corrupting every amount in the dataset.
    """
    assert collapse_repeats("10,000") == "10,000"
    assert "10,000" in normalize("Pay Rs 10,000 now")


def test_two_letter_runs_are_left_alone():
    """
    'khaata' and 'jeet' keep their double letters on purpose.

    Those are normal Romanized Hindi spellings; matching them to 'khata' is F6's job
    (the phonetic layer), not the normalizer's.
    """
    assert collapse_repeats("khaata") == "khaata"
    assert collapse_repeats("jeet") == "jeet"


def test_normalize_handles_a_full_scam_message():
    """The whole pipeline on one realistic disguised message."""
    raw = "SBI KYC B@ND ho jaaayega!!! Turant verify: http://sbi-kyc.xyz"
    clean = normalize(raw)
    assert "band" in clean               # leetspeak undone
    assert "jayega" in clean             # 3+ repeat collapsed
    assert clean == clean.lower()        # lowercased
    assert "sbi-kyc.xyz" in clean        # the URL survives for F2 to count


def test_normalize_survives_missing_text():
    """pandas hands us NaN for empty cells, so normalize must not crash."""
    assert normalize(None) == ""
    assert normalize(float("nan")) == ""


def test_tokenize_keeps_digits_and_drops_punctuation():
    """F2 and F6 both need a clean list of word tokens."""
    assert tokenize("kyc band ho jayega!") == ["kyc", "band", "ho", "jayega"]
    assert tokenize("pay rs 5000 now") == ["pay", "rs", "5000", "now"]
