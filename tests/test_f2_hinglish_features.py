"""
Tests for F2 (src/f2_hinglish_features.py).

Each test is written so that reading the test tells you what the feature means.
"""

import pandas as pd

from src.f2_hinglish_features import (
    SCAM_LEXICON,
    extract_hosts,
    caps_ratio,
    code_mixing_index,
    count_lexicon_hits,
    count_script_switches,
    count_urls,
    digit_ratio,
    extract_text_features,
    find_money_amounts,
    link_risk,
    tag_token,
    tag_tokens,
)


# ---------------------------------------------------------------------------
# Language tagging
# ---------------------------------------------------------------------------
def test_tag_token_separates_the_four_kinds_of_word():
    """Hindi, English, genuinely neutral, and words our lists do not know."""
    assert tag_token("aapka") == "hindi"
    assert tag_token("account") == "english"
    assert tag_token("sbi") == "neutral"        # a brand: belongs to no language
    assert tag_token("482913") == "neutral"     # a number: belongs to no language
    assert tag_token("lonavala") == "unknown"   # a place name we never listed


def test_code_mixing_index_is_zero_for_one_language():
    """A sentence in a single language is not mixed, by definition."""
    assert code_mixing_index(["hindi", "hindi", "hindi"]) == 0.0
    assert code_mixing_index(["english", "english"]) == 0.0


def test_code_mixing_index_is_fifty_for_an_even_mix():
    """An exact half-and-half mix is the most code-mixed a sentence can be."""
    assert code_mixing_index(["hindi", "english"]) == 50.0
    assert code_mixing_index(["hindi", "hindi", "english", "english"]) == 50.0


def test_code_mixing_index_ignores_untaggable_words():
    """
    Neutral and unknown words must not count as a language.

    Otherwise "SBI" and every unlisted word would silently push the score around.
    """
    assert code_mixing_index(["hindi", "neutral", "unknown"]) == 0.0
    assert code_mixing_index(["neutral", "neutral"]) == 0.0


def test_script_switches_counts_flips_between_languages():
    """Switches measure interleaving, which is different from how much of each."""
    assert count_script_switches(["hindi", "hindi", "english", "hindi"]) == 2
    assert count_script_switches(["hindi", "hindi"]) == 0
    # a neutral word in the middle must not invent a switch
    assert count_script_switches(["hindi", "neutral", "hindi"]) == 0


def test_tag_tokens_labels_a_whole_sentence():
    """The real sentence "Aapka account band" is Hindi-English-Hindi."""
    assert tag_tokens(["aapka", "account", "band"]) == ["hindi", "english", "hindi"]


# ---------------------------------------------------------------------------
# Scam signals
# ---------------------------------------------------------------------------
def test_lexicon_hits_counts_scam_words():
    """Both the Hindi and the English half of the lexicon must work."""
    assert count_lexicon_hits(["kyc", "band", "hello"], SCAM_LEXICON) == 2
    assert count_lexicon_hits(["hello", "friend"], SCAM_LEXICON) == 0


def test_money_amounts_need_a_currency_marker():
    """Amounts are found when a currency marker says they are amounts."""
    assert find_money_amounts("Pay Rs 10,000 today") == [10000]
    assert find_money_amounts("Pay $499 and ₹250") == [499, 250]


def test_otp_is_not_mistaken_for_money():
    """
    REGRESSION TEST. A bare 6-digit number must NOT become an amount.

    "482913 is your OTP" would otherwise be read as nearly five lakh rupees, which
    would make genuine bank OTP messages look like huge-money scams.
    """
    assert find_money_amounts("482913 is your OTP") == []
    assert find_money_amounts("call 08452810075") == []


def test_urls_are_counted():
    """Nearly every phishing message needs the victim to open a link."""
    assert count_urls("go to http://a.xyz now") == 1
    assert count_urls("http://a.xyz and www.b.in") == 2
    assert count_urls("no links here") == 0


def test_bare_domains_are_counted_too():
    """
    Real scam SMS often write a link with no http:// at all.

    An http-only pattern missed "cashback-amazon.in" completely.
    """
    assert count_urls("claim at cashback-amazon.in now") == 1
    assert count_urls("lunch at 9 baje ok") == 0
    assert count_urls("Rs 5000 credited. Balance updated.") == 0


def test_host_is_parsed_not_substring_matched():
    """
    REGRESSION TEST for a real bug.

    `"t.co" in "https://www.flipkart.com/orders"` is True, because "flipkart.com"
    contains "t.co". Every legitimate Flipkart tracking link was therefore being
    labelled a URL shortener. We now compare the parsed HOST instead.
    """
    assert extract_hosts("https://www.flipkart.com/orders") == ["flipkart.com"]
    assert link_risk("https://www.flipkart.com/orders") == (0, 0)


def test_link_risk_spots_shorteners_and_throwaway_domains():
    """A shortener hides the destination; .xyz is not where a bank lives."""
    assert link_risk("click http://bit.ly/3xKyc9") == (1, 0)
    assert link_risk("go to http://sbi-kyc.xyz") == (0, 1)
    assert link_risk("visit https://www.sbi.co.in") == (0, 0)


def test_a_lookalike_domain_is_flagged_by_its_real_tld():
    """
    "sbi.co.in-update.xyz" is designed to read as "sbi.co.in" at a glance.

    The TLD check must look at the END of the host, which is the part that decides
    where the browser actually goes.
    """
    assert link_risk("https://sbi.co.in-update.xyz") == (0, 1)
    assert link_risk("https://www.hdfcbank.com/login") == (0, 0)


# ---------------------------------------------------------------------------
# Structural features, which must read the RAW text
# ---------------------------------------------------------------------------
def test_caps_ratio_measures_shouting_on_raw_text():
    """
    Shouting must still be visible.

    This is why caps_ratio takes the RAW message: normalize() lowercases everything,
    so measuring after normalization would always return 0.
    """
    assert caps_ratio("URGENT") == 1.0
    assert caps_ratio("urgent") == 0.0
    assert caps_ratio("") == 0.0


def test_digit_ratio_measures_how_numeric_the_text_is():
    """Scam messages are dense with amounts, OTPs and reference numbers."""
    assert digit_ratio("1234") == 1.0
    assert digit_ratio("abcd") == 0.0
    assert digit_ratio("") == 0.0


# ---------------------------------------------------------------------------
# The entry function
# ---------------------------------------------------------------------------
def test_extract_returns_one_row_per_message():
    """F3 relies on the row order matching the input order."""
    frame = extract_text_features(["hello there", "Aapka KYC band ho jayega"])
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 2


def test_extract_has_the_columns_f1_depends_on():
    """F1's Bayesian network needs urgency, credential and link evidence."""
    frame = extract_text_features(["Aapka KYC turant verify karein: http://a.xyz"])
    for column in ["urgency_hits", "credential_hits", "has_suspicious_tld",
                   "cmi", "scam_lexicon_hits", "caps_ratio"]:
        assert column in frame.columns


def test_a_scam_message_scores_higher_than_a_chat_message():
    """The end-to-end sanity check: the features must point the right way."""
    scam = "URGENT! Aapka SBI KYC band ho jayega. Turant verify: http://sbi-kyc.xyz"
    chat = "Yaar kal movie chalein? Show 9 baje ka hai."
    rows = extract_text_features([scam, chat])
    assert rows.loc[0, "scam_lexicon_hits"] > rows.loc[1, "scam_lexicon_hits"]
    assert rows.loc[0, "urgency_hits"] > rows.loc[1, "urgency_hits"]
    assert rows.loc[0, "url_count"] > rows.loc[1, "url_count"]
    assert rows.loc[0, "caps_ratio"] > rows.loc[1, "caps_ratio"]


def test_empty_and_missing_text_do_not_crash():
    """pandas gives us NaN for blank cells, and real datasets contain them."""
    frame = extract_text_features(["", None])
    assert len(frame) == 2
    assert frame["n_tokens"].tolist() == [0, 0]
