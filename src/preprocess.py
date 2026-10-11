"""
Text normalization shared by every other file in this project.

WHAT: turns a raw message into a clean, comparable form.
      "SB1 KYC B@ND ho jayega!!!" -> "sbi kyc band ho jayega!!!"
WHY:  scammers deliberately misspell words to dodge keyword filters. They swap letters
      for look-alikes from other alphabets (homoglyphs), swap letters for digits
      (leetspeak), and hide invisible characters inside words. If we match our scam
      lexicon against the raw text, all of those tricks work. If we normalize first,
      most of them stop working. This file is also "defence config (b)" in F8.
HOW:  `normalize(text)` runs the five cleaning steps in order. `tokenize(text)` splits
      normalized text into word tokens.

IMPORTANT design decision
-------------------------
Normalization is for *lexical* matching (does this message say "kyc"?).
It is NOT used for *structural* features. F2 counts URLs, digits and CAPITALS on the
RAW text, because normalizing would destroy exactly the evidence those features need
(lowercasing erases SHOUTING, and un-leetspeaking "$499" would damage the amount).
"""

import re

# ---------------------------------------------------------------------------
# Step 1 data: homoglyphs (look-alike letters from other alphabets)
# ---------------------------------------------------------------------------
# A Cyrillic "а" (U+0430) looks identical to a Latin "a" on screen, but to a computer
# it is a completely different character, so "kyc" written with a Cyrillic "c" will
# never equal our lexicon's "kyc". We map each look-alike back to its Latin twin.
HOMOGLYPHS = {
    # Cyrillic lowercase
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "у": "y", "х": "x", "і": "i", "ѕ": "s", "к": "k",
    "в": "b", "н": "h", "м": "m", "т": "t",
    # Cyrillic uppercase
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C",
    "У": "Y", "Х": "X", "К": "K", "В": "B", "М": "M",
    # Greek lowercase
    "α": "a", "ο": "o", "ν": "v", "ρ": "p", "τ": "t",
    "ι": "i", "ε": "e", "κ": "k", "υ": "u", "ϲ": "c",
    # Greek uppercase
    "Α": "A", "Ο": "O", "Ε": "E", "Ρ": "P", "Τ": "T",
}

# ---------------------------------------------------------------------------
# Step 2 data: invisible characters
# ---------------------------------------------------------------------------
# These render as nothing at all. "k<ZWSP>yc" looks like "kyc" to a human but is a
# different string to a filter. We simply delete them.
ZERO_WIDTH_CHARS = [
    "​",  # zero width space
    "‌",  # zero width non-joiner
    "‍",  # zero width joiner
    "⁠",  # word joiner
    "﻿",  # byte order mark
    "­",  # soft hyphen
]

# ---------------------------------------------------------------------------
# Step 4 data: leetspeak
# ---------------------------------------------------------------------------
# Digit/symbol substitutions scammers use: "b@nk", "0tp", "acc0unt", "pa$$word".
LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}


def fold_homoglyphs(text):
    """
    Replace look-alike letters from other alphabets with their Latin twin.

    WHAT: "kус" (Cyrillic у and с) -> "kyc"
    WHY:  so a disguised word matches our plain-English scam lexicon.
    """
    # str.translate needs a dict keyed by the character's number, not the character.
    return text.translate({ord(bad): good for bad, good in HOMOGLYPHS.items()})


def strip_zero_width(text):
    """
    Delete invisible characters.

    WHAT: "k​yc" -> "kyc"
    WHY:  an invisible character splits a word in two for any keyword matcher,
          while a human reading the message sees nothing wrong.
    """
    for ch in ZERO_WIDTH_CHARS:
        text = text.replace(ch, "")
    return text


def undo_leetspeak_token(token):
    """
    Undo digit-for-letter swaps in ONE word, but only if it is really a word.

    WHAT: "b@nk" -> "bank",  "acc0unt" -> "account",  but "499" stays "499".
    WHY:  a token with no letters at all is a number we care about (an amount, an OTP,
          a phone number). Rewriting "10000" into "ioooo" would destroy the money
          features in F2, so we leave letter-free tokens completely alone.
    EXAMPLE: undo_leetspeak_token("0tp") -> "otp"
    """
    # any() over the characters: does this token contain at least one real letter?
    if not any(ch.isalpha() for ch in token):
        return token
    return "".join(LEET_MAP.get(ch, ch) for ch in token)


def undo_leetspeak(text):
    """
    Apply `undo_leetspeak_token` to every whitespace-separated chunk of the text.

    WHAT: "pay 500 to b@nk" -> "pay 500 to bank"
    WHY:  done per word so that the "is this a pure number?" test above can protect
          amounts while still cleaning up disguised words.
    """
    return " ".join(undo_leetspeak_token(chunk) for chunk in text.split())


def collapse_repeats(text):
    r"""
    Shorten any run of 3 or more identical NON-DIGIT characters down to one.

    WHAT: "urgeeeent!!!" -> "urgent!",  but "10,000" stays "10,000"
    WHY:  "URGEEENT" is padding meant to look loud and dodge exact matching.
          Two things are deliberately protected:
          1. DIGITS are never collapsed. "₹10,000" contains a run of three zeros, and
             squashing it to "₹10,0" would wreck the money features in F2. This is why
             the pattern below excludes \d.
          2. Runs of only TWO letters are left alone, because the "aa" in "khaata" or
             the "ee" in "jeet" are normal Romanized Hindi spellings. Matching those
             variants is F6's job (the phonetic layer), not ours.
    """
    # ([^\d\s]) = one character that is not a digit and not whitespace.
    # \1{2,}      = that same character, 2 or more times again (so 3+ in total).
    return re.sub(r"([^\d\s])\1{2,}", r"\1", text)


def normalize(text):
    """
    Run the whole cleaning pipeline on one message.

    WHAT: raw message in, clean lowercase message out.
    WHY:  one function that every other file calls, so the whole project cleans text
          the same way. In F8 this function is defence configuration (b).
    EXAMPLE: normalize("KYC B@ND ho jaaayega!!") -> "kyc band ho jaayega!"
    """
    if not isinstance(text, str):
        return ""                      # guard against NaN / None coming from pandas
    text = fold_homoglyphs(text)       # 1. disguised letters -> Latin
    text = strip_zero_width(text)      # 2. delete invisible characters
    text = text.lower()                # 3. SHOUTING -> shouting
    text = undo_leetspeak(text)        # 4. b@nk -> bank (numbers untouched)
    text = collapse_repeats(text)      # 5. urgeeeent -> urgent
    return re.sub(r"\s+", " ", text).strip()   # squeeze runs of spaces into one


def tokenize(text):
    """
    Split already-normalized text into word tokens.

    WHAT: "kyc band ho jayega" -> ["kyc", "band", "ho", "jayega"]
    WHY:  F2 tags each token as Hindi or English, and F6 gives each token a phonetic
          code, so both need a list of words rather than one long string.
    NOTE: [a-z0-9]+ keeps digits (so "₹5000" gives the token "5000") and throws away
          punctuation and currency symbols.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


# A tiny demo so you can see every step. Run: python -m src.preprocess
if __name__ == "__main__":
    examples = [
        "SBI KYC B@ND ho jaaayega!!! Turant verify karein: http://sbi-kyc.xyz",
        "Ааpka acc0unt bl0ck ho gaya hai",          # Cyrillic + leetspeak
        "k​y​c update karo",                         # hidden zero-width spaces
        "Pay ₹10,000 and $499 now",                       # amounts must survive
    ]
    print("RAW  ->  NORMALIZED  ->  TOKENS")
    print("=" * 78)
    for raw in examples:
        clean = normalize(raw)
        print(f"\nraw   : {raw!r}")
        print(f"clean : {clean!r}")
        print(f"tokens: {tokenize(clean)}")
