# FUNCTIONALITY F6 - Phonetic embedding layer for Romanized Hindi
"""
F6: matches Hinglish words by how they SOUND, not how they are spelled.

WHAT: `phonetic_code(word)` gives a short code such that every sensible spelling of
      one Hindi word lands on the same code:
          turant / turrant / turat / tarant  ->  all "trt"
WHY:  Hindi has no official English spelling. A scammer writing "turant" (= at once)
      may type turant, turrant, turat or tarant - and all four defeat an exact-match
      keyword filter. Adding more spellings to the lexicon by hand is hopeless; there
      are too many. Instead we compare SOUNDS, so one lexicon entry covers them all.
      This is also defence config (c) in F8.
HOW:  run `python -m src.f6_phonetic` for the demo table.

How the encoder works, in one breath
------------------------------------
Soundex (1918) keeps a word's first letter and throws away its vowels, so "Robert"
and "Rupert" match. That is not enough for Romanized Hindi, so we add rules for the
ways Hindi sounds get written in English letters: "ph" and "f" are one sound, "v" and
"w" are one sound, "aa" is just a long "a", and the little nasal sound in "turant" is
often not written at all. Those extra rules are the whole contribution of this file.

Reference: this is the same idea as the Soundex variants used for code-mixed Indian
text (see docs/REFERENCES.md). We write our own 30-line version rather than depend on
a library, so we can explain every rule.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.f2_hinglish_features import SCAM_LEXICON
from src.preprocess import normalize, tokenize

# ---------------------------------------------------------------------------
# Step 1: pairs of letters that spell ONE sound. Order matters, so this is a list.
# ---------------------------------------------------------------------------
DIGRAPH_RULES = [
    # Aspirated consonants: Hindi writes these as a letter plus "h". English speakers
    # hear roughly the plain consonant, and people drop the "h" freely when typing.
    ("chh", "c"), ("ch", "c"), ("sh", "s"), ("ph", "f"), ("kh", "k"), ("gh", "g"),
    ("th", "t"), ("dh", "d"), ("bh", "b"), ("jh", "j"), ("ck", "k"),
    # Long vowels: "khaata" and "khata" are the same word written two ways.
    ("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"),
    # Diphthongs: "paise" and "pese" are the same word.
    ("ai", "e"), ("ei", "e"), ("au", "o"), ("ou", "u"),
]

# ---------------------------------------------------------------------------
# Step 2: single letters that stand for the same sound as another letter.
# ---------------------------------------------------------------------------
SINGLE_LETTER_MAP = {
    "w": "v",   # "varna" / "warna"
    "z": "j",   # "zaroori" / "jaroori"
    "q": "k",
    "x": "k",
    "y": "i",   # y is used as a vowel here: "kyc" sounds like "kaisee"
}

VOWELS = set("aeiou")

# A code shorter than this is too generic to trust, so we ignore it when matching.
#
# WHY 3, and not 2 - we measured this on our own dataset:
#   with a floor of 2, codes like "lk" (from "link") also match the everyday word
#   "like", "bd" (from "band") matches "bad" and "bed", and "bj" (from "bhej") matches
#   "baje". The result was 1,865 false hits in benign messages, and the phonetic-only
#   signal became worthless: 0.32 extra hits per benign message versus 0.35 per scam
#   message, a ratio of 1.1x.
#   With a floor of 3 the same comparison is 0.042 versus 0.129, a ratio of 3.1x.
#
# THE TRADE-OFF, stated honestly: a floor of 3 excludes short scam words from phonetic
# matching - kyc->"kc", upi->"up", band->"bd", pin->"pn", khata->"kt". Those are not
# lost, because F2 still matches them by exact spelling. The division of labour is:
#   F2 catches short words exactly; F6 catches the longer words where people actually
#   misspell (turant->"trt" survives, which is our flagship example).
MIN_CODE_LENGTH = 3


def _keep_letters(word):
    """
    Throw away anything that is not a letter and lowercase the rest.

    WHAT: "KYC!" -> "kyc"
    WHY:  punctuation and digits carry no sound.
    """
    return "".join(character for character in word.lower() if character.isalpha())


def _apply_digraphs(word):
    """
    Replace every two-letter spelling of one sound with a single letter.

    WHAT: "khaata" -> "kata"  ("kh" -> "k", "aa" -> "a")
    WHY:  this is the rule that makes Hindi spelling variants collapse together.
    """
    for pair, replacement in DIGRAPH_RULES:
        word = word.replace(pair, replacement)
    return word


def _apply_single_letters(word):
    """
    Map single letters that share a sound onto one representative.

    WHAT: "warna" -> "varna",  "zaroori" -> "jaroori"
    WHY:  Hindi speakers write these pairs interchangeably in English letters.
    """
    return "".join(SINGLE_LETTER_MAP.get(character, character) for character in word)


def _drop_nasal_before_consonant(word):
    """
    Delete an "n" or "m" that sits in front of another consonant.

    WHAT: "turant" -> "turat",  "band" -> "bad"
    WHY:  THIS IS THE MOST IMPORTANT RULE FOR HINDI. Hindi marks a small nasal sound
          with a dot above the letter (anusvara). English letters have no dot, so
          people simply leave it out: "turant" and "turat" are the same word, and so
          are "band" and "bad", "chand" and "chad".
    """
    letters = list(word)
    kept = []
    for position, character in enumerate(letters):
        following = letters[position + 1] if position + 1 < len(letters) else ""
        is_nasal = character in ("n", "m")
        before_consonant = following != "" and following not in VOWELS
        if is_nasal and before_consonant:
            continue                       # drop this nasal
        kept.append(character)
    return "".join(kept)


def _squeeze_doubles(word):
    """
    Collapse a letter repeated twice into one.

    WHAT: "turrant" -> "turant"
    WHY:  doubling a letter is a typing habit, not a different sound.
    """
    squeezed = []
    for character in word:
        if not squeezed or squeezed[-1] != character:
            squeezed.append(character)
    return "".join(squeezed)


def _drop_inner_vowels(word):
    """
    Keep the first letter, then drop every vowel after it.

    WHAT: "turat" -> "trt"
    WHY:  this is the original Soundex trick. Vowels are the least reliably written
          part of a Romanized word, so dropping them removes most of the variation.
          The first letter is kept because a word's opening sound is rarely wrong.
    """
    if not word:
        return ""
    return word[0] + "".join(c for c in word[1:] if c not in VOWELS)


# FUNCTIONALITY F6 - the core function
def phonetic_code(word):
    """
    Turn one word into its phonetic code.

    WHAT: phonetic_code("turrant") -> "trt"
    WHY:  two words with the same code are treated as the same word by our lexicon.
    EXAMPLE: turant, turrant, turat and tarant all return "trt".
    HOW:  the seven steps below, in this order. Each step is its own tiny function
          above, so you can run them one at a time and watch the word change.
    """
    word = _keep_letters(word)                   # 1. letters only, lowercase
    word = _apply_digraphs(word)                 # 2. kh->k, aa->a, ai->e ...
    word = _apply_single_letters(word)           # 3. w->v, z->j, y->i
    word = _squeeze_doubles(word)                # 4. turrant -> turant
    word = _drop_nasal_before_consonant(word)    # 5. turant -> turat
    word = _drop_inner_vowels(word)              # 6. turat -> trt
    word = _squeeze_doubles(word)                # 7. tidy up any new doubles
    return word[:6]                              # long codes add nothing useful


def phonetic_codes(tokens):
    """
    Code a whole list of words.

    WHAT: ["turant","kyc"] -> ["trt","kk"]
    WHY:  used both for features and for the bag-of-codes vector.
    """
    return [phonetic_code(token) for token in tokens]


def _build_lexicon_codes():
    """
    Pre-compute the phonetic code of every scam word, once, at import time.

    WHAT: returns a set like {"kk", "trt", "bd", ...}
    WHY:  F2's lexicon is matched by exact spelling. Here we match the same lexicon by
          sound, which is the entire point of F6. Codes shorter than MIN_CODE_LENGTH
          are dropped because a 1-letter code matches far too many innocent words.
    """
    codes = set()
    for word in SCAM_LEXICON:
        code = phonetic_code(word)
        if len(code) >= MIN_CODE_LENGTH:
            codes.add(code)
    return codes


SCAM_LEXICON_CODES = _build_lexicon_codes()


def phonetic_hits(text):
    """
    Count scam words found by SOUND, and how many exact matching missed.

    WHAT: returns (phonetic_scam_hits, phonetic_only_hits).
    WHY:  `phonetic_only_hits` is the number that proves F6 earns its place: those are
          scam words that F2's exact matching did NOT find, and F6 did. If this is
          always zero, the phonetic layer is adding nothing.
    EXAMPLE: "turrant verifiy karo" -> exact finds nothing, phonetic finds 2.
    """
    tokens = tokenize(normalize(text))
    total = 0
    missed_by_exact = 0
    for token, code in zip(tokens, phonetic_codes(tokens)):
        if len(code) >= MIN_CODE_LENGTH and code in SCAM_LEXICON_CODES:
            total += 1
            if token not in SCAM_LEXICON:          # exact match would have missed it
                missed_by_exact += 1
    return total, missed_by_exact


def codes_as_text(text):
    """
    Rewrite a message as a string of phonetic codes.

    WHAT: "Turant verify karo" -> "trt vrf kr"
    WHY:  CountVectorizer expects text. Feeding it codes instead of words gives us a
          spelling-proof bag of words, which is the "embedding" part of this
          functionality.
    """
    return " ".join(phonetic_codes(tokenize(normalize(text))))


def build_code_vectorizer(texts, max_features=200):
    """
    Build the bag-of-phonetic-codes matrix used by the feature model in F4.

    WHAT: returns (matrix, vectorizer). The matrix has one row per message and one
          column per common phonetic code.
    WHY:  a normal bag of words treats "turant" and "turrant" as two unrelated
          columns, so each gets too few examples to learn from. Coding first merges
          them into one strong column.
    NOTE: we cap the vocabulary at max_features so the table stays small enough to
          read and fast enough to train on.
    """
    vectorizer = CountVectorizer(max_features=max_features, token_pattern=r"\S+")
    matrix = vectorizer.fit_transform([codes_as_text(text) for text in texts])
    return matrix, vectorizer


# FUNCTIONALITY F6 - main entry function
def extract_phonetic_features(texts):
    """
    Turn many messages into the two phonetic feature columns.

    WHAT: a list of messages -> DataFrame with phonetic_scam_hits and
          phonetic_only_hits, one row per message.
    WHY:  F3 glues these columns onto F2's text features.
    EXAMPLE: extract_phonetic_features(["turrant kyc"])["phonetic_only_hits"] -> 1
    """
    rows = []
    for text in texts:
        total, only = phonetic_hits(text)
        rows.append({"phonetic_scam_hits": total, "phonetic_only_hits": only})
    return pd.DataFrame(rows)


# Demo. Run: python -m src.f6_phonetic
if __name__ == "__main__":
    print("THE POINT OF F6: one word, four spellings, one code")
    print("=" * 62)
    for group in [["turant", "turrant", "turat", "tarant"],
                  ["kyc", "keyc"],
                  ["khata", "khaata", "khatha"],
                  ["band", "bnd", "bandh"],
                  ["verify", "varify", "verifiy"],
                  ["paise", "paisa", "pese"],
                  ["warna", "varna"]]:
        codes = [phonetic_code(word) for word in group]
        verdict = "SAME" if len(set(codes)) == 1 else "DIFFERENT"
        pairs = ",  ".join(f"{w} -> {c}" for w, c in zip(group, codes))
        print(f"  [{verdict:9s}] {pairs}")

    print("\nWHAT F6 CATCHES THAT EXACT MATCHING MISSES")
    print("=" * 62)
    examples = [
        "Aapka khaata bnd ho jayega, turrant varify karein",   # all misspelled
        "Aapka khata band ho jayega, turant verify karein",     # all spelled normally
        "Yaar kal movie chalein? Show 9 baje ka hai.",          # benign
    ]
    for text in examples:
        exact = sum(1 for t in tokenize(normalize(text)) if t in SCAM_LEXICON)
        total, only = phonetic_hits(text)
        print(f"\n  {text[:55]}")
        print(f"    exact-match hits : {exact}")
        print(f"    phonetic hits    : {total}  (of which {only} exact matching missed)")
    print(f"\nlexicon: {len(SCAM_LEXICON)} words -> {len(SCAM_LEXICON_CODES)} codes long enough to"
          f" match (minimum {MIN_CODE_LENGTH} characters)")
    short = sorted(w for w in SCAM_LEXICON if len(phonetic_code(w)) < MIN_CODE_LENGTH)
    print(f"short words left to F2's exact matching ({len(short)}): {', '.join(short)}")
