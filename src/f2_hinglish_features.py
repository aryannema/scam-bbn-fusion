# FUNCTIONALITY F2 - Multilingual code-mixed (Hindi-English) feature hybridization
"""
F2: turns one Hinglish message into a row of numbers a model can learn from.

WHAT: `extract_text_features(texts)` -> a pandas DataFrame, one row per message,
      with named columns like hindi_ratio, cmi, scam_lexicon_hits, url_count.
WHY:  Indian scam SMS is written in Hinglish - Romanized Hindi mixed with English in
      the same sentence ("Aapka KYC band ho jayega, turant verify karein"). An English
      spam model has never seen "band ho jayega". Instead of a huge multilingual
      neural model, we measure the MIXING ITSELF, plus the handful of word families
      that scams cannot avoid using.
HOW:  run `python -m src.f2_hinglish_features` for a demo table.

Two kinds of features, computed from two different versions of the text
-----------------------------------------------------------------------
* LEXICAL features ("does it say kyc?") come from preprocess.normalize(), so that
  disguises like "B@ND" and "k<invisible>yc" are undone first.
* STRUCTURAL features ("is it SHOUTING? how many digits?") come from the RAW text,
  because normalizing destroys exactly the evidence they measure - lowercasing erases
  the shouting we want to count.

The word lists below are deliberately small and hand-written. A professor can read
them in a minute and we can defend every entry. They are also the lists F6 reuses.
"""

import re

import pandas as pd

from src.preprocess import normalize, tokenize

# ---------------------------------------------------------------------------
# Romanized Hindi words. These are the everyday "glue" words of Hindi written in
# English letters. If a message contains many of these, it is at least partly Hindi.
# ---------------------------------------------------------------------------
HINDI_WORDS = {
    "aap", "aapka", "aapke", "aapko", "aapki", "hai", "hain", "ho", "hoga", "hogi",
    "jayega", "jayegi", "jaayega", "kar", "karo", "karein", "karen", "kare", "karna",
    "kiya", "kijiye", "nahi", "nahin", "mat", "warna", "turant", "turat", "jaldi",
    "abhi", "kripya", "paise", "paisa", "pese", "rupaye", "rupay", "khata", "khaata",
    "band", "bandh", "bnd", "bhej", "bhejo", "bhejein", "bhejna", "mila", "milega",
    "lena", "dena", "chahiye", "raha", "rahe", "rahi", "gaya", "gayi", "diya", "liya",
    "mera", "meri", "tera", "uska", "humara", "tumhara", "beta", "bhai", "behen",
    "mummy", "papa", "ghar", "kal", "aaj", "raat", "subah", "shaam", "baje", "din",
    "ka", "ke", "ki", "ko", "se", "me", "mein", "par", "aur", "ya", "toh", "bhi",
    "sab", "sabko", "kuch", "kya", "kaun", "kahan", "kyun", "ek", "do", "teen",
    "inaam", "jeeta", "sasta", "muft", "naya", "purana", "acha", "theek", "thik",
    "zaroori", "jaruri", "dhyan", "suno", "dekho", "batao", "batayein", "samjho",
    "shaadi", "doodh", "sabzi", "khana", "kha", "pani", "dawa", "bijli", "vibhag",
    "yaar", "chalein", "milte", "nikal", "lekar", "wala", "wali", "hua", "hui",
    "liye", "bina", "sirf", "bachane", "jaayein", "jayein", "kholo", "daalein",
}

# ---------------------------------------------------------------------------
# Common English words, for the other side of the comparison.
# ---------------------------------------------------------------------------
ENGLISH_WORDS = {
    "the", "is", "are", "was", "your", "you", "my", "our", "this", "that", "and",
    "for", "with", "from", "have", "has", "will", "can", "not", "all", "please",
    "account", "bank", "click", "link", "verify", "update", "urgent", "now", "today",
    "free", "win", "won", "prize", "offer", "claim", "pay", "payment", "card",
    "blocked", "block", "suspended", "expire", "expires", "security", "confirm",
    "details", "number", "call", "reply", "message", "dear", "customer", "sir",
    "madam", "thank", "thanks", "order", "delivery", "delivered", "parcel", "track",
    "meeting", "report", "team", "project", "email", "time", "day", "week", "month",
    "money", "refund", "cash", "loan", "job", "work", "home", "salary", "fee",
    "registration", "minimum", "guaranteed", "limited", "last", "chance", "final",
    "notice", "warning", "alert", "check", "login", "password", "immediately",
    "within", "hours", "minutes", "balance", "transaction", "failed", "successful",
    "shipped", "arrive", "available", "share", "anyone", "code", "valid", "request",
    # everyday chat words, so casual English inside a Hindi sentence is still seen
    "movie", "show", "lunch", "dinner", "office", "come", "coming", "going", "see",
    "good", "morning", "night", "ok", "okay", "yes", "no", "sorry", "love", "miss",
    "meet", "tomorrow", "friday", "sunday", "monday", "saturday", "exam", "class",
    "college", "phone", "market", "doctor", "hospital", "police", "emergency",
    "birthday", "party", "gift", "book", "books", "file", "files", "photo", "video",
    "match", "game", "ground", "station", "mall", "cafe", "shop", "price", "bill",
}

# ---------------------------------------------------------------------------
# Tokens that belong to NO language: brand names, abbreviations, pure numbers.
# "SBI", "OTP" and "UPI" are written the same way in a Hindi or an English sentence,
# so counting them as either one would distort the code-mixing measurement.
# ---------------------------------------------------------------------------
NEUTRAL_WORDS = {
    "otp", "kyc", "upi", "sms", "atm", "pin", "cvv", "id", "sbi", "hdfc", "icici",
    "axis", "pnb", "kotak", "rbi", "paytm", "ybl", "okaxis", "amazon", "flipkart",
    "netflix", "whatsapp", "google", "aadhaar", "aadhar", "pan", "fastag", "lpg",
    "itr", "tds", "http", "https", "www", "com", "in", "xyz", "rs", "inr",
}

# ---------------------------------------------------------------------------
# Scam lexicon: the words a scam cannot do its job without. Hindi AND English.
# F6 reuses this list, matching it by SOUND instead of spelling.
# ---------------------------------------------------------------------------
SCAM_LEXICON = {
    # English
    "kyc", "otp", "upi", "block", "blocked", "verify", "update", "expire", "expires",
    "suspend", "suspended", "deactivate", "refund", "lottery", "winner", "prize",
    "claim", "fee", "urgent", "password", "pin", "cvv", "aadhaar", "pan", "loan",
    "link", "click", "disconnect", "customs", "parcel", "credited", "cashback",
    # Romanized Hindi
    "band", "bandh", "turant", "jaldi", "khata", "khaata", "inaam", "paise", "bhej",
    "bhejein", "warna", "bijli", "muft", "sasta", "jeeta",
}

# Words that create time pressure. Urgency is the single most reliable scam signal,
# and it feeds the UrgencyCue node of the Bayesian network in F1.
URGENCY_WORDS = {
    "turant", "turat", "jaldi", "abhi", "urgent", "immediately", "now", "today",
    "warna", "hurry", "quickly", "final", "deadline", "expire", "expires", "expiring",
    "last", "limited", "soon", "before", "tonight", "aaj",
}

# Words that ask for a secret. This feeds the CredentialAsk node in F1.
CREDENTIAL_WORDS = {
    "otp", "pin", "password", "cvv", "kyc", "aadhaar", "aadhar", "pan", "upi",
    "netbanking", "login", "credentials", "details", "verify", "confirm",
}

# Link shorteners hide the real destination, so the reader cannot judge it.
SHORTENER_DOMAINS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "rb.gy", "cutt.ly"}

# Cheap top-level domains, heavily used by phishing pages. A real bank uses .co.in.
SUSPICIOUS_TLDS = (".xyz", ".top", ".tk", ".ml", ".ga", ".cf", ".click", ".work")

# How we recognise a link. Two shapes, because scam SMS use both:
#   1. a normal link that starts with http:// or www.
#   2. a BARE domain with no prefix at all, like "cashback-amazon.in" - very common in
#      real scam messages, and completely invisible to an http-only pattern.
# The TLD list in shape 2 is explicit so that ordinary sentence text does not match.
URL_PATTERN = (
    r"(?:https?://|www\.)\S+"
    r"|\b[a-z0-9][a-z0-9.\-]*\.(?:com|in|net|org|info|xyz|top|tk|ml|ga|cf|click|work|ly|io|co)\b(?:/\S*)?"
)


def tag_token(token):
    """
    Decide whether one word is Hindi, English, or neither.

    WHAT: "aapka" -> "hindi", "account" -> "english", "sbi" -> "neutral",
          "lonavala" -> "unknown"
    WHY:  every code-mixing measurement starts from per-word language labels.
    NOTE: "neutral" and "unknown" are different on purpose. "neutral" means the word
          genuinely belongs to no language (a brand, an abbreviation, a number).
          "unknown" means our hand-written lists simply do not contain it. Keeping
          them apart lets us REPORT how much of a message we could not tag, instead
          of quietly pretending it was language-free.
    """
    if token in NEUTRAL_WORDS or token.isdigit():
        return "neutral"
    if token in HINDI_WORDS:
        return "hindi"
    if token in ENGLISH_WORDS:
        return "english"
    return "unknown"


def tag_tokens(tokens):
    """
    Label every word in a message.

    WHAT: ["aapka","account"] -> ["hindi","english"]
    WHY:  separated from tag_token so both are trivial to test.
    """
    return [tag_token(token) for token in tokens]


def code_mixing_index(tags):
    """
    The Code-Mixing Index (CMI): how evenly two languages are mixed, 0 to 100.

    WHAT: all one language -> 0. A perfect 50/50 mix -> 50.
    WHY:  this is the standard published measure for code-mixed text (Das & Gamback).
          A pure English scam and a pure Hindi chat both score 0; a Hinglish scam
          scores high. It is the one number that says "this text is code-mixed".
    HOW:  100 * (1 - most_common_language / language_carrying_words).
          Neutral words (SBI, OTP, numbers) and words our lists do not know are both
          excluded, because neither can be assigned to a language.
    LIMITATION: because unknown words are excluded, CMI is a LOWER BOUND on the real
          mixing. A message full of English words missing from our list looks less
          mixed than it is, which is why we also report unknown_ratio as a feature.
    """
    hindi = tags.count("hindi")
    english = tags.count("english")
    total = hindi + english
    if total == 0:                      # nothing language-bearing: no mixing to report
        return 0.0
    return 100.0 * (1 - max(hindi, english) / total)


def count_script_switches(tags):
    """
    How many times the message flips between Hindi and English.

    WHAT: hindi, hindi, english, hindi -> 2 switches
    WHY:  CMI says *how much* of each language; this says how *interleaved* they are.
          "Aapka account band, verify karein" switches repeatedly - very Hinglish.
          Neutral and unknown words are skipped, so "SBI" sitting in the middle of a
          Hindi phrase does not hide a switch that is not really there.
    """
    languages = [tag for tag in tags if tag in ("hindi", "english")]
    return sum(1 for a, b in zip(languages, languages[1:]) if a != b)


def count_lexicon_hits(tokens, lexicon):
    """
    How many words of this message appear in the given word list.

    WHAT: count_lexicon_hits(["kyc","band"], SCAM_LEXICON) -> 2
    WHY:  one small function serves the scam, urgency and credential lists.
    """
    return sum(1 for token in tokens if token in lexicon)


def find_money_amounts(text):
    """
    Find amounts of money, and only amounts of money.

    WHAT: "Rs 10,000 and $499" -> [10000, 499]
    WHY:  scams name a sum - a prize to win or a fee to pay.
    CAREFUL: we require a currency marker (Rs / INR / the rupee or dollar sign). A bare
          6-digit number is far more likely to be an OTP or a phone number, and counting
          "482913" as nearly five lakh rupees would be nonsense. Bare digit runs get
          their own separate feature instead.
    """
    pattern = r"(?:₹|rs\.?|inr|\$)\s*([\d,]+)"
    amounts = []
    for found in re.findall(pattern, text, flags=re.IGNORECASE):
        digits = found.replace(",", "").strip()
        if digits.isdigit():
            amounts.append(int(digits))
    return amounts


def extract_hosts(text):
    """
    Pull the host name out of every link in the message.

    WHAT: "go to https://www.flipkart.com/orders" -> ["flipkart.com"]
    WHY:  THIS FUNCTION EXISTS BECAUSE OF A REAL BUG. We first judged links by asking
          `"t.co" in text`, which is True for "flipkart.com" - so every Flipkart
          tracking link was labelled a shortener. Comparing the HOST, not the whole
          message, is the fix.
    HOW:  take the part after "://", cut at the first "/", drop a leading "www." and
          any trailing punctuation from the end of a sentence.
    """
    hosts = []
    for url in re.findall(URL_PATTERN, text, flags=re.IGNORECASE):
        host = url.split("://")[-1].split("/")[0].lower()
        host = host.split("?")[0].split(":")[0].strip(".,!)")
        if host.startswith("www."):
            host = host[4:]
        if host:
            hosts.append(host)
    return hosts


def count_urls(text):
    """
    Count web links in the message.

    WHAT: "go to http://a.xyz and cashback-amazon.in" -> 2
    WHY:  almost every phishing message needs the victim to open a link.
    NOTE: the pattern also catches links written WITHOUT "http://", like
          "cashback-amazon.in", because real scam SMS very often do that and an
          http-only pattern missed them completely.
    """
    return len(re.findall(URL_PATTERN, text, flags=re.IGNORECASE))


def link_risk(text):
    """
    Judge how dangerous the links look, without visiting them.

    WHAT: returns (has_shortener, has_suspicious_tld) as 0/1 integers.
    WHY:  feeds the LinkRisk node of the Bayesian network in F1. A shortener hides
          where you are really going; a .xyz domain is a throwaway a bank would never
          use. Both are judgements a human makes at a glance, so we encode them.
          Note what this does NOT do: it never says "a link is bad". Benign delivery
          and bank messages contain links too, so the useful signal is the KIND of
          link, not its presence.
    """
    hosts = extract_hosts(text)
    shortener = int(any(host in SHORTENER_DOMAINS for host in hosts))
    suspicious = int(any(host.endswith(tld) for host in hosts for tld in SUSPICIOUS_TLDS))
    return shortener, suspicious


def caps_ratio(raw_text):
    """
    What share of the letters are CAPITALS, measured on the RAW text.

    WHAT: "URGENT now" -> 0.6
    WHY:  shouting is a pressure tactic. This must use the raw text, because
          normalize() lowercases everything and would always return 0.
    """
    letters = [character for character in raw_text if character.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for character in letters if character.isupper()) / len(letters)


def digit_ratio(raw_text):
    """
    What share of all characters are digits, measured on the RAW text.

    WHAT: "pay 500" -> 0.43
    WHY:  scam messages are dense with numbers: amounts, OTPs, phone numbers,
          reference ids. Benign chat usually is not.
    """
    if not raw_text:
        return 0.0
    return sum(1 for character in raw_text if character.isdigit()) / len(raw_text)


def language_features(tags, n_tokens):
    """
    The code-mixing half of the feature row.

    WHAT: {"hindi_ratio": 0.46, "cmi": 14.3, "script_switches": 2, ...}
    WHY:  split out of extract_one so that each function does one job and stays short
          enough to read in one go.
    """
    return {
        "hindi_ratio": tags.count("hindi") / n_tokens,
        "english_ratio": tags.count("english") / n_tokens,
        "unknown_ratio": tags.count("unknown") / n_tokens,   # how much we could not tag
        "cmi": code_mixing_index(tags),
        "script_switches": count_script_switches(tags),
    }


def structural_features(raw):
    """
    The features that must be measured on the RAW, un-normalized message.

    WHAT: {"caps_ratio": 0.29, "digit_ratio": 0.07, "exclamations": 3, ...}
    WHY:  normalize() lowercases the text, so measuring shouting afterwards would
          always give 0. Keeping these four together makes that reason obvious.
    """
    return {
        "caps_ratio": caps_ratio(raw),
        "digit_ratio": digit_ratio(raw),
        "exclamations": raw.count("!"),
        "text_length": len(raw),
    }


def extract_one(raw_text, normalizer=normalize):
    """
    Turn ONE message into a dictionary of named features.

    WHAT: a raw message string -> {"n_tokens": 9, "cmi": 48.0, ...}
    WHY:  this is the per-message core; extract_text_features just runs it over a list.
    ARGS: normalizer - how to clean the text before the lexical features are read.
          Defaults to the full pipeline. F8 swaps in a weaker one to measure what the
          cleaning is actually worth against a disguised message; without this hook,
          "no defence" could not be measured at all.
    EXAMPLE: extract_one("Aapka KYC band ho jayega")["hindi_ratio"] -> 0.8
    """
    # pandas hands us NaN (a float) for a blank cell, and the structural features
    # iterate over characters, so coerce to a string ONCE here rather than guarding in
    # every helper.
    raw = raw_text if isinstance(raw_text, str) else ""
    clean = normalizer(raw)                      # lexical features use cleaned text
    tokens = tokenize(clean)
    tags = tag_tokens(tokens)
    amounts = find_money_amounts(clean)
    shortener, suspicious_tld = link_risk(clean)
    row = {"n_tokens": len(tokens)}
    row.update(language_features(tags, max(len(tokens), 1)))   # max() avoids 0 division
    row.update({
        "scam_lexicon_hits": count_lexicon_hits(tokens, SCAM_LEXICON),
        "urgency_hits": count_lexicon_hits(tokens, URGENCY_WORDS),
        "credential_hits": count_lexicon_hits(tokens, CREDENTIAL_WORDS),
        "money_mentions": len(amounts),
        "max_amount": max(amounts) if amounts else 0,
        "long_digit_runs": len(re.findall(r"\d{5,}", clean)),   # OTPs, phone numbers
        "url_count": count_urls(clean),
        "has_shortener": shortener,
        "has_suspicious_tld": suspicious_tld,
    })
    row.update(structural_features(raw))
    return row


# FUNCTIONALITY F2 - main entry function
def extract_text_features(texts, normalizer=normalize):
    """
    Turn many messages into a feature table.

    WHAT: a list (or pandas Series) of messages -> DataFrame, one row per message.
    WHY:  this is the function F3 and F4 call. The column order is fixed, so the
          feature names in our results tables always mean the same thing.
    ARGS: normalizer - see extract_one. F8 uses it to build the "no defence" baseline.
    EXAMPLE: extract_text_features(["Aapka KYC band ho jayega"]).shape -> (1, 19)
    """
    return pd.DataFrame([extract_one(text, normalizer) for text in texts])


# A demo you can read out loud. Run: python -m src.f2_hinglish_features
if __name__ == "__main__":
    samples = [
        "Aapka SBI KYC band ho jayega! Turant verify karein: http://sbi-kyc.xyz",
        "URGENT: Your account is BLOCKED. Click http://bit.ly/3xKyc9 to verify now",
        "482913 is your OTP for SBI login. Kisi ke saath share na karein.",
        "Yaar kal movie chalein? Show 9 baje ka hai.",
    ]
    table = extract_text_features(samples)
    table.insert(0, "message", [text[:42] + "..." for text in samples])
    pd.set_option("display.width", 200)
    print("F2 FEATURES (first columns)")
    print("=" * 110)
    print(table[["message", "n_tokens", "hindi_ratio", "english_ratio", "unknown_ratio",
                 "cmi", "script_switches"]].round(2).to_string(index=False))
    print("\nSCAM-SIGNAL COLUMNS")
    print("=" * 110)
    print(table[["message", "scam_lexicon_hits", "urgency_hits", "credential_hits",
                 "money_mentions", "url_count", "has_shortener", "has_suspicious_tld",
                 "caps_ratio"]].round(2).to_string(index=False))
    print("\nNOTE how row 3 (a GENUINE bank OTP) has credential words but no link:")
    print("     that is exactly the message a naive keyword filter gets wrong.")
