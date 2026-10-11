# FUNCTIONALITY F8 - Adversarial robustness module
"""
F8: attack our own detector the way a real scammer would, and measure the damage.

WHAT: `run_attack_evaluation()` perturbs held-out scam messages five ways and reports
      how much recall each attack destroys, under three defence settings.
WHY:  a detector is only as good as its behaviour against someone trying to beat it.
      Scammers iterate: when "kyc" gets blocked they send "k​yc", "KYC" with a
      Cyrillic C, or "keyc". Reporting clean accuracy alone would be misleading.
HOW:  run `python -m src.f8_adversarial`.

The three defence settings, which are the whole point
------------------------------------------------------
  (a) none      - lexical features read the text almost as it arrives (lowercase only)
  (b) normalize - preprocess.normalize() undoes the disguises first
  (c) normalize + phonetic - as (b), plus F6's sound-based matching

If (b) beats (a), the normalizer earns its place. If (c) beats (b), the phonetic layer
earns its place. If they do not, we say so.

A note on what is being attacked
--------------------------------
These attacks change TEXT, so they can only affect text features. The graph features
from F3 are untouched - a scammer rewriting their message does not change who they
sent it to. So this file deliberately attacks the TEXT-ONLY detector: that is where
the attack surface is, and mixing in graph features would hide the damage behind
evidence the attacker cannot touch. That resilience is itself an argument for fusion,
and we say so in the report rather than hiding the text model's weakness.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score

from src.config import (COMBINED_CSV, FRAUD_THRESHOLD, OUTPUTS_DIR, RANDOM_SEED)
from src.f2_hinglish_features import SCAM_LEXICON, extract_text_features
from src.f6_phonetic import extract_phonetic_features
from src.preprocess import normalize

# Latin letters and the look-alikes from other alphabets that scammers substitute.
HOMOGLYPH_SWAPS = {"a": "а", "e": "е", "o": "о", "c": "с",
                   "p": "р", "y": "у", "x": "х", "i": "і"}

# Letter-to-digit substitutions. Readable to a human, invisible to a keyword matcher.
LEET_SWAPS = {"o": "0", "i": "1", "e": "3", "a": "@", "s": "$"}

# Characters that render as nothing at all.
INVISIBLE = ["​", "‌", "‍"]

# Romanized Hindi respellings - the attack F6 is specifically built to survive.
RESPELLINGS = {"turant": "turrant", "kyc": "keyc", "khata": "khaata",
               "band": "bnd", "verify": "varify", "jaldi": "jaldee",
               "paise": "pese", "warna": "varna", "bhejein": "bhejien"}


def homoglyph_attack(text, rng):
    """
    Swap a few letters for identical-looking letters from other alphabets.

    WHAT: "bank" -> "bаnk" (a Cyrillic a)
    WHY:  it looks the same to the victim and is a different string to a filter.
    """
    characters = list(text)
    targets = [i for i, c in enumerate(characters) if c.lower() in HOMOGLYPH_SWAPS]
    for index in rng.permutation(targets)[:max(1, len(targets) // 3)]:
        characters[index] = HOMOGLYPH_SWAPS[characters[index].lower()]
    return "".join(characters)


def leetspeak_attack(text, rng):
    """
    Replace letters with look-alike digits and symbols.

    WHAT: "account" -> "acc0unt",  "bank" -> "b@nk"
    WHY:  the oldest trick there is, and still effective against exact matching.
    """
    characters = list(text)
    targets = [i for i, c in enumerate(characters) if c.lower() in LEET_SWAPS]
    for index in rng.permutation(targets)[:max(1, len(targets) // 3)]:
        characters[index] = LEET_SWAPS[characters[index].lower()]
    return "".join(characters)


def zero_width_attack(text, rng):
    """
    Insert invisible characters inside words.

    WHAT: "kyc" -> "k​yc"
    WHY:  the message looks untouched to the reader, but every keyword is broken in
          half for anything matching on strings.
    """
    characters = list(text)
    positions = sorted(rng.permutation(len(characters))[:max(1, len(characters) // 12)],
                       reverse=True)
    for index in positions:
        characters.insert(int(index), str(rng.choice(INVISIBLE)))
    return "".join(characters)


def typo_attack(text, rng):
    """
    Swap neighbouring letters, like a typing slip.

    WHAT: "verify" -> "vreify"
    WHY:  a scammer can send sloppy text on purpose. A human reads straight past it;
          a model that learned exact words does not.
    """
    characters = list(text)
    if len(characters) < 4:
        return text
    for _ in range(max(1, len(characters) // 25)):
        index = int(rng.integers(1, len(characters) - 2))
        characters[index], characters[index + 1] = characters[index + 1], characters[index]
    return "".join(characters)


def respell_attack(text, rng):
    """
    Respell Romanized Hindi words the way a real sender might.

    WHAT: "turant verify" -> "turrant varify"
    WHY:  THIS IS THE ATTACK F6 EXISTS FOR. It needs no tricks at all - just one of
          the several legitimate spellings of the same Hindi word.
    """
    words = []
    for word in text.split():
        stripped = word.lower().strip(".,!:")
        words.append(RESPELLINGS.get(stripped, word))
    return " ".join(words)


ATTACKS = {
    "homoglyphs": homoglyph_attack,
    "leetspeak": leetspeak_attack,
    "zero-width": zero_width_attack,
    "typos": typo_attack,
    "hindi respellings": respell_attack,
}


def weak_normalizer(text):
    """
    The "no defence" baseline: lowercase, and nothing else.

    WHAT: "B@NK" -> "b@nk"  (the disguise survives)
    WHY:  we need a floor to measure against. Lowercasing alone is what a naive
          keyword filter does, so this is the realistic unprotected starting point.
    """
    return text.lower() if isinstance(text, str) else ""


def features_for(texts, defence):
    """
    Build text features under one of the three defence settings.

    WHAT: returns a DataFrame of features.
    WHY:  the three settings differ ONLY here, so the comparison is fair everywhere else.
    """
    if defence == "none":
        return extract_text_features(texts, normalizer=weak_normalizer)
    text = extract_text_features(texts, normalizer=normalize)
    if defence == "normalize":
        return text
    phonetic = extract_phonetic_features(texts)          # defence "normalize+phonetic"
    return pd.concat([text.reset_index(drop=True), phonetic.reset_index(drop=True)],
                     axis=1)


def train_text_detector(texts, labels, defence):
    """
    Train the text-only detector under one defence setting.

    WHAT: returns the fitted model.
    WHY:  each defence changes what the model can see, so each needs its own training.
    """
    model = GradientBoostingClassifier(random_state=RANDOM_SEED, n_estimators=120)
    return model.fit(features_for(texts, defence).to_numpy(), labels)


def recall_on(model, texts, labels, defence):
    """
    What share of scam messages does the detector still catch?

    WHAT: a number between 0 and 1.
    WHY:  recall is the metric an attacker tries to destroy - their goal is to get
          through, which means being missed.
    """
    probability = model.predict_proba(features_for(texts, defence).to_numpy())[:, 1]
    return recall_score(labels, (probability >= FRAUD_THRESHOLD).astype(int),
                        zero_division=0)


def attack_many(texts, attack, seed=RANDOM_SEED):
    """
    Apply one attack to a whole list of messages.

    WHAT: returns the perturbed list.
    WHY:  a fixed seed keeps the attack reproducible, so the numbers in the report can
          be regenerated exactly.
    """
    rng = np.random.default_rng(seed)
    return [attack(str(text), rng) for text in texts]


# FUNCTIONALITY F8 - main entry function
def run_attack_evaluation():
    """
    Run every attack against every defence and build the results table.

    WHAT: returns a DataFrame indexed by attack, with one column per defence, holding
          recall on the attacked messages. Also writes a CSV and a bar chart.
    WHY:  this is the robustness result in the paper, and the evidence for whether
          F6's phonetic layer is worth including.
    EXAMPLE: python -m src.f8_adversarial
    """
    messages = pd.read_csv(COMBINED_CSV)
    is_train = (messages["split_leaksafe"] == "train").to_numpy()
    train, test = messages[is_train], messages[~is_train]
    scam_test = test[test["label"] == 1]
    defences = ["none", "normalize", "normalize+phonetic"]
    rows = {}
    for defence in defences:
        model = train_text_detector(train["text"], train["label"].to_numpy(), defence)
        clean = recall_on(model, scam_test["text"], scam_test["label"], defence)
        column = {"clean (no attack)": clean}
        for name, attack in ATTACKS.items():
            attacked = attack_many(scam_test["text"].tolist(), attack)
            column[name] = recall_on(model, attacked, scam_test["label"], defence)
        rows[defence] = column
    table = pd.DataFrame(rows)
    table["attack success vs none"] = _attack_success(table, "none")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUTS_DIR / "f8_adversarial.csv")
    save_chart(table[defences], OUTPUTS_DIR / "f8_adversarial.png")
    return table


def _attack_success(table, defence):
    """
    The attacker's own metric: what fraction of caught messages got through.

    WHAT: (clean recall - attacked recall) / clean recall, per attack.
    WHY:  recall alone understates the damage. "Recall fell from 0.95 to 0.60" is
          clearer stated as "the attack got 37% of blocked messages through".
    """
    clean = table.loc["clean (no attack)", defence]
    return ((clean - table[defence]) / max(clean, 1e-9)).clip(lower=0)


def save_chart(table, path):
    """
    Draw the recall-under-attack bar chart.

    WHAT: writes a PNG and returns its path.
    WHY:  three defences across six conditions is far easier to read as bars.
    """
    axis = table.plot.bar(figsize=(10, 5), rot=20, width=0.8)
    axis.set_ylabel("recall on scam messages")
    axis.set_title("F8: how much recall survives each attack, by defence")
    axis.axhline(FRAUD_THRESHOLD, linestyle=":", color="grey", linewidth=1)
    axis.legend(title="defence", loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    return path


# Demo. Run: python -m src.f8_adversarial
if __name__ == "__main__":
    pd.set_option("display.width", 150)
    print("ONE MESSAGE, FIVE ATTACKS")
    print("=" * 78)
    original = "Aapka SBI KYC band ho jayega. Turant verify karein: http://sbi-kyc.xyz"
    rng = np.random.default_rng(RANDOM_SEED)
    print(f"  original          : {original}")
    for name, attack in ATTACKS.items():
        print(f"  {name:18s}: {attack(original, rng)}")

    print("\nRECALL ON HELD-OUT SCAM MESSAGES, BY ATTACK AND DEFENCE")
    print("=" * 78)
    table = run_attack_evaluation()
    print(table.round(4).to_string())
    print("\nHigher is better. The 'none' column is a detector with no cleaning at all.")
    print(f"chart and CSV written to {OUTPUTS_DIR}")
