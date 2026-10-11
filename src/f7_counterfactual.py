# FUNCTIONALITY F7 - Counterfactual explanation generation
"""
F7: answer "what would have had to be different for this verdict to change?"

WHAT: `find_counterfactuals(engine, evidence)` -> the SMALLEST sets of changes that
      flip the verdict, each written as a sentence:
      "If the account had looked trustworthy (HIGH), this message would have been
       marked SAFE (0.93 -> 0.31)."
WHY:  a probability and a reason chain tell you WHY the model decided something. A
      counterfactual tells you what would have CHANGED it, which is the question
      people actually ask when they disagree with a decision - and the question an
      appeals process has to answer.
HOW:  run `python -m src.f7_counterfactual`.

Based on Wachter, Mittelstadt and Russell, "Counterfactual Explanations without
Opening the Black Box" (arXiv 1711.00399). Their idea is the "smallest change to the
world" that changes the outcome. We can do something they could not do easily: because
our model is a Bayesian network over five variables with three values each, there are
only 3^5 = 243 possible worlds. We can simply try them ALL and report the genuinely
smallest changes, rather than searching approximately and hoping.

Why "smallest" matters
----------------------
"If everything about this message had been different, it would have been safe" is
true and useless. A counterfactual is only an explanation if it is minimal, so we try
all one-variable changes first, and only look at pairs if no single change is enough.
"""

from itertools import combinations, product

from src.config import BIN_LABELS, FRAUD_THRESHOLD, MAX_COUNTERFACTUAL_CHANGES
from src.f1_bbn import NODE_WORDS, _probability_of_fraud

# Readable wording for each bin, used in the output sentences.
VALUE_WORDS = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}


def verdict_of(probability):
    """
    Turn a probability into the verdict, using the one shared threshold.

    WHAT: 0.93 -> "FRAUD", 0.31 -> "SAFE"
    WHY:  F1, F7 and F8 must all agree on what counts as fraud, so the threshold lives
          in config.py and is read, never retyped.
    """
    return "FRAUD" if probability >= FRAUD_THRESHOLD else "SAFE"


def alternatives(value):
    """
    The other values a variable could have taken.

    WHAT: "low" -> ["medium", "high"]
    WHY:  a counterfactual has to CHANGE something, so the current value is excluded.
    """
    return [label for label in BIN_LABELS if label != value]


def changes_of_size(evidence, size):
    """
    Every way to change exactly `size` of the variables.

    WHAT: yields dicts like {"SenderTrust": "high", "LinkRisk": "low"}.
    WHY:  generating candidates separately from testing them keeps both parts simple.
          With 5 variables and 3 values this is a few hundred possibilities at most,
          so brute force is not just acceptable - it is exact.
    """
    for nodes in combinations(sorted(evidence), size):
        options = [alternatives(evidence[node]) for node in nodes]
        for values in product(*options):
            yield dict(zip(nodes, values))


def describe(change, before, after):
    """
    Write the counterfactual as one English sentence.

    WHAT: "If how trustworthy the sender account looks had been HIGH, this message
           would have been marked SAFE (0.93 -> 0.31)."
    WHY:  this sentence is the deliverable. Everything else in the file exists to
          produce it.
    """
    parts = [f"{NODE_WORDS[node]} had been {VALUE_WORDS[value]}"
             for node, value in sorted(change.items())]
    joined = " and ".join(parts)
    return (f"If {joined}, this message would have been marked "
            f"{verdict_of(after)} ({before:.2f} -> {after:.2f}).")


# FUNCTIONALITY F7 - main entry function
def find_counterfactuals(engine, evidence, max_changes=MAX_COUNTERFACTUAL_CHANGES):
    """
    Find the smallest changes that would flip the verdict.

    WHAT: returns a list of dicts, each with change, probability, verdict, sentence.
          The list is empty when no change of up to `max_changes` variables flips it.
    WHY:  this is the explanation people ask for when they disagree with a decision.
    HOW:  try all single changes; if any of them flips the verdict, STOP and report
          only those. Only if none works do we try pairs, then triples. That ordering
          is what makes the answer minimal.
    EXAMPLE: find_counterfactuals(engine, {"SenderTrust": "low"})[0]["sentence"]
    """
    before = _probability_of_fraud(engine, evidence)
    original = verdict_of(before)
    for size in range(1, max_changes + 1):
        found = []
        for change in changes_of_size(evidence, size):
            candidate = dict(evidence)
            candidate.update(change)
            after = _probability_of_fraud(engine, candidate)
            if verdict_of(after) != original:
                found.append({"change": change, "probability": after,
                              "verdict": verdict_of(after), "n_changes": size,
                              "sentence": describe(change, before, after)})
        if found:                        # smallest size that works wins; stop here
            return sorted(found, key=lambda item: abs(item["probability"] - before),
                          reverse=True)
    return []


def text_counterfactual(engine, text, sender_row, bin_edges, lexicon_words):
    """
    The text-level version: which single word, removed, changes the verdict?

    WHAT: returns (word, new_probability, sentence) or None if no single word does it.
    WHY:  evidence-level counterfactuals talk about "urgency" in the abstract. This
          points at the actual word in the actual message, which is far more concrete
          for a user: "remove 'turant' and this message would pass".
    HOW:  drop one scam word at a time, recompute the features, re-score.
    """
    from src.f1_bbn import evidence_from_features
    from src.preprocess import normalize, tokenize
    present = [word for word in tokenize(normalize(text)) if word in lexicon_words]
    before = _probability_of_fraud(
        engine, evidence_from_features(sender_row(text), bin_edges))
    for word in dict.fromkeys(present):              # de-duplicate, keep order
        reduced = " ".join(token for token in text.split()
                           if word not in normalize(token))
        after = _probability_of_fraud(
            engine, evidence_from_features(sender_row(reduced), bin_edges))
        if verdict_of(after) != verdict_of(before):
            return (word, after, f"Removing the word '{word}' would have made this "
                                 f"{verdict_of(after)} ({before:.2f} -> {after:.2f}).")
    return None


# Demo. Run: python -m src.f7_counterfactual
if __name__ == "__main__":
    import pandas as pd

    from src.config import COMBINED_CSV, PROCESSED_DIR
    from src.f1_bbn import evidence_from_features, fit_network, score_many
    from src.f3_fused_features import build_fused_table

    messages = pd.read_csv(COMBINED_CSV)
    senders = pd.read_csv(PROCESSED_DIR / "senders.csv")
    table = build_fused_table(messages, senders)
    is_train = (messages["split_leaksafe"] == "train").to_numpy()
    network, engine, bin_edges = fit_network(table[is_train], messages["label"][is_train])

    test = messages[~is_train].assign(score=score_many(engine, table[~is_train], bin_edges))
    flagged = test[test.score >= FRAUD_THRESHOLD].sort_values("score")
    for row in [flagged.index[-1], flagged.index[0]]:       # most and least certain
        evidence = evidence_from_features(table.loc[row], bin_edges)
        results = find_counterfactuals(engine, evidence)
        print("=" * 78)
        print(f"MESSAGE: {messages.loc[row, 'text'][:68]}")
        print(f"EVIDENCE: {evidence}")
        print(f"SCORE: {_probability_of_fraud(engine, evidence):.2f} -> FRAUD\n")
        if not results:
            print("  No change of up to 3 variables would have made this safe.")
            continue
        print(f"SMALLEST CHANGES THAT WOULD FLIP IT "
              f"({results[0]['n_changes']} variable(s), {len(results)} option(s)):")
        for item in results[:4]:
            print(f"  - {item['sentence']}")
