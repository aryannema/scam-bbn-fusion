# FUNCTIONALITY F1 - BBN reasoning layer + human-readable explanation chain
"""
F1: a Bayesian Belief Network that scores a message AND explains itself in English.

WHAT: `score_message(evidence)` -> {"probability": 0.93, "verdict": "FRAUD",
      "explanation": ["Prior 0.25", "sender trust LOW: 0.34", ...]}
WHY:  a gradient-boosting model can say "0.93" but not WHY. A Bayesian network holds
      the reasoning explicitly, as a small diagram of causes, so the explanation is
      the actual computation rather than a story told about it afterwards.
HOW:  run `python -m src.f1_bbn`.

The network (drawn by us, from how scams actually work)
-------------------------------------------------------
    UrgencyCue  CredentialAsk  LinkRisk
            \\        |        /
             v       v       v
              PatternAnomaly          SenderTrust     NetworkCentrality
                        \\                  |             /
                         v                 v            v
                                     Fraud

Read it as causes pointing at effects: a scam message is written with urgency, asks
for a secret and carries a risky link, which together make the WORDING anomalous.
The wording, who sent it, and how that account sits in the network then decide Fraud.

Why discretize everything into low/medium/high
----------------------------------------------
A discrete Bayesian network needs each variable to take one of a few values, so every
number is cut into three bins at the 33rd and 67th percentile of the TRAINING data.
Three bins is a deliberate compromise: two would throw away the middle ground, and
five would split the data so thinly that the probability tables become unreliable.

How the explanation chain works
-------------------------------
We start from the prior - what we believe before reading anything. Then we add ONE
piece of evidence at a time, strongest first, and record how the probability moves.
That list of moves IS the explanation, and it is exactly what the model computed.
"""

import warnings

# pgmpy 1.1.2 prints a FutureWarning saying BayesianEstimator will be replaced by
# pgmpy.parameter_estimator.DiscreteBayesianEstimator in v1.3.0. We checked the
# replacement in this exact version and it does not work yet: its .fit() returns the
# estimator, leaves the network with zero CPDs, and .get_fitted_params() then reports
# that nothing has been fitted. So we keep the working class for our pinned version and
# silence only this one message, so a live demo is not covered in warnings.
# Matched on the MESSAGE, not the module: the warning is raised at our call site, so
# filtering by module="pgmpy" would never match it.
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*BayesianEstimator.*")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*StructureScore.*")

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from src.config import BIN_LABELS, FRAUD_THRESHOLD, N_BINS

# The expert structure: (cause, effect) pairs. This is the diagram above, as code.
EXPERT_EDGES = [
    ("UrgencyCue", "PatternAnomaly"),
    ("CredentialAsk", "PatternAnomaly"),
    ("LinkRisk", "PatternAnomaly"),
    ("PatternAnomaly", "Fraud"),
    ("SenderTrust", "Fraud"),
    ("NetworkCentrality", "Fraud"),
]

# Which measured feature feeds which node of the network.
# Keeping this in one dict means the mapping is auditable in one glance.
NODE_SOURCES = {
    "UrgencyCue": "urgency_hits",
    "CredentialAsk": "credential_hits",
    "LinkRisk": "link_risk_score",
    "SenderTrust": "sender_trust_score",
    "NetworkCentrality": "centrality_score",
}

# Plain English for each node, used when writing the explanation sentences.
NODE_WORDS = {
    "UrgencyCue": "time pressure in the wording",
    "CredentialAsk": "a request for a secret (OTP, PIN, KYC)",
    "LinkRisk": "the riskiness of the link",
    "PatternAnomaly": "how unusual the wording is overall",
    "SenderTrust": "how trustworthy the sender account looks",
    "NetworkCentrality": "how the account sits in the message network",
}


def link_risk_score(features):
    """
    Combine the three link columns into one number for the LinkRisk node.

    WHAT: 0 (no link) .. 3 (a shortened link on a throwaway domain).
    WHY:  the network node needs ONE variable, and "is there a link, and does it look
          dodgy" is one question a human asks in one glance.
    """
    return (features["url_count"].clip(0, 1)
            + features["has_shortener"]
            + features["has_suspicious_tld"])


def sender_trust_score(features):
    """
    Turn the sender's history into a trust number. HIGHER MEANS MORE TRUSTWORTHY.

    WHAT: old account with no complaints -> high; new account with complaints -> low.
    WHY:  SenderTrust is the one node where "high" should be reassuring, so the
          direction is set here, once, rather than being remembered everywhere.
    """
    age = np.log1p(features["account_age_days"])
    complaints = np.log1p(features["past_reports"])
    return age - 2.0 * complaints          # complaints weigh more than age


def centrality_score(features):
    """
    Combine the graph columns into one "campaign shape" number.

    WHAT: high when an account sprays many strangers in a burst and shares templates.
    WHY:  NetworkCentrality is meant to capture "this account behaves like a campaign",
          which is several graph features saying the same thing.
    """
    return (features["unique_recipient_ratio"]
            + features["forwarding_ratio"]
            + np.log1p(features["burstiness"])
            - features["clustering"])      # friends-of-friends pulls it back down


def add_node_inputs(features):
    """
    Add the three combined columns the network needs, next to the raw ones.

    WHAT: adds link_risk_score, sender_trust_score, centrality_score.
    WHY:  separated from discretization so each step can be tested on its own.
    """
    table = features.copy()
    table["link_risk_score"] = link_risk_score(features)
    table["sender_trust_score"] = sender_trust_score(features)
    table["centrality_score"] = centrality_score(features)
    return table


def bin_edges(values):
    """
    Choose the two cut points that turn a column into low / medium / high.

    WHAT: returns a list of up to N_BINS - 1 cut points.
    WHY:  plain quantiles are not enough. Most messages contain no link at all, so for
          `link_risk_score` BOTH the 33rd and the 67th percentile are 0. They collapse
          into a single cut point, the node ends up with only two states (low and
          medium), and asking the network about LinkRisk = "high" then raises
          `KeyError: state high is an unknown for variable LinkRisk`.
    HOW:  try quantiles first, because they balance the bins. If a skewed column
          collapses them, fall back to the smallest distinct VALUES as cut points,
          which for a 0/1/2/3 count gives low=0, medium=1, high=2+.
    """
    quantiles = np.linspace(0, 1, N_BINS + 1)[1:-1]            # 0.33 and 0.67
    edges = list(np.unique(np.quantile(values, quantiles)))
    if len(edges) < N_BINS - 1:
        distinct = np.unique(np.asarray(values))
        if len(distinct) >= N_BINS:
            edges = list(distinct[:N_BINS - 1])
    return edges


def discretize(values, edges=None):
    """
    Cut a column of numbers into low / medium / high.

    WHAT: returns (labels, edges). `edges` are the two cut points used.
    WHY:  a discrete Bayesian network cannot read a raw number. Quantile cuts put
          roughly a third of the training data in each bin, which keeps every
          probability table populated.
    IMPORTANT: the edges must be learned from TRAINING data and then REUSED for test
          data. Re-cutting the test set by its own quantiles would leak information
          about the test distribution into the model.
    EXAMPLE: discretize(pd.Series([1, 5, 9]))[0] -> ["low", "medium", "high"]
    """
    if edges is None:
        edges = bin_edges(values)
    labels = np.full(len(values), BIN_LABELS[0], dtype=object)
    for position, edge in enumerate(edges):
        labels[np.asarray(values) > edge] = BIN_LABELS[min(position + 1, N_BINS - 1)]
    return pd.Series(labels, index=getattr(values, "index", None)), edges


def discretize_table(features, labels=None, edges=None):
    """
    Discretize every node input, and attach the Fraud column.

    WHAT: returns (frame_of_labels, edges_per_node).
    WHY:  pgmpy learns its probability tables from a table of category names, so this
          is the exact shape it wants.
    """
    table = add_node_inputs(features)
    out, learned = {}, {}
    for node, column in NODE_SOURCES.items():
        learned_edges = None if edges is None else edges[node]
        out[node], learned[node] = discretize(table[column], learned_edges)
    frame = pd.DataFrame(out)
    if labels is not None:
        frame["Fraud"] = np.where(np.asarray(labels) == 1, "yes", "no")
    return frame, learned


def pattern_anomaly_from_parents(frame):
    """
    Fill in PatternAnomaly, the one node we never measure directly.

    WHAT: counts how many of its three parents are "high", then bins that 0..3 count.
    WHY:  PatternAnomaly is a hidden middle layer: "is the wording anomalous overall?"
          We have no column for it, so we derive it from its three causes. This keeps
          the expert diagram intact instead of quietly deleting the node.
    """
    parents = ["UrgencyCue", "CredentialAsk", "LinkRisk"]
    high_count = sum((frame[parent] == "high").astype(int) for parent in parents)
    mapping = {0: "low", 1: "medium", 2: "high", 3: "high"}
    return high_count.map(mapping)


def build_network():
    """
    Create the empty network with our hand-drawn edges.

    WHAT: returns a pgmpy DiscreteBayesianNetwork with no probabilities yet.
    WHY:  separating structure from fitting makes F5 (structure learning) a fair
          comparison: same code, different edges.
    NOTE: pgmpy 1.x calls this class DiscreteBayesianNetwork. Older tutorials say
          BayesianNetwork - see docs/API_NOTES.md.
    """
    return DiscreteBayesianNetwork(EXPERT_EDGES)


def fit_network(train_features, train_labels, edges=None):
    """
    Learn the probability tables from training data.

    WHAT: returns (fitted_network, inference_engine, bin_edges).
    WHY:  the STRUCTURE is expert knowledge, but the NUMBERS come from the data. We use
          a Bayesian estimator with a BDeu prior rather than plain counting, because
          some parent combinations are rare and plain counting would give them a
          probability of exactly 0 or 1 on the strength of two examples.
    """
    frame, learned_edges = discretize_table(train_features, train_labels, edges)
    frame["PatternAnomaly"] = pattern_anomaly_from_parents(frame)
    network = build_network()
    # pgmpy 1.x: network.fit() does NOT pass options through to the estimator, so we
    # build the estimator ourselves to choose the BDeu prior. See docs/API_NOTES.md.
    estimator = BayesianEstimator(network, frame)
    network.add_cpds(*estimator.get_parameters(prior_type="BDeu",
                                               equivalent_sample_size=10))
    return network, VariableElimination(network), learned_edges


def _probability_of_fraud(engine, evidence):
    """
    Ask the network for P(Fraud = yes) given whatever evidence we have.

    WHAT: a float between 0 and 1.
    WHY:  one place for the pgmpy call, so the rest of the file reads like English.
    """
    result = engine.query(["Fraud"], evidence=evidence or None, show_progress=False)
    return float(result.get_value(Fraud="yes"))


def rank_evidence(engine, evidence):
    """
    Order the evidence by how much each piece moves the probability on its own.

    WHAT: returns [(node, value, probability_with_just_this)] strongest first.
    WHY:  the explanation should lead with the strongest reason, the way a person
          would. "Strongest" is measured, not assumed: we try each piece alone.
    """
    prior = _probability_of_fraud(engine, {})
    scored = []
    for node, value in evidence.items():
        alone = _probability_of_fraud(engine, {node: value})
        scored.append((abs(alone - prior), node, value, alone))
    scored.sort(reverse=True)
    return [(node, value, alone) for _, node, value, alone in scored]


# FUNCTIONALITY F1 - main entry function
def score_message(engine, evidence):
    """
    Score one message and build its explanation chain.

    WHAT: returns {"probability", "verdict", "prior", "explanation", "steps"}.
    WHY:  this is the function the Streamlit app and F7 both call.
    HOW:  start from the prior, then add evidence one piece at a time, strongest
          first, recording the probability after each addition. The resulting list is
          the explanation, and it is literally the computation that produced the score.
    EXAMPLE: score_message(engine, {"SenderTrust": "low"})["probability"]
    """
    prior = _probability_of_fraud(engine, {})
    chain = [f"Prior (before reading anything): {prior:.2f}"]
    steps = [("prior", "", prior)]
    running = {}
    for node, value, _ in rank_evidence(engine, evidence):
        running[node] = value
        probability = _probability_of_fraud(engine, running)
        chain.append(f"{NODE_WORDS[node]} is {value.upper()}: {probability:.2f}")
        steps.append((node, value, probability))
    final = _probability_of_fraud(engine, evidence) if evidence else prior
    verdict = "FRAUD" if final >= FRAUD_THRESHOLD else "SAFE"
    return {"probability": final, "verdict": verdict, "prior": prior,
            "explanation": chain, "steps": steps,
            "sentence": _verdict_sentence(verdict, final, steps)}


def _verdict_sentence(verdict, probability, steps):
    """
    Write the one-sentence summary a human actually reads.

    WHAT: "Marked FRAUD (0.930): the strongest reason was a request for a secret."
    WHY:  the chain shows the arithmetic; this says what it means.
    NOTE: three decimals, not two, on purpose. A score of 0.499 is SAFE under a 0.50
          threshold, but printed as "SAFE with probability 0.50" it reads like a bug.
    """
    if len(steps) > 1:
        reason = NODE_WORDS[steps[1][0]]
        return (f"Marked {verdict} with probability {probability:.3f}; "
                f"the strongest single reason was {reason}.")
    return f"Marked {verdict} with probability {probability:.3f} on the prior alone."


def evidence_from_features(features_row, edges):
    """
    Turn one row of measured features into network evidence.

    WHAT: {"UrgencyCue": "high", "CredentialAsk": "medium", ...}
    WHY:  bridges the gap between F2/F3's numbers and the network's categories, using
          the bin edges learned from the TRAINING data.
    """
    frame, _ = discretize_table(pd.DataFrame([features_row]), None, edges)
    return {node: frame.iloc[0][node] for node in NODE_SOURCES}


def score_many(engine, features, edges):
    """
    Score a whole table of messages with the network.

    WHAT: returns a numpy array of fraud probabilities, one per row.
    WHY:  F4 needs the BBN's opinion on every message so it can stack it against the
          feature model, and F8 needs it to re-score attacked messages.
    HOW:  rows with identical evidence get identical answers, so we cache by evidence
          tuple. Exact inference on 7,758 rows one at a time would otherwise be slow
          for no reason - there are only 3^5 = 243 possible evidence combinations.
    """
    frame, _ = discretize_table(features, None, edges)
    cache = {}
    answers = []
    for _, row in frame.iterrows():
        evidence = {node: row[node] for node in NODE_SOURCES}
        key = tuple(sorted(evidence.items()))
        if key not in cache:
            cache[key] = _probability_of_fraud(engine, evidence)
        answers.append(cache[key])
    return np.array(answers)


def evaluate_network(engine, features, labels, edges):
    """
    Measure how the network does on its own, as a baseline for F4.

    WHAT: returns a dict of precision, recall, f1, roc_auc, false_positive_rate.
    WHY:  the ensemble in F4 has to be compared against the BBN alone, or we cannot
          claim the combination helped.
    """
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score)
    probability = score_many(engine, features, edges)
    predicted = (probability >= FRAUD_THRESHOLD).astype(int)
    truth = np.asarray(labels)
    benign = truth == 0
    return {
        "precision": precision_score(truth, predicted, zero_division=0),
        "recall": recall_score(truth, predicted, zero_division=0),
        "f1": f1_score(truth, predicted, zero_division=0),
        "roc_auc": roc_auc_score(truth, probability),
        "false_positive_rate": (predicted[benign] == 1).sum() / max(benign.sum(), 1),
    }


# Demo. Run: python -m src.f1_bbn
if __name__ == "__main__":
    from src.config import COMBINED_CSV, PROCESSED_DIR
    from src.f3_fused_features import build_fused_table

    messages = pd.read_csv(COMBINED_CSV)
    senders = pd.read_csv(PROCESSED_DIR / "senders.csv")
    table = build_fused_table(messages, senders)
    is_train = (messages["split_leaksafe"] == "train").to_numpy()
    print("fitting the expert network on the training split...")
    network, engine, bin_edges = fit_network(table[is_train], messages["label"][is_train])
    print(f"nodes: {sorted(network.nodes())}")
    print(f"edges: {EXPERT_EDGES}\n")

    print("scoring the held-out test split...")
    test_rows = ~is_train
    scores = score_many(engine, table[test_rows], bin_edges)
    result = evaluate_network(engine, table[test_rows], messages["label"][test_rows],
                              bin_edges)
    print("THE NETWORK ON ITS OWN (held-out test split):")
    print("  " + ", ".join(f"{k}={v:.3f}" for k, v in result.items()))
    print()

    # Show the clearest case of each kind: the scam the network is most sure about,
    # and the benign message it is least worried about.
    test = messages[test_rows].assign(score=scores)
    scam_example = test[test.label == 1].sort_values("score").index[-1]
    benign_example = test[test.label == 0].sort_values("score").index[0]
    examples = [scam_example, benign_example]
    for row in examples:
        evidence = evidence_from_features(table.loc[row], bin_edges)
        result = score_message(engine, evidence)
        print("=" * 78)
        print(f"MESSAGE: {messages.loc[row, 'text'][:70]}")
        print(f"TRUE LABEL: {'scam' if messages.loc[row, 'label'] == 1 else 'benign'}")
        print(f"EVIDENCE: {evidence}")
        print("\nEXPLANATION CHAIN")
        print("  " + "\n  -> ".join(result["explanation"]))
        print(f"\n{result['sentence']}")
