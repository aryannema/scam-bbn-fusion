# FUNCTIONALITY F3 - Fused feature selection (graph + NLP)
"""
F3: glues the text features onto the graph features and asks which ones matter.

WHAT: `build_fused_table(messages)` -> one row per message with 34 columns:
      21 from the text side (F2 + F6) and 13 from the sender/graph side.
      `run_comparison()` produces the project's first key result table.
WHY:  the claim of this project is that TEXT and BEHAVIOUR together beat either alone.
      That claim has to be measured, not asserted, so this file trains the same model
      three times - text only, graph only, both - and prints the three scores.
HOW:  run `python -m src.f3_fused_features`.

Why feature selection at all
----------------------------
34 features is more than we need, and some are nearly copies of each other
(`burstiness` and `messages_per_hour` are the same idea twice). Mutual information
scores each feature against the label on its own, so we can report a short list of
what actually carries signal. We print a model-based ranking next to it, because the
two disagree in an interesting way: mutual information judges each feature alone,
while the model judges it in the presence of all the others.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.config import COMBINED_CSV, OUTPUTS_DIR, PROCESSED_DIR, RANDOM_SEED, TOP_K_FEATURES
from src.f2_hinglish_features import extract_text_features
from src.f6_phonetic import extract_phonetic_features
from src.graph_builder import build_sender_features

# The sender's own history. These are not graph shape, but they are what the NETWORK
# knows about a sender rather than what the message says, so they belong on the
# behaviour side of the comparison.
SENDER_HISTORY = ["past_reports", "account_age_days"]


def build_fused_table(messages, senders=None, graph_source=None):
    """
    Build the full feature table: text features and behaviour features side by side.

    WHAT: returns a DataFrame with one row per message, in the same order as `messages`.
    WHY:  F3 and F4 both train on this table, so it is built in exactly one place.
    NOTE: graph features are per SENDER, so every message from the same sender carries
          the same behaviour numbers. That is correct - it is what a detector would
          know about the account at the moment the message arrives.
    ARGS: graph_source - the messages the GRAPH is built from, when that has to differ
          from the messages being scored. This matters for the time-based split: if we
          measured a sender's burstiness using messages they sent AFTER the cutoff, we
          would be feeding the future into the past. Senders never seen in the graph
          source get zeros, which is the honest cold-start case.
    """
    text = extract_text_features(messages["text"])
    phonetic = extract_phonetic_features(messages["text"])
    graph = build_sender_features(messages if graph_source is None else graph_source)
    if senders is not None:                     # attach the sender's complaint history
        graph = graph.join(senders.set_index("sender_id")[SENDER_HISTORY], how="left")
    per_message = graph.reindex(messages["sender_id"]).reset_index(drop=True)
    table = pd.concat([text, phonetic, per_message], axis=1)
    return table.fillna(0.0)


def feature_groups(table):
    """
    Say which columns are text and which are behaviour.

    WHAT: returns (text_columns, graph_columns).
    WHY:  the three-way comparison needs the split to be explicit and auditable,
          not guessed from column names at the point of use.
    """
    text_columns = list(extract_text_features(["x"]).columns)
    text_columns += list(extract_phonetic_features(["x"]).columns)
    graph_columns = [column for column in table.columns if column not in text_columns]
    return text_columns, graph_columns


def select_features(features, labels, k=TOP_K_FEATURES):
    """
    Rank features by mutual information with the label and keep the best k.

    WHAT: returns (selected_names, scores_series).
    WHY:  mutual information asks "how much does knowing this feature tell me about
          the label?". It needs no model and it catches non-linear relationships,
          unlike a plain correlation.
    NOTE: fitted on TRAINING rows only. Fitting it on everything would let the test
          set influence which features we chose, which is a subtle form of leakage.
    """
    chooser = SelectKBest(mutual_info_classif, k=min(k, features.shape[1]))
    chooser.fit(features, labels)
    scores = pd.Series(chooser.scores_, index=features.columns).sort_values(ascending=False)
    selected = list(features.columns[chooser.get_support()])
    return selected, scores


def model_ranking(features, labels):
    """
    Rank features by how much a trained model actually leaned on them.

    WHAT: returns a Series of importances, highest first.
    WHY:  reported next to mutual information because the two answer different
          questions. A feature can score well alone and still be ignored by the model
          because another feature already carries the same information.
    """
    model = GradientBoostingClassifier(random_state=RANDOM_SEED, n_estimators=120)
    model.fit(features, labels)
    return pd.Series(model.feature_importances_, index=features.columns).sort_values(
        ascending=False)


def train_and_score(features, labels, is_train):
    """
    Train the feature model on the training rows and score it on the held-out rows.

    WHAT: returns a dict of precision, recall, f1, roc_auc and false_positive_rate.
    WHY:  one function used for all three feature sets, so the comparison is fair by
          construction - same model, same split, same metrics.
    FPR matters here: flagging a real person's message is the costly mistake.
    """
    model = GradientBoostingClassifier(random_state=RANDOM_SEED, n_estimators=120)
    model.fit(features[is_train], labels[is_train])
    predicted = model.predict(features[~is_train])
    probability = model.predict_proba(features[~is_train])[:, 1]
    truth = labels[~is_train]
    benign = truth == 0
    false_positives = (predicted[benign] == 1).sum()
    return {
        "precision": precision_score(truth, predicted, zero_division=0),
        "recall": recall_score(truth, predicted, zero_division=0),
        "f1": f1_score(truth, predicted, zero_division=0),
        "roc_auc": roc_auc_score(truth, probability),
        "false_positive_rate": false_positives / max(benign.sum(), 1),
    }


def compare_feature_sets(table, labels, is_train):
    """
    THE KEY RESULT: text only vs graph only vs both.

    WHAT: returns a DataFrame with one row per feature set.
    WHY:  this table is the evidence for the central claim of the project. If "fused"
          does not beat both halves, the fusion idea does not work and we must say so.
    """
    text_columns, graph_columns = feature_groups(table)
    sets = {
        "text only (F2+F6)": text_columns,
        "graph only (F3)": graph_columns,
        "fused (text+graph)": text_columns + graph_columns,
    }
    rows = {}
    for name, columns in sets.items():
        rows[name] = train_and_score(table[columns].to_numpy(), labels, is_train)
        rows[name]["n_features"] = len(columns)
    return pd.DataFrame(rows).T


# FUNCTIONALITY F3 - main entry function
def run_comparison(split_column="split_leaksafe"):
    """
    Build everything, run the comparison, print and save the tables.

    WHAT: returns (comparison_table, mutual_info_scores, model_importances).
    WHY:  one command reproduces the first results table in the paper.
    EXAMPLE: python -m src.f3_fused_features
    """
    messages = pd.read_csv(COMBINED_CSV)
    senders = pd.read_csv(PROCESSED_DIR / "senders.csv")
    labels = messages["label"].to_numpy()
    is_train = (messages[split_column] == "train").to_numpy()
    # For the time split, the graph may only use messages from the training PERIOD.
    # For the leak-safe split no sender appears on both sides, so a test sender has no
    # training traffic at all; using their own messages is both necessary and realistic
    # (a network sees an account's traffic before deciding anything about it).
    graph_source = messages[is_train] if split_column == "split_time" else None
    table = build_fused_table(messages, senders, graph_source)
    comparison = compare_feature_sets(table, labels, is_train)
    selected, scores = select_features(table[is_train], labels[is_train])
    importances = model_ranking(table[is_train], labels[is_train])
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUTS_DIR / f"f3_comparison_{split_column}.csv")
    scores.to_csv(OUTPUTS_DIR / "f3_mutual_information.csv")
    return comparison, scores, importances, selected


# Demo. Run: python -m src.f3_fused_features
if __name__ == "__main__":
    pd.set_option("display.width", 160)
    for split in ["split_leaksafe", "split_time"]:
        comparison, scores, importances, selected = run_comparison(split)
        print(f"\n{'=' * 78}\nKEY RESULT - feature sets compared on the {split} split")
        print("=" * 78)
        print(comparison.round(4).to_string())
        if split == "split_leaksafe":
            print(f"\nTOP 12 FEATURES BY MUTUAL INFORMATION (fitted on training rows only)")
            print("-" * 78)
            print(scores.head(12).round(4).to_string())
            print(f"\nTOP 12 FEATURES BY MODEL IMPORTANCE")
            print("-" * 78)
            print(importances.head(12).round(4).to_string())
            print(f"\nSELECTED FEATURE LIST (k={len(selected)}):")
            print("  " + ", ".join(selected))
    print(f"\ntables written to {OUTPUTS_DIR}")
