# FUNCTIONALITY F4 - Graph + BBN ensemble fusion
"""
F4: combine the two detectors - the feature model and the Bayesian network.

WHAT: `train_ensemble()` returns the trained stack and a comparison table of four
      approaches: BBN alone, feature model alone, a plain average, and stacking.
WHY:  the two detectors are good at different things. The feature model reads 34
      numbers and finds patterns we never thought of; the Bayesian network reads 5
      pieces of evidence and can explain itself. Combining them should beat both,
      and this file measures whether it actually does.
HOW:  run `python -m src.f4_ensemble`.

What stacking is, in one paragraph
----------------------------------
A "stack" is a small model that learns how much to trust each of the bigger models.
It takes two inputs - the feature model's probability and the network's probability -
and learns a weighting between them. We use logistic regression for this, because with
two inputs it is basically a learned weighted average, and we can read its two
coefficients out loud: they say which detector the stack trusts more.

Why out-of-fold predictions, and why this is the easiest thing to get wrong
---------------------------------------------------------------------------
The stack must be trained on predictions made for data the base model had NOT seen.
If we trained the feature model on all the training rows and then asked it about those
same rows, it would look nearly perfect on them (it memorised them), the stack would
learn "always believe the feature model", and the whole thing would fall apart on real
data. So we split the training rows into 5 folds and, for each fold, predict it with a
model trained on the other four. That is what `cross_val_predict` does for us.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score)

from src.config import (COMBINED_CSV, FRAUD_THRESHOLD, N_FOLDS, OUTPUTS_DIR,
                        PROCESSED_DIR, RANDOM_SEED)
from src.f1_bbn import fit_network, score_many
from src.f3_fused_features import build_fused_table


def make_feature_model():
    """
    The "feature model": gradient boosting over all 34 fused features.

    WHAT: returns an untrained scikit-learn classifier.
    WHY:  gradient boosting handles mixed scales and non-linear thresholds without any
          preparation, which suits a table mixing ratios, counts and log-ages.
          Defined in one place so every comparison uses an identical model.
    """
    return GradientBoostingClassifier(random_state=RANDOM_SEED, n_estimators=120)


def out_of_fold_probabilities(features, labels):
    """
    Predict every training row using models that never saw that row.

    WHAT: returns an array of probabilities, one per training row.
    WHY:  THIS IS THE STEP THAT KEEPS THE STACK HONEST. Asking a model about rows it
          was trained on gives memorised answers, and the stack would learn to trust
          it blindly. Stratified folds keep the scam/benign ratio steady in each fold.
    """
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    return cross_val_predict(make_feature_model(), features, labels, cv=folds,
                             method="predict_proba")[:, 1]


def train_stack(feature_probability, bbn_probability, labels):
    """
    Train the little model that decides how much to trust each detector.

    WHAT: returns a fitted LogisticRegression taking two inputs.
    WHY:  with two inputs its coefficients are directly readable - they are the answer
          to "which detector does the stack believe?".
    """
    inputs = np.column_stack([feature_probability, bbn_probability])
    return LogisticRegression(max_iter=1000).fit(inputs, labels)


def metrics(labels, probability):
    """
    The five numbers we report for every approach.

    WHAT: precision, recall, f1, roc_auc, false_positive_rate.
    WHY:  one function, so all four approaches are judged identically.
          FPR is reported separately because flagging a real person's message is the
          expensive mistake - a bank that blocks genuine messages loses customers.
    """
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


def _load_and_prepare(split_column):
    """
    Load the data and build the feature table for one split.

    WHAT: returns (messages, table, labels, is_train).
    WHY:  split out of train_ensemble purely to keep each function short enough to
          read in one go.
    NOTE: same rule as F3 - for the time split the graph may only use training-period
          traffic, or a sender's burstiness would be measured using the future.
    """
    messages = pd.read_csv(COMBINED_CSV)
    senders = pd.read_csv(PROCESSED_DIR / "senders.csv")
    labels = messages["label"].to_numpy()
    is_train = (messages[split_column] == "train").to_numpy()
    graph_source = messages[is_train] if split_column == "split_time" else None
    return messages, build_fused_table(messages, senders, graph_source), labels, is_train


def _train_all_detectors(table, labels, is_train):
    """
    Train the BBN, the feature model and the stack that sits on top of them.

    WHAT: returns (parts, probabilities_on_test).
    WHY:  this is the actual fusion. Kept separate from the metric reporting so the
          two concerns can be read, and tested, apart.
    """
    network, engine, bin_edges = fit_network(table[is_train], labels[is_train])
    bbn_train = score_many(engine, table[is_train], bin_edges)
    bbn_test = score_many(engine, table[~is_train], bin_edges)
    features = table.to_numpy()
    feature_model = make_feature_model().fit(features[is_train], labels[is_train])
    feature_test = feature_model.predict_proba(features[~is_train])[:, 1]
    # out-of-fold, so the stack never sees a memorised answer
    stack = train_stack(out_of_fold_probabilities(features[is_train], labels[is_train]),
                        bbn_train, labels[is_train])
    stack_test = stack.predict_proba(np.column_stack([feature_test, bbn_test]))[:, 1]
    parts = {"network": network, "engine": engine, "bin_edges": bin_edges,
             "feature_model": feature_model, "stack": stack,
             "weights": dict(zip(["feature_model", "bbn"], stack.coef_[0]))}
    return parts, {"bbn": bbn_test, "feature": feature_test, "stack": stack_test}


# FUNCTIONALITY F4 - main entry function
def train_ensemble(split_column="split_leaksafe"):
    """
    Train everything and compare the four approaches on held-out data.

    WHAT: returns (comparison_table, parts) where parts holds the fitted models,
          the bin edges and the stack's learned weights.
    WHY:  this is the main results table of the project.
    EXAMPLE: python -m src.f4_ensemble
    """
    messages, table, labels, is_train = _load_and_prepare(split_column)
    parts, predicted = _train_all_detectors(table, labels, is_train)
    truth = labels[~is_train]
    comparison = pd.DataFrame({
        "BBN alone (F1)": metrics(truth, predicted["bbn"]),
        "feature model alone (F3 features)": metrics(truth, predicted["feature"]),
        "plain average of the two": metrics(
            truth, (predicted["feature"] + predicted["bbn"]) / 2),
        "stacked ensemble (F4)": metrics(truth, predicted["stack"]),
    }).T
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUTS_DIR / f"f4_ensemble_{split_column}.csv")
    return comparison, parts


def predict_one(parts, features_row):
    """
    Score a single message with the full ensemble, for the demo app.

    WHAT: returns {"feature_model", "bbn", "ensemble"} probabilities.
    WHY:  the Streamlit app needs one message scored by the finished system, and
          showing all three numbers makes the fusion visible rather than magical.
    """
    from src.f1_bbn import evidence_from_features, score_message
    frame = pd.DataFrame([features_row])
    feature_probability = parts["feature_model"].predict_proba(frame.to_numpy())[:, 1][0]
    evidence = evidence_from_features(features_row, parts["bin_edges"])
    bbn = score_message(parts["engine"], evidence)
    combined = parts["stack"].predict_proba([[feature_probability,
                                              bbn["probability"]]])[:, 1][0]
    return {"feature_model": float(feature_probability), "bbn": bbn["probability"],
            "ensemble": float(combined), "bbn_detail": bbn, "evidence": evidence}


# Demo. Run: python -m src.f4_ensemble
if __name__ == "__main__":
    pd.set_option("display.width", 170)
    for split in ["split_leaksafe", "split_time"]:
        comparison, parts = train_ensemble(split)
        print("=" * 100)
        print(f"MAIN RESULT - four approaches on the {split} split")
        print("=" * 100)
        print(comparison.round(4).to_string())
        weights = parts["weights"]
        print(f"\nwhat the stack learned to trust: "
              f"feature model {weights['feature_model']:+.2f}, BBN {weights['bbn']:+.2f}")
        print("(a larger positive weight means that detector moves the final answer more)")
    print(f"\ntables written to {OUTPUTS_DIR}")
