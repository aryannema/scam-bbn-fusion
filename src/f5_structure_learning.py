# FUNCTIONALITY F5 - Data-driven BBN structure learning
"""
F5: let the DATA draw the network, then compare it against the one we drew by hand.

WHAT: `run_comparison()` learns a structure with hill-climbing search and reports how
      it differs from the expert structure in F1, on edges, held-out log-likelihood
      and F1 score. Saves a PNG of both graphs.
WHY:  in F1 we claimed a diagram based on how we think scams work. That claim should
      be tested. If the data draws almost the same picture, our expert knowledge is
      supported. If it draws something different, that is a finding worth reporting.
HOW:  run `python -m src.f5_structure_learning`.

How hill-climbing search works, in plain words
----------------------------------------------
Start with no edges. Try every single change you could make - add an edge, remove one,
reverse one - and keep whichever change improves the score most. Repeat until nothing
improves. It is "hill climbing" because it only ever steps uphill, which also means it
can stop on a small hill rather than the mountain: the result is a good structure, not
a guaranteed best one.

What the score means
--------------------
BIC = (how well the structure explains the data) - (a penalty for each extra number
the structure needs). Without the penalty, the best structure would always be the one
with every possible edge, which explains the training data perfectly and generalises
terribly. BDeu is the same idea from a Bayesian starting point, with a prior instead of
an explicit penalty. We report both because they disagree, and the disagreement is
informative.

The blacklist, and why it is legitimate
---------------------------------------
Search only sees correlations, so it may point an arrow from Fraud INTO the sender's
account age - which reads as "being a fraud makes your account younger". That is
backwards in time and meaningless. We forbid edges out of Fraud. This is expert
knowledge about causal DIRECTION, not about which variables matter, so it does not
hand the search its answer.
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
# Same story for HillClimbSearch: pgmpy.causal_discovery.HillClimbSearch exists in
# 1.1.2 but has no .estimate() method, so the documented replacement cannot be called.
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*HillClimbSearch.*")

import matplotlib
matplotlib.use("Agg")                    # draw to a file, never to a screen
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, ExpertKnowledge, HillClimbSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from src.config import (COMBINED_CSV, FRAUD_THRESHOLD, MAX_INDEGREE, OUTPUTS_DIR,
                        PROCESSED_DIR)
from src.f1_bbn import (EXPERT_EDGES, NODE_SOURCES, discretize_table, fit_network,
                        pattern_anomaly_from_parents)

# The scores we compare. pgmpy 1.x takes these as strings: "bic-d" is BIC for discrete
# data, "bdeu" is the Bayesian Dirichlet score. See docs/API_NOTES.md.
SCORES = ["bic-d", "bdeu"]


def build_training_frame(features, labels):
    """
    Build the all-categories table that structure search runs on.

    WHAT: a DataFrame of low/medium/high strings plus a Fraud column of yes/no.
    WHY:  search needs every variable discrete, exactly like the network in F1, so the
          two structures are learned from identical data and the comparison is fair.
    """
    frame, edges = discretize_table(features, labels)
    frame["PatternAnomaly"] = pattern_anomaly_from_parents(frame)
    return frame, edges


def forbidden_edges(nodes):
    """
    List the edges search is not allowed to draw.

    WHAT: every edge pointing OUT of Fraud.
    WHY:  "Fraud causes the account to be new" is backwards. Forbidding the direction
          keeps the learned graph readable as a causal story without telling the search
          which variables are important.
    NOTE: pgmpy 1.x replaced the old `black_list=` argument with an ExpertKnowledge
          object. This function provides the pairs that go into it.
    """
    return [("Fraud", node) for node in nodes if node != "Fraud"]


def learn_structure(frame, score="bic-d"):
    """
    Learn a structure from the data with hill-climbing search.

    WHAT: returns a list of (parent, child) edges.
    WHY:  this is the data's own answer to "what causes what", for comparison with ours.
    NOTE: max_indegree caps how many parents a node may have. Without a cap, a node can
          collect so many parents that its probability table has more cells than we have
          rows of data to fill them.
    """
    search = HillClimbSearch(frame)
    knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges(list(frame.columns)))
    learned = search.estimate(scoring_method=score, max_indegree=MAX_INDEGREE,
                              expert_knowledge=knowledge, show_progress=False)
    return list(learned.edges())


def compare_edges(expert, learned):
    """
    Say exactly how the two structures differ.

    WHAT: returns {"shared", "added_by_data", "removed_by_data", "reversed"}.
    WHY:  "the graphs differ" is not a result. Which edges, and in which direction, is.
    """
    expert_set, learned_set = set(expert), set(learned)
    reversed_edges = {(a, b) for a, b in learned_set if (b, a) in expert_set}
    return {
        "shared": sorted(expert_set & learned_set),
        "added_by_data": sorted(learned_set - expert_set - reversed_edges),
        "removed_by_data": sorted(expert_set - learned_set
                                  - {(b, a) for a, b in reversed_edges}),
        "reversed": sorted(reversed_edges),
    }


def fit_structure(edges, frame):
    """
    Fit probability tables onto any structure.

    WHAT: returns (network, inference_engine).
    WHY:  used for BOTH structures, so the only thing that differs in the comparison is
          the edges themselves.
    NOTE: isolated nodes are added explicitly - search may leave a variable with no
          edges at all, and we still want it in the model.
    """
    network = DiscreteBayesianNetwork(edges)
    network.add_nodes_from([column for column in frame.columns
                            if column not in network.nodes()])
    estimator = BayesianEstimator(network, frame)
    network.add_cpds(*estimator.get_parameters(prior_type="BDeu",
                                               equivalent_sample_size=10))
    return network, VariableElimination(network)


def holdout_log_likelihood(network, frame):
    """
    How surprised is this model by data it has never seen? Higher (less negative) is better.

    WHAT: average log-probability per row of the held-out data.
    WHY:  F1 score only checks the Fraud column. Log-likelihood checks whether the whole
          joint distribution is right, which is what a structure is really claiming.
    HOW:  for each row, multiply P(value | parents) for every node and take the log.
          pgmpy 1.1.2 has no log_likelihood_score helper, so we read the fitted
          probability tables directly.
    """
    totals = np.zeros(len(frame))
    for node in network.nodes():
        cpd = network.get_cpds(node)
        parents = cpd.get_evidence()
        for position, (_, row) in enumerate(frame.iterrows()):
            lookup = {node: row[node]}
            lookup.update({parent: row[parent] for parent in parents})
            totals[position] += np.log(max(cpd.get_value(**lookup), 1e-12))
    return float(totals.mean())


def fraud_scores(engine, frame):
    """
    Ask a fitted network for P(Fraud) on every row of a table.

    WHAT: returns a numpy array of probabilities.
    WHY:  needed to score both structures with the same metric. Results are cached per
          evidence combination, because there are only a few hundred of them.
    """
    evidence_nodes = [node for node in NODE_SOURCES if node in frame.columns]
    cache, answers = {}, []
    for _, row in frame.iterrows():
        evidence = {node: row[node] for node in evidence_nodes}
        key = tuple(sorted(evidence.items()))
        if key not in cache:
            result = engine.query(["Fraud"], evidence=evidence, show_progress=False)
            cache[key] = float(result.get_value(Fraud="yes"))
        answers.append(cache[key])
    return np.array(answers)


def evaluate_structure(engine, frame, labels):
    """
    Score a structure on held-out data: F1 and ROC-AUC.

    WHAT: returns a dict of f1 and roc_auc.
    WHY:  a structure that explains the data well should also detect fraud well. If the
          learned graph wins on log-likelihood but loses on F1, that is worth saying.
    """
    from sklearn.metrics import f1_score, roc_auc_score
    probability = fraud_scores(engine, frame)
    predicted = (probability >= FRAUD_THRESHOLD).astype(int)
    return {"f1": f1_score(labels, predicted, zero_division=0),
            "roc_auc": roc_auc_score(labels, probability)}


def save_graph_png(edges, path, title):
    """
    Draw a structure and save it as a PNG.

    WHAT: writes the file and returns its path.
    WHY:  the two pictures side by side are the clearest way to present this result in
          the report and in the viva.
    """
    graph = nx.DiGraph(edges)
    plt.figure(figsize=(7, 5))
    layout = nx.spring_layout(graph, seed=7)
    nx.draw(graph, layout, with_labels=True, node_color="#cfe3ff", node_size=2600,
            font_size=8, arrowsize=18, edge_color="#444444")
    plt.title(title)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130)
    plt.close()
    return path


def prepare_frames():
    """
    Build the discretized train and test frames that every structure is judged on.

    WHAT: returns (train_frame, test_frame, test_labels).
    WHY:  split out of run_comparison to keep each function short. Both structures see
          exactly the same frames, which is what makes the comparison fair.
    """
    from src.f3_fused_features import build_fused_table
    messages = pd.read_csv(COMBINED_CSV)
    senders = pd.read_csv(PROCESSED_DIR / "senders.csv")
    table = build_fused_table(messages, senders)
    is_train = (messages["split_leaksafe"] == "train").to_numpy()
    train_frame, bin_edges = build_training_frame(table[is_train],
                                                 messages["label"][is_train])
    test_frame, _ = discretize_table(table[~is_train], messages["label"][~is_train],
                                     bin_edges)
    test_frame["PatternAnomaly"] = pattern_anomaly_from_parents(test_frame)
    return train_frame, test_frame, messages["label"][~is_train].to_numpy()


# FUNCTIONALITY F5 - main entry function
def run_comparison():
    """
    Learn structures, compare them against the expert one, save everything.

    WHAT: returns (results_table, differences_per_score, structures).
    WHY:  one command reproduces the expert-vs-learned result in the paper.
    EXAMPLE: python -m src.f5_structure_learning
    """
    train_frame, test_frame, labels = prepare_frames()
    structures = {"expert (F1, hand-drawn)": EXPERT_EDGES}
    for score in SCORES:
        structures[f"learned ({score})"] = learn_structure(train_frame, score)
    rows, differences = {}, {}
    for name, edges in structures.items():
        network, engine = fit_structure(edges, train_frame)
        rows[name] = {"n_edges": len(edges),
                      "holdout_log_likelihood": holdout_log_likelihood(network, test_frame)}
        rows[name].update(evaluate_structure(engine, test_frame, labels))
        differences[name] = compare_edges(EXPERT_EDGES, edges)
        save_graph_png(edges, OUTPUTS_DIR / f"f5_structure_{_safe_name(name)}.png", name)
    results = pd.DataFrame(rows).T
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUTS_DIR / "f5_structure_comparison.csv")
    return results, differences, structures


def _safe_name(name):
    """
    Turn a structure's display name into something safe for a filename.

    WHAT: "learned (bic-d)" -> "learned_bic-d"
    WHY:  the display names contain spaces, brackets and commas.
    """
    return name.split(" ")[0] + "_" + name.split("(")[-1].rstrip(")").replace(" ", "_")


# Demo. Run: python -m src.f5_structure_learning
if __name__ == "__main__":
    pd.set_option("display.width", 160)
    results, differences, structures = run_comparison()
    print("=" * 78)
    print("EXPERT STRUCTURE vs STRUCTURE LEARNED FROM DATA")
    print("=" * 78)
    print(results.round(4).to_string())
    for name, difference in differences.items():
        if name.startswith("expert"):
            continue
        print(f"\n--- {name} compared with the expert structure ---")
        for kind, pairs in difference.items():
            arrows = ", ".join(f"{a}->{b}" for a, b in pairs) or "(none)"
            print(f"  {kind:16s}: {arrows}")
    print(f"\nPNGs and the comparison table written to {OUTPUTS_DIR}")
