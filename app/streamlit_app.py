"""
The demo app. One page, seven sections, each tagged with the functionality it shows.

WHAT: paste a message, pick a sender, and watch every part of the system react.
WHY:  this is what we open in the viva. Each section header carries its F-number so we
      can point straight at the functionality being asked about.
HOW:  streamlit run app/streamlit_app.py

Everything heavy - the dataset, the graph, the trained models - is loaded ONCE and
cached by Streamlit. Without that, every keystroke would retrain the ensemble.
"""

import sys
from pathlib import Path

# Make `import src...` work when Streamlit runs this file directly from app/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from src.config import COMBINED_CSV, FRAUD_THRESHOLD, OUTPUTS_DIR, PROCESSED_DIR
from src.f1_bbn import evidence_from_features, score_message
from src.f2_hinglish_features import SCAM_LEXICON, extract_text_features
from src.f4_ensemble import predict_one, train_ensemble
from src.f6_phonetic import SCAM_LEXICON_CODES, phonetic_code
from src.f7_counterfactual import find_counterfactuals
from src.f8_adversarial import ATTACKS, features_for, train_text_detector
from src.f3_fused_features import build_fused_table
from src.graph_builder import build_graph, neighbourhood
from src.preprocess import normalize, tokenize

EXAMPLE = "Aapka SBI KYC band ho jayega! Turant verify karein: http://sbi-kyc.xyz"


@st.cache_resource(show_spinner="Loading data and training the models (once)...")
def load_everything():
    """
    Load the dataset and train every model, once per app session.

    WHAT: returns a dict with the messages, senders, graph and fitted models.
    WHY:  training the ensemble takes a few seconds. Streamlit re-runs this script on
          every interaction, so without caching the app would be unusable.
    """
    messages = pd.read_csv(COMBINED_CSV)
    senders = pd.read_csv(PROCESSED_DIR / "senders.csv")
    comparison, parts = train_ensemble("split_leaksafe")
    is_train = (messages["split_leaksafe"] == "train").to_numpy()
    detectors = {name: train_text_detector(messages["text"][is_train],
                                           messages["label"][is_train].to_numpy(), name)
                 for name in ["none", "normalize", "normalize+phonetic"]}
    return {"messages": messages, "senders": senders, "parts": parts,
            "ensemble_table": comparison, "graph": build_graph(messages),
            "detectors": detectors}


def features_for_message(bundle, text, sender_id):
    """
    Build one feature row: this text, as if sent by this sender.

    WHAT: returns a pandas Series of all 34 fused features.
    WHY:  the models expect the same columns in the same order as during training, so
          we reuse build_fused_table rather than assembling the row by hand.
    NOTE: the graph features come from the sender's REAL traffic in the dataset. A
          sender id we have never seen gets zeros, which is the honest cold-start case.
    """
    row = pd.DataFrame([{"text": text, "sender_id": sender_id,
                         "recipient_id": "R00000", "timestamp": "2026-01-01"}])
    table = build_fused_table(row, bundle["senders"], graph_source=bundle["messages"])
    return table.iloc[0]


def section_score(bundle, text, sender_id):
    """Section 1: the headline number, from all three detectors."""
    st.subheader("1. Fraud score  ·  F1 + F3 + F4")
    features = features_for_message(bundle, text, sender_id)
    result = predict_one(bundle["parts"], features)
    verdict = "FRAUD" if result["ensemble"] >= FRAUD_THRESHOLD else "SAFE"
    left, middle, right = st.columns(3)
    left.metric("Ensemble (F4)", f"{result['ensemble']:.2f}", verdict)
    middle.metric("Feature model (F3)", f"{result['feature_model']:.2f}")
    right.metric("Bayesian network (F1)", f"{result['bbn']:.2f}")
    st.progress(min(float(result["ensemble"]), 1.0))
    st.caption(f"Threshold is {FRAUD_THRESHOLD:.2f}. The three numbers can disagree - "
               "the stack in F4 learned how much to trust each one.")
    return features, result


def section_explanation(result):
    """Section 2: the F1 explanation chain, in plain English."""
    st.subheader("2. Why  ·  F1 explanation chain")
    detail = result["bbn_detail"]
    st.write("Evidence read from this message and sender:")
    st.json({node: value for node, value in result["evidence"].items()})
    st.write("The probability moves as each piece of evidence is added, strongest first:")
    for line in detail["explanation"]:
        st.markdown(f"- {line}")
    st.success(detail["sentence"])


def section_counterfactual(bundle, result):
    """Section 3: the F7 smallest-change explanation."""
    st.subheader("3. What would have changed the verdict  ·  F7 counterfactual")
    options = find_counterfactuals(bundle["parts"]["engine"], result["evidence"])
    if not options:
        st.info("No change of up to three pieces of evidence would flip this verdict.")
        return
    st.caption(f"Smallest change that flips it: {options[0]['n_changes']} "
               f"piece(s) of evidence. {len(options)} option(s) found.")
    for option in options[:5]:
        st.markdown(f"- {option['sentence']}")


def section_graph(bundle, sender_id):
    """Section 4: draw the sender's corner of the network."""
    st.subheader("4. Who this sender talks to  ·  F3 sender graph")
    around = neighbourhood(bundle["graph"], sender_id)
    if around.number_of_nodes() == 0:
        st.info(f"'{sender_id}' sent nothing in our dataset, so it has no neighbourhood "
                "(this is the cold-start case).")
        return
    figure, axis = plt.subplots(figsize=(7, 4.5))
    colours = ["#ff6b6b" if node == sender_id else "#cfe3ff" for node in around.nodes()]
    nx.draw(around, nx.spring_layout(around, seed=3), ax=axis, node_color=colours,
            node_size=420, with_labels=True, font_size=6, arrowsize=10,
            edge_color="#999999")
    axis.set_title(f"{sender_id}: {around.number_of_nodes() - 1} contacts")
    st.pyplot(figure)
    plt.close(figure)


def section_phonetic(text):
    """Section 5: every token next to its phonetic code."""
    st.subheader("5. How the words sound  ·  F6 phonetic layer")
    tokens = tokenize(normalize(text))
    rows = []
    for token in tokens:
        code = phonetic_code(token)
        rows.append({"token": token, "phonetic code": code,
                     "exact lexicon hit": "yes" if token in SCAM_LEXICON else "",
                     "sound-alike hit": "yes" if code in SCAM_LEXICON_CODES else ""})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("A 'sound-alike hit' with no exact hit is a scam word caught only "
               "because of its sound - for example turrant, which F2 alone misses.")


def section_attack(bundle, text):
    """Section 6: attack the live message and re-score it under each defence."""
    st.subheader("6. Attack this message  ·  F8 adversarial robustness")
    if not st.button("Perturb it and re-score"):
        st.caption("Rewrites the message the way a scammer would, then scores each "
                   "version with and without our defences.")
        return
    rng = np.random.default_rng(0)
    rows = []
    for name, attack in list(ATTACKS.items()):
        attacked = attack(text, rng)
        row = {"attack": name, "message after attack": attacked[:60]}
        for defence, model in bundle["detectors"].items():
            matrix = features_for([attacked], defence).to_numpy()
            row[defence] = round(float(model.predict_proba(matrix)[:, 1][0]), 3)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Each number is the probability the TEXT-ONLY detector still assigns. "
               "Higher is better: it means the attack failed.")


def section_tables(bundle):
    """Section 7: the result tables from F3, F4 and F5."""
    st.subheader("7. Results tables  ·  F3 / F4 / F5")
    f3_tab, f4_tab, f5_tab, f8_tab = st.tabs(
        ["F3 text vs graph vs fused", "F4 ensemble", "F5 structures", "F8 robustness"])
    with f3_tab:
        _show_csv(OUTPUTS_DIR / "f3_comparison_split_leaksafe.csv",
                  "Run `python -m src.f3_fused_features` to generate this.")
    with f4_tab:
        st.dataframe(bundle["ensemble_table"].round(4), width="stretch")
        st.write("What the stack learned to trust:", bundle["parts"]["weights"])
    with f5_tab:
        _show_csv(OUTPUTS_DIR / "f5_structure_comparison.csv",
                  "Run `python -m src.f5_structure_learning` to generate this.")
        for picture in sorted(OUTPUTS_DIR.glob("f5_structure_*.png")):
            st.image(str(picture), caption=picture.stem)
    with f8_tab:
        _show_csv(OUTPUTS_DIR / "f8_adversarial.csv",
                  "Run `python -m src.f8_adversarial` to generate this.")
        chart = OUTPUTS_DIR / "f8_adversarial.png"
        if chart.exists():
            st.image(str(chart))


def _show_csv(path, hint):
    """Show a saved results table, or say how to create it."""
    if path.exists():
        st.dataframe(pd.read_csv(path, index_col=0).round(4), width="stretch")
    else:
        st.info(hint)


def main():
    """Lay out the page and run every section in order."""
    st.set_page_config(page_title="Explainable Scam Detection", layout="wide")
    st.title("Explainable Bayesian Graph Fusion for Scam Detection")
    st.caption("BCSE306L Artificial Intelligence · every section is tagged with the "
               "functionality it demonstrates")
    bundle = load_everything()
    known_senders = sorted(bundle["messages"]["sender_id"].unique())
    with st.sidebar:
        st.header("Input")
        text = st.text_area("Message", EXAMPLE, height=140)
        sender_id = st.selectbox("Sender (from the dataset)", known_senders,
                                 index=min(7, len(known_senders) - 1))
        custom = st.text_input("...or type an unknown sender id", "")
        st.caption("An unknown sender has no network history, so the graph features "
                   "are zero. That is the cold-start case the paper discusses.")
    sender_id = custom.strip() or sender_id
    features, result = section_score(bundle, text, sender_id)
    section_explanation(result)
    section_counterfactual(bundle, result)
    section_graph(bundle, sender_id)
    section_phonetic(text)
    section_attack(bundle, text)
    section_tables(bundle)


main()
