"""
Builds the sender -> recipient graph and measures the shape of each sender's behaviour.

WHAT: `build_sender_features(messages)` -> one row of graph numbers per sender.
WHY:  wording is only half of a scam. The other half is BEHAVIOUR: a scam campaign is
      a few accounts blasting hundreds of strangers inside one hour, while a real
      person texts the same three people for weeks. That difference is visible in the
      graph even when the text is perfectly written.
HOW:  run `python -m src.graph_builder` for a demo.

HONEST NOTE: this graph is simulated (see src/data_builder.py and
docs/PAPER_NOTES.md). No public dataset carries Hinglish scam text together with
sender-network information. These results show the METHOD works on network structure
of this kind; they are not evidence about the size of the effect on real telecom data.

No labels are used anywhere in this file. Every number below is computed from who
messaged whom and when, which is exactly what a phone network can see before anyone
has decided whether a message is a scam.
"""

import networkx as nx
import numpy as np
import pandas as pd

from src.config import BETWEENNESS_SAMPLE, RANDOM_SEED
from src.preprocess import normalize, tokenize

# How many opening words define a "template fingerprint". Five is enough to tell
# "aapka sbi kyc band ho" apart from "beta khana kha liya".
FINGERPRINT_TOKENS = 5


def build_graph(messages):
    """
    Build the directed graph: an arrow from each sender to each recipient.

    WHAT: returns a networkx DiGraph. Edge weight = how many messages were sent.
    WHY:  once the data is a graph, networkx gives us PageRank and betweenness for
          free, instead of us writing centrality algorithms ourselves.
    EXAMPLE: build_graph(df).number_of_nodes() -> about 4,000
    """
    graph = nx.DiGraph()
    for sender, recipient in zip(messages["sender_id"], messages["recipient_id"]):
        if graph.has_edge(sender, recipient):
            graph[sender][recipient]["weight"] += 1     # same pair messaging again
        else:
            graph.add_edge(sender, recipient, weight=1)
    return graph


def degree_features(messages, graph):
    """
    Count how many people each sender writes to, and how often.

    WHAT: columns out_degree, in_degree, messages_sent, unique_recipient_ratio.
    WHY:  `unique_recipient_ratio` is the single clearest campaign signal: near 1.0
          means "a different stranger every time", low means "the same few contacts".
    """
    sent = messages.groupby("sender_id").size().rename("messages_sent")
    frame = pd.DataFrame(sent)
    frame["out_degree"] = [graph.out_degree(s) for s in frame.index]
    frame["in_degree"] = [graph.in_degree(s) for s in frame.index]
    # how much of the sending went to DIFFERENT people rather than the same person
    frame["unique_recipient_ratio"] = frame["out_degree"] / frame["messages_sent"]
    return frame


def centrality_features(graph):
    """
    Measure how central each sender is in the whole network.

    WHAT: columns pagerank and betweenness.
    WHY:  PageRank asks "how much of the traffic flows through here"; betweenness asks
          "how often is this account on the shortest path between two others", which
          is how a forwarding hub looks.
    NOTE: exact betweenness is slow on a graph this size, so we approximate it from a
          random sample of source nodes (k=BETWEENNESS_SAMPLE). networkx supports this
          directly and the seed keeps it reproducible.
    """
    pagerank = nx.pagerank(graph, weight="weight")
    sample = min(BETWEENNESS_SAMPLE, graph.number_of_nodes())
    betweenness = nx.betweenness_centrality(graph, k=sample, seed=RANDOM_SEED)
    return pd.DataFrame({"pagerank": pagerank, "betweenness": betweenness})


def clustering_features(graph):
    """
    Measure whether a sender's contacts also talk to each other.

    WHAT: a column `clustering`.
    WHY:  a real person sits inside a friend group, so their contacts overlap. A
          scammer's victims are strangers who have nothing to do with one another, so
          the clustering is near zero.
    NOTE: clustering is defined on an undirected graph, so we drop the arrow
          directions first. That is a modelling choice, not a bug.
    """
    undirected = graph.to_undirected()
    return pd.DataFrame({"clustering": nx.clustering(undirected)})


def timing_features(messages):
    """
    Measure how bursty each sender is.

    WHAT: columns active_hours, messages_per_hour, burstiness.
    WHY:  a campaign empties its list in one sitting; a person spreads messages over
          weeks. `burstiness` is messages divided by the hours they span, so a sender
          who sent 25 messages in one hour scores far above one who sent 6 in a month.
    """
    stamps = pd.to_datetime(messages["timestamp"])
    grouped = messages.assign(timestamp=stamps).groupby("sender_id")["timestamp"]
    span_hours = (grouped.max() - grouped.min()).dt.total_seconds() / 3600.0
    counts = grouped.size()
    frame = pd.DataFrame({"active_hours": span_hours, "messages_sent_t": counts})
    # +1 hour so a sender with a single message does not divide by zero
    frame["burstiness"] = frame["messages_sent_t"] / (frame["active_hours"] + 1.0)
    frame["messages_per_hour"] = frame["burstiness"]
    return frame[["active_hours", "burstiness", "messages_per_hour"]]


def template_fingerprint(text):
    """
    Reduce a message to the first few words of its cleaned text.

    WHAT: "Aapka SBI KYC band ho jayega! Turant..." -> "aapka sbi kyc band ho"
    WHY:  two senders running the same campaign send messages that START the same way.
          Comparing fingerprints finds that, without ever looking at a label.
    """
    return " ".join(tokenize(normalize(text))[:FINGERPRINT_TOKENS])


def sharing_features(messages):
    """
    Measure how much of a sender's text is also sent by OTHER senders.

    WHAT: a column `forwarding_ratio`, between 0 and 1.
    WHY:  this is the shared-template signal that gives a ring away. If three accounts
          all open with "aapka sbi kyc band ho", they are running one script. A person
          writing their own messages scores 0.
    """
    frame = messages[["sender_id"]].copy()
    frame["fingerprint"] = [template_fingerprint(text) for text in messages["text"]]
    # for each fingerprint, how many DIFFERENT senders use it
    senders_per_print = frame.groupby("fingerprint")["sender_id"].nunique()
    frame["shared"] = frame["fingerprint"].map(senders_per_print) > 1
    return frame.groupby("sender_id")["shared"].mean().rename("forwarding_ratio").to_frame()


def build_sender_features(messages):
    """
    The entry function: every graph feature for every sender, in one table.

    WHAT: a DataFrame indexed by sender_id.
    WHY:  F3 joins this onto the text features to make the fused feature set.
    EXAMPLE: build_sender_features(df).loc["S00007", "unique_recipient_ratio"]
    """
    graph = build_graph(messages)
    parts = [
        degree_features(messages, graph),
        centrality_features(graph),
        clustering_features(graph),
        timing_features(messages),
        sharing_features(messages),
    ]
    # join on sender_id; centrality/clustering also cover recipients, so keep senders
    features = parts[0]
    for part in parts[1:]:
        features = features.join(part, how="left")
    return features.fillna(0.0)


def neighbourhood(graph, sender, hops=1):
    """
    Pull out the small piece of the graph around one sender, for the demo app.

    WHAT: returns a subgraph with the sender and the people they messaged.
    WHY:  the Streamlit app (section 4) draws this so a viewer can SEE the fan-out.
    """
    if sender not in graph:
        return nx.DiGraph()
    nodes = {sender}
    for _ in range(hops):
        for node in list(nodes):
            nodes |= set(graph.successors(node)) | set(graph.predecessors(node))
    return graph.subgraph(nodes).copy()


# Demo. Run: python -m src.graph_builder
if __name__ == "__main__":
    from src.config import COMBINED_CSV, PROCESSED_DIR

    messages = pd.read_csv(COMBINED_CSV)
    senders = pd.read_csv(PROCESSED_DIR / "senders.csv")
    print(f"building graph from {len(messages)} messages...")
    features = build_sender_features(messages)
    graph = build_graph(messages)
    print(f"graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"features: {features.shape[0]} senders x {features.shape[1]} columns\n")
    joined = senders.set_index("sender_id").join(features, how="inner")
    columns = ["messages_sent", "out_degree", "unique_recipient_ratio", "pagerank",
               "clustering", "burstiness", "forwarding_ratio"]
    print("AVERAGE BEHAVIOUR, by what the sender really is")
    print("=" * 96)
    print(joined.groupby("is_scammer")[columns].mean().round(4).to_string())
    print("\nSame table, but grouped by how the sender BEHAVES (12% of scammers lie low):")
    print(joined.groupby("behaves_scammy")[columns].mean().round(4).to_string())
    print("\nNOTE: the two tables differ. That is the label noise doing its job - the")
    print("graph shows behaviour, which is not always the same as guilt.")
