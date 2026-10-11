"""
Central configuration for the whole project.

WHAT: every path, random seed and threshold used anywhere in this project lives here.
WHY:  if a number appears in two files, the two files will disagree one day. Importing
      from here means there is exactly ONE place to look and ONE place to change.
HOW:  other files do `from src.config import RANDOM_SEED, PROCESSED_DIR`.

Nothing in this file does any work. It only defines constants.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# __file__ is ".../scam-bbn-fusion/src/config.py".
# .resolve() makes it absolute, .parent is "src/", .parent.parent is the repo root.
# Doing it this way means the code works no matter which folder you run it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # downloaded public datasets (gitignored)
PROCESSED_DIR = DATA_DIR / "processed"  # small CSVs we generate (committed)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"  # PNG charts and result tables (gitignored)
DOCS_DIR = PROJECT_ROOT / "docs"

# The one combined dataset that every functionality reads.
# Columns: message_id, text, sender_id, recipient_id, timestamp, label, language_mix, source
COMBINED_CSV = PROCESSED_DIR / "combined_dataset.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# Every random choice (synthetic data, train/test splits, model init) uses this seed,
# so re-running the project gives the same numbers as the ones in our paper.
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Decision threshold
# ---------------------------------------------------------------------------
# A message is called FRAUD when its predicted probability is >= this value.
# Used by F1 (verdict), F7 (a counterfactual must push the score below it) and
# F8 (an attack "succeeds" when it drags a scam message below it).
FRAUD_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------
# A Bayesian network over discrete nodes needs each number turned into a category.
# We use three bins by quantile, so each bin holds roughly a third of the training data.
BIN_LABELS = ("low", "medium", "high")
N_BINS = 3

# ---------------------------------------------------------------------------
# Dataset sources (used by src/data_builder.py in Phase 1)
# ---------------------------------------------------------------------------
# 1) English spam/ham, ~5.5k real messages, public.
UCI_SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

# 2) Small real Hinglish scam seed (~60 hand-written rows) from Hugging Face.
HF_HINGLISH_DATASET = "bolewara/hinglish-scam-text-dataset"

# 3) How many Hinglish messages we generate ourselves from templates.
#    HONEST NOTE: these are synthetic. No public dataset has Hinglish scam text AND
#    sender-network information, so we generate both. This is stated as a limitation
#    in docs/PAPER_NOTES.md.
N_SYNTHETIC_MESSAGES = 2500

# ---------------------------------------------------------------------------
# Synthetic sender network (used by src/data_builder.py in Phase 1)
# ---------------------------------------------------------------------------
# We do NOT fix the sender count directly. Instead we fix how BUSY a sender is and let
# the count follow from the number of messages, because that is what creates the signal
# F3 looks for: a scam campaign is a few senders blasting many strangers, while a normal
# person sends a few messages to the same handful of contacts.
# These are AVERAGES, not fixed quotas. Each sender also gets a random activity weight
# (see ACTIVITY_SPREAD), so the two groups overlap heavily instead of sitting either
# side of a clean threshold.
#
# WHY THE NUMBERS ARE THIS CLOSE: with 25 against 6 and no spread, message volume WAS
# the label. `out_degree` alone reached F1 = 1.00 and the model ignored all 33 other
# features. Perfect scores mean the simulation leaked the answer, not that the method
# works, so the gap is deliberately modest and blurred.
MESSAGES_PER_SCAM_SENDER = 14    # scam senders are busier on average...
MESSAGES_PER_BENIGN_SENDER = 6   # ...but plenty of normal people are busy too

# Sigma of the lognormal activity weight. Higher = more overlap between the groups.
# Real traffic is heavy tailed: a few accounts send far more than everyone else, in
# BOTH groups (a shop or a chatty friend, and a cautious low-volume fraudster).
ACTIVITY_SPREAD = 0.9

# A campaign does not always finish in an afternoon. Burst windows are drawn from this
# range in hours, so some rings blast in 2 hours and others trickle over 10 days -
# otherwise `burstiness` separates the classes perfectly on its own.
BURST_HOURS_MIN = 2
BURST_HOURS_MAX = 240
BENIGN_CONTACTS = 3              # a normal person texts the same ~3 people repeatedly

# Normal people are grouped into small social circles and mostly message inside them.
# This matters for the graph: friends-of-friends create TRIANGLES, which is what the
# clustering coefficient measures. Without circles the graph is strictly
# sender -> recipient (bipartite), every clustering value is exactly 0, and the feature
# is dead weight.
SOCIAL_CIRCLE = 6

# A scam ring shares one victim list, so ring members hit overlapping targets - which
# is what makes a ring detectable as a group rather than as separate accounts.
RING_VICTIM_POOL = 150

# With the message counts we actually build, the two numbers above work out to roughly
# 1,000 senders, which is the scale the project brief asks for.
N_RECIPIENTS = 3000

# The most parents any node may have during structure learning (F5). A cap is needed
# because a node with many parents needs a probability table bigger than our data can
# fill reliably.
MAX_INDEGREE = 3

# F7 tries changing 1 variable, then 2, then 3. Beyond 3 a 'counterfactual' stops
# being an explanation anyone can act on.
MAX_COUNTERFACTUAL_CHANGES = 3

# Folds used for the out-of-fold predictions that train the F4 stack.
N_FOLDS = 5

# How many features F3 keeps when it reports a selected shortlist.
TOP_K_FEATURES = 15

# Betweenness centrality is slow to compute exactly, so F3 approximates it from this
# many randomly chosen source nodes. 200 is plenty for a graph of a few thousand nodes.
BETWEENNESS_SAMPLE = 200

# How many messages one "ring" (one campaign) accounts for. Rings are the unit we hold
# out in the leak-safe split, so smaller rings = finer control over the split size.
MESSAGES_PER_RING = 300

# Label noise, so the graph features are NOT a giveaway and the task stays realistic:
# some scam senders behave like normal people, and some innocent senders look odd.
FRAC_SCAM_SENDERS_LOOK_NORMAL = 0.12  # 12% of scam senders hide in plain sight
FRAC_BENIGN_SENDERS_LOOK_ODD = 0.05   # 5% of benign senders look suspicious
