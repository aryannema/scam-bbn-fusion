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
MESSAGES_PER_SCAM_SENDER = 25    # few scam senders, each very busy  -> high fan-out
MESSAGES_PER_BENIGN_SENDER = 6   # many normal senders, each quiet   -> low fan-out
BENIGN_CONTACTS = 3              # a normal person texts the same ~3 people repeatedly

# With the message counts we actually build, the two numbers above work out to roughly
# 1,000 senders, which is the scale the project brief asks for.
N_RECIPIENTS = 3000

# How many messages one "ring" (one campaign) accounts for. Rings are the unit we hold
# out in the leak-safe split, so smaller rings = finer control over the split size.
MESSAGES_PER_RING = 300

# Label noise, so the graph features are NOT a giveaway and the task stays realistic:
# some scam senders behave like normal people, and some innocent senders look odd.
FRAC_SCAM_SENDERS_LOOK_NORMAL = 0.12  # 12% of scam senders hide in plain sight
FRAC_BENIGN_SENDERS_LOOK_ODD = 0.05   # 5% of benign senders look suspicious
