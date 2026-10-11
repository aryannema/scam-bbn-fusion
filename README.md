# Explainable Bayesian Graph Fusion for Real-Time Scam/Phishing Detection

A scam-SMS detector for **Hinglish** (Romanized Hindi mixed with English) that fuses
message wording with sender-network behaviour in a Bayesian network, and **explains
every verdict in one sentence**.

> BCSE306L Artificial Intelligence · VIT
> Eight functionalities (F1–F8), 145 tests, one-command reproduction.

```
Prior (before reading anything):                       0.16
a request for a secret (OTP, PIN, KYC) is HIGH:        0.28
the riskiness of the link is HIGH:                     0.73
how trustworthy the sender account looks is MEDIUM:    0.62
```

That chain is not a description of the decision. It **is** the decision — the
probability after each piece of evidence is added.

## Why this problem is hard

A scam message in Hinglish looks like this:

> *Aapka SBI khaata **bnd** ho jayega! **Turrant** verify karein: http://sbi-kyc-verify.xyz*

Hindi written in English letters has no fixed spelling. `turant`, `turrant`, `turat` and
`tarant` are all the same word, so a keyword filter catches none of them. Meanwhile a
detector that just blocks anything containing "OTP" would block every genuine bank
message. And when a detector does flag something, the user deserves to know why.

## What it does

| | |
|---|---|
| **Reads code-mixed text** | Tags each word Hindi/English, measures the code-mixing index, scam lexicon, urgency, credential requests and link risk (**F2**) |
| **Matches by sound, not spelling** | A 7-rule phonetic encoder maps `turant`/`turrant`/`turat`/`tarant` → `trt` (**F6**) |
| **Reads sender behaviour** | A sender→recipient graph gives degree, PageRank, clustering, burstiness and template-sharing (**F3**) |
| **Reasons, and explains** | A Bayesian network produces P(fraud) plus the evidence chain above (**F1**) |
| **Checks its own structure** | Hill-climbing search re-learns the network from data and we compare (**F5**) |
| **Combines both detectors** | A stack trained on out-of-fold predictions learns how much to trust each (**F4**) |
| **Answers "what would have changed this?"** | The smallest evidence change that flips the verdict (**F7**) |
| **Attacks itself** | Homoglyphs, leetspeak, zero-width characters, typos and respellings, against three defences (**F8**) |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/run_all.sh                  # builds the data, runs F1-F8, saves every table
streamlit run app/streamlit_app.py       # the demo: 7 sections, one per functionality
```

Full instructions, including the Windows commands and a Python 3.12 fallback, are in
[docs/SETUP.md](docs/SETUP.md).

## Headline results

**Fusion beats either half** (leak-safe split, held-out campaigns):

| feature set | F1 | ROC-AUC | false-positive rate |
|---|---|---|---|
| text only (F2+F6) | 0.963 | 0.997 | 0.013 |
| graph only (F3) | 0.949 | 0.993 | 0.000 |
| **fused** | **0.989** | **1.000** | **0.000** |

**But graph features need a sender you have seen before.** Under a time-based split,
graph-only recall collapses to 0.358 (cold start) while text holds at 0.932. That is why
both halves stay in the system.

**The phonetic layer earns its place under attack** (recall on attacked scam messages):

| attack | no defence | normalize | normalize + phonetic |
|---|---|---|---|
| homoglyphs | 0.707 | 0.888 | **0.937** |
| Hindi respellings | 0.889 | 0.831 | **0.894** |

Normalization cannot fix `turrant`, because it is not a trick — it is simply another
correct spelling. Only matching by sound recovers it.

Every table, with the time-split numbers and the expert-vs-learned structure comparison,
is in [docs/PAPER_NOTES.md](docs/PAPER_NOTES.md).

## Honest limitations

- **The sender graph is simulated.** No public dataset has Hinglish scam text *and*
  sender-network data. The graph results show the *method* works on structure of this
  kind; they are not evidence about real telecom traffic. This is our biggest threat to
  validity and it is discussed, not buried.
- **Scores on the leak-safe split are near ceiling**, because scam and benign senders
  come from different generators. The time-split numbers are more informative.
- **A public dataset we planned to use turned out to be 97.7% a re-upload of UCI.**
  We measured it, kept only the 87 genuinely new rows, and wrote it up — combining those
  two sources naively puts the same message in both train and test.

All limitations: [docs/PAPER_NOTES.md](docs/PAPER_NOTES.md#limitations-state-plainly-in-the-paper).

## Repository layout

```
src/
  config.py                  every path, seed and threshold in one place
  data_builder.py            builds the dataset + the simulated sender network
  preprocess.py              shared text normalization (also defence (b) in F8)
  f2_hinglish_features.py    F2  code-mixed text features
  f6_phonetic.py             F6  phonetic encoder for Romanized Hindi
  graph_builder.py           the sender graph
  f3_fused_features.py       F3  fused feature selection + key comparison
  f1_bbn.py                  F1  expert Bayesian network + explanation chain
  f5_structure_learning.py   F5  structure learning, expert vs learned
  f4_ensemble.py             F4  stacked ensemble
  f7_counterfactual.py       F7  counterfactual explanations
  f8_adversarial.py          F8  adversarial attacks and defences
app/streamlit_app.py         the demo, 7 sections tagged F1-F8
tests/                       145 tests, one file per functionality
docs/                        see below
scripts/run_all.sh           runs everything end to end
outputs/                     generated tables and charts (gitignored)
```

## Documentation

| Document | Read it for |
|---|---|
| [FUNCTIONALITY_MAP.md](docs/FUNCTIONALITY_MAP.md) | **Start here.** One row per functionality: file, entry function, how to demo it |
| [WALKTHROUGH.md](docs/WALKTHROUGH.md) | One message traced through all nine stages with real numbers |
| [VIVA_QA.md](docs/VIVA_QA.md) | 44 likely examiner questions with answers |
| [PAPER_NOTES.md](docs/PAPER_NOTES.md) | Contributions, all result tables, limitations, threats to validity |
| [REFERENCES.md](docs/REFERENCES.md) | 23 references, every link verified |
| [API_NOTES.md](docs/API_NOTES.md) | pgmpy 1.x API differences we hit, and why versions are pinned |
| [SETUP.md](docs/SETUP.md) | Environment setup and troubleshooting |

## How the code is written

Deliberate constraints, so that any author can open one file and walk an examiner
through it:

- One functionality per file, tagged `# FUNCTIONALITY F<n>`.
- Every function does one job, in about 25 lines or fewer, with a what/why/example
  docstring.
- Well-known libraries do the heavy lifting: pandas, scikit-learn, networkx, pgmpy,
  Streamlit. No custom ML algorithms where a library has one.
- Exact pinned versions, because pgmpy 1.x renamed the classes we depend on.
- Many tests are named after the bug they prevent, for example
  `test_digit_runs_never_collapse` (normalization once turned `₹10,000` into `₹10,0`)
  and `test_host_is_parsed_not_substring_matched` (`"t.co" in "flipkart.com"` is true).

## Team

- **Aryan** — branch `aryan`
- **Sumit** — own branch

Both branches merge into `main` by pull request.
