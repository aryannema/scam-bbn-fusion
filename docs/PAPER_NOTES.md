# Paper notes

Everything needed to write the paper: the claims, the numbers, and the things we must
admit. Every number here was produced by `bash scripts/run_all.sh`.

## Problem statement

Scam SMS aimed at Indian users is written in *Hinglish* — Romanized Hindi mixed with
English, with no fixed spelling (`turant` / `turrant` / `turat` are one word). Detectors
trained on English spam miss it, and black-box detectors cannot tell a user *why* a
message was flagged. We want a detector that works on code-mixed text, uses
who-sent-what network structure as well as wording, and explains itself in one sentence.

## Claimed contributions

1. An explainable **Bayesian fusion** of graph signals and code-mixed text signals,
   where the explanation is the inference chain itself rather than a post-hoc
   approximation.
2. A **phonetic normalization layer** for Romanized Hindi that survives spelling
   variation, with the anusvara (unwritten nasal) rule as its key component.
3. **Counterfactual explanations** over Bayesian evidence, enumerated exhaustively
   rather than approximated, because the evidence space is only 3⁵ = 243 states.
4. An **adversarial evaluation of our own detector** across five attacks and three
   defence configurations.
5. A comparison of an **expert-drawn** against a **data-learned** network structure.

## The dataset we built

| Source | Rows kept | Scam | Benign | Real or synthetic |
|---|---|---|---|---|
| UCI SMS Spam Collection | 5,171 | 653 | 4,518 | **real** English |
| Hugging Face Hinglish seed | 87 | 47 | 40 | **real** Hinglish |
| Our template generator | 2,500 | 1,250 | 1,250 | synthetic Hinglish |
| **Total** | **7,758** | **1,950** | **5,808** | |

Network: 1,108 simulated senders in 53 campaign rings, ~2,900 recipients, 60-day window.
9 scam template families, 7 benign families. Byte-identical across runs (md5 verified).

### A finding about the public Hinglish dataset (worth a paragraph in the paper)

`bolewara/hinglish-scam-text-dataset` advertises 3,787 rows, but we measured that
**3,700 of them (97.7%) are byte-identical to messages in the UCI SMS Spam
Collection** — it is largely a re-upload. Only 87 rows are new, and those are the
genuinely hand-written Hinglish ones. We deduplicate and keep only those 87.

This matters beyond tidiness: had we concatenated the two sources naively, 3,700
duplicate texts would have entered the pool, the *same message* could have landed in
both the train and test split, and every reported score would have been inflated.
Anyone combining these two public sources without checking will have this problem
silently. UCI itself also contains 403 internally duplicated messages, which we drop.

## Results

### Table 1 — text vs graph vs fused features (F3)

Leak-safe split (hold out whole campaigns):

| feature set | precision | recall | F1 | ROC-AUC | FPR | n features |
|---|---|---|---|---|---|---|
| text only (F2+F6) | 0.979 | 0.947 | 0.963 | 0.997 | 0.013 | 21 |
| graph only (F3) | 1.000 | 0.903 | 0.949 | 0.993 | 0.000 | 13 |
| **fused** | **1.000** | **0.978** | **0.989** | **1.000** | **0.000** | 34 |

Time-based split (train on the past, test on the future):

| feature set | precision | recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|
| text only | 0.945 | 0.932 | 0.939 | 0.989 | 0.006 |
| graph only | 1.000 | 0.358 | 0.527 | 0.942 | 0.000 |
| fused | 0.951 | 0.926 | 0.938 | 0.998 | 0.005 |

**Read these two tables together — this is the interesting result.** Fusion clearly
beats both halves when the sender is known (0.989 vs 0.963 / 0.949). But under a time
split, graph-only recall collapses to 0.358: a sender who first appears after the cutoff
has no history, so their graph features are zero. This is the **cold-start** problem, and
it is why the text side cannot be dropped. Fusion is never worse than the better half.

### Table 2 — the ensemble (F4), leak-safe split

| approach | precision | recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|
| BBN alone (F1) | 0.830 | 0.883 | 0.856 | 0.965 | 0.112 |
| feature model alone | 1.000 | 0.978 | 0.989 | 1.000 | 0.000 |
| plain average of the two | 0.998 | 0.969 | 0.984 | 1.000 | 0.001 |
| **stacked ensemble (F4)** | **1.000** | **0.980** | **0.990** | **1.000** | **0.000** |

Learned stack weights: **+9.65 feature model, +1.95 BBN**.

**Be honest about the size of this win.** Stacking clearly beats a plain average
(0.990 vs 0.984) — learning the weights is worth it. But it barely beats the feature
model alone (0.9897 vs 0.9888), and on the time split it is slightly *worse*
(0.931 vs 0.938). The BBN's contribution is explainability and counterfactuals, not
accuracy. On the time split the BBN alone falls to F1 0.238 / AUC 0.568, because two of
its five clues are graph-derived and therefore zero for unseen senders.

### Table 3 — expert vs learned structure (F5)

| structure | edges | held-out log-likelihood | F1 | ROC-AUC |
|---|---|---|---|---|
| expert (hand-drawn) | 6 | −4.674 | 0.8555 | **0.9651** |
| learned (BIC) | 13 | **−3.916** | 0.8548 | 0.9418 |
| learned (BDeu) | 14 | −3.944 | 0.8548 | 0.9418 |

Both learned structures **dropped `PatternAnomaly → Fraud`** and wired
`UrgencyCue → Fraud` directly, adding many edges among the evidence nodes themselves
(e.g. `LinkRisk → CredentialAsk`). So the data says our evidence variables are not
conditionally independent, which our hand-drawn graph assumed.

The trade-off is clean and worth reporting: the learned structures explain held-out data
better (log-likelihood −3.92 vs −4.67) but **rank** fraud worse (AUC 0.942 vs 0.965).
Our hidden `PatternAnomaly` layer costs likelihood and buys ranking quality.

### Table 4 — adversarial robustness (F8), recall on held-out scam messages

| attack | no defence | normalize | normalize + phonetic |
|---|---|---|---|
| clean (no attack) | 0.932 | 0.884 | **0.947** |
| homoglyphs | 0.707 | 0.888 | **0.937** |
| leetspeak | 0.905 | 0.949 | **0.957** |
| zero-width characters | 0.900 | 0.894 | **0.939** |
| typos | 0.881 | 0.847 | 0.865 |
| Hindi respellings | 0.889 | 0.831 | **0.894** |

The homoglyph attack is the most damaging undefended (0.932 → 0.707, a 24% attack
success rate) and normalization almost fully repairs it. The **Hindi respelling attack
is the one that justifies F6**: normalization alone makes it *worse* (0.831), because
`turrant` is not a trick to undo — it is simply another correct spelling — and only
sound-based matching recovers it (0.894). Typos are the weakest point of all three
configurations; we do not claim a defence against them.

Note that normalization costs clean recall (0.932 → 0.884): it discards some signal,
since unusual characters are themselves mildly predictive. Adding the phonetic layer
recovers that and more (0.947), which is why `normalize + phonetic` is the recommended
configuration.

## Limitations (state plainly in the paper)

- **The sender graph is simulated.** No public dataset carries Hinglish scam text *and*
  sender-network information, so the network comes from our own ring/fan-out model.
  Graph-feature results show the *method* works on network structure of this kind; they
  are **not** evidence about effect size on real telecom data. This is the single
  biggest threat to validity.
- **Scores on the leak-safe split are near ceiling** (fused F1 0.989), because scam and
  benign senders are generated by different processes. Even after deliberately blurring
  the simulation (see below), graph features remain more separable than they would be in
  reality. The time-split numbers are the more informative ones.
- **Most Hinglish messages are synthetic**, from 9 scam and 7 benign template families.
  Only 87 rows are real hand-written Hinglish.
- **The language word lists are hand-written and small.** An unlisted word is tagged
  "unknown", so the code-mixing index is a *lower bound* on real mixing. We publish
  `unknown_ratio` as a feature (0.65 benign, 0.37 scam) so this is visible.
- **Phonetic matching cannot be perfect.** About 10.6% of benign messages pick up at
  least one phonetic hit from an innocent word that sounds like a scam word
  (`later` / `lottery` both encode to `ltr`).
- **No defence against typo attacks** (Table 4, row 5).
- **The BBN is the weakest detector on its own** and collapses under cold start.

## Threats to validity, and what we did about each

| Threat | Mitigation |
|--------|------------|
| Random row splits would leak template wording across train/test | Split by **campaign**: each sender ring runs exactly one template family, so holding out a ring and holding out a wording are the same action. Verified: 0 of 15 synthetic families, 0 of 53 rings and 0 of 1,108 senders appear on both sides |
| Duplicate text across the split boundary | Deduplicated globally, including the 3,700 UCI/Hugging Face overlaps; a test asserts 0 duplicates and 0 shared texts |
| Feature selection could see the test set | `SelectKBest` and the discretization bin edges are fitted on training rows only |
| Graph features could leak the future into the past | For the time split, graph features are computed from training-period messages only; later senders get the honest cold-start zeros |
| Message volume alone identified the class | It once reached F1 = 1.00. Sender activity now comes from an overlapping lognormal; volume alone is AUC 0.70, with 1,004 of 1,012 senders in the overlap band |
| Account age alone identified the class | The ranges once had **zero** overlap (1–120 days vs 200–3,000). Now 1–400 against 30–3,000 |
| Graph shape could be a perfect lie detector | Contact lists key on observable **behaviour**, never on the hidden label, so 12% of scammers keep innocent-looking social circles and 5% of innocent senders look odd — exact counts, not per-sender dice rolls |
| "Has a link" could identify a scam | Only 1 of 5,808 benign messages had a URL. Benign templates now carry legitimate links (7.8%), and a `promo_optin` family uses real shortened links, so **P(scam \| risky link) = 0.757**, not 1.0 |
| Hand-picked thresholds tuned to the test set | One threshold (`FRAUD_THRESHOLD = 0.50`) fixed in `src/config.py` before any evaluation, shared by F1, F7 and F8 |
| Results might not be reproducible | The dataset is byte-identical across runs (md5). This required removing Python's `hash()`, whose string hashing is salted per process |

## Bugs we found in our own code (worth a "lessons learned" paragraph)

All three were found by measuring our own output, not by reading the code:

1. `collapse_repeats` turned `₹10,000` into `₹10,0` — three zeros are a repeated
   character — corrupting every money amount in the dataset.
2. Link risk was judged with `"t.co" in text`, which is **true** for "flipkar**t.co**m",
   so every legitimate Flipkart tracking link was labelled a URL shortener.
3. `LinkRisk` silently had only two states. It is a skewed count, so both the 33rd and
   67th percentile were 0 and the quantile bins collapsed; querying `LinkRisk = high`
   raised `KeyError`.

Each now has a named regression test.

## Future work

- Obtain real sender-network data under an NDA, or a telecom collaboration, to replace
  the simulated graph — the clearest path to a stronger paper.
- A defence against typo attacks (character n-grams, or a RETVec-style vectorizer).
- Learn the phonetic rules from transliteration pairs rather than writing them by hand.
- Handle cold-start senders with a prior over account age rather than zeros.
