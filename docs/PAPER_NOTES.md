# Paper notes

> **Stub — grows every phase, finalized in Phase 9.**

## Problem statement

Scam SMS aimed at Indian users is written in *Hinglish* — Romanized Hindi mixed with English —
with no fixed spelling. Detectors trained on English spam miss it, and black-box detectors cannot
tell a user *why* a message was flagged. We want a detector that is accurate on code-mixed text,
uses who-sent-what network structure as well as wording, and can explain itself in one sentence.

## Claimed contributions

1. An explainable **Bayesian** fusion of graph signals and code-mixed text signals, where the
   explanation is the inference chain itself rather than a post-hoc approximation.
2. A **phonetic normalization layer** for Romanized Hindi that survives spelling variation.
3. **Counterfactual explanations** over Bayesian evidence ("the smallest change that would have
   made this safe").
4. An **adversarial evaluation** of our own detector, with a measured defence.
5. A comparison of an **expert-drawn** network structure against a **data-learned** one.

## The dataset we actually built (Phase 1)

| Source | Rows kept | Scam | Benign | Real or synthetic |
|---|---|---|---|---|
| UCI SMS Spam Collection | 5,171 | 653 | 4,518 | **real** English |
| Hugging Face Hinglish seed | 87 | 47 | 40 | **real** Hinglish |
| Our template generator | 2,500 | 1,250 | 1,250 | synthetic Hinglish |
| **Total** | **7,758** | **1,950** | **5,808** | |

Network: 1,047 simulated senders in 51 campaign rings, 2,917 recipients, 60-day window.

### A finding about the public Hinglish dataset (worth one paragraph in the paper)

The Hugging Face dataset `bolewara/hinglish-scam-text-dataset` advertises 3,787 rows, but we
measured that **3,700 of them (97.7%) are byte-identical to messages in the UCI SMS Spam
Collection** — it is largely a re-upload. Only 87 rows are new, and those are the genuinely
hand-written Hinglish ones. We therefore deduplicate against UCI and keep only those 87.

This matters beyond tidiness: had we concatenated the two sources naively, 3,700 duplicate
texts would have entered the pool, the *same message* could have landed in both the training
and test split, and every reported score would have been inflated. Anyone combining these two
public sources without checking will have this problem silently.

UCI itself also contains 403 internally duplicated messages, which we drop for the same reason.

## Limitations (to state plainly in the paper)

- **The sender graph is simulated.** No public dataset carries Hinglish scam text *and* sender-network
  information, so the network is generated from our own ring/fan-out model. Graph-feature results therefore
  demonstrate that the *method* works on network structure of this kind; they are not evidence about the
  magnitude of the effect on real telecom data. This is the single biggest threat to validity.
- **Most Hinglish messages are synthetic**, generated from ~9 scam template families plus benign templates.
  Only a small seed (~60 rows) is real, hand-written Hinglish.
- Template-generated text can be *too* learnable, which is exactly why the evaluation split is
  leak-safe (see below) rather than random.

## Threats to validity and what we do about them

| Threat | Mitigation |
|--------|------------|
| Random row splits would leak template wording across train/test and inflate scores | Split by **campaign**: each sender ring runs exactly one template family, so holding out a ring and holding out a wording are the same action. Verified by test: 0 of 15 synthetic families, 0 of 51 rings and 0 of 1,047 senders appear on both sides |
| Duplicate text across the split boundary | Deduplicated globally; a test asserts 0 duplicates and 0 shared texts between train and test |
| Time-ordered deployment differs from a random split | Also report a **time-based split** |
| Synthetic graphs are trivially separable | Inject label noise: ~12% of scam senders behave normally, ~5% of benign senders look odd |
| Hand-picked thresholds could be tuned to the test set | One threshold (`FRAUD_THRESHOLD = 0.50`) fixed in `src/config.py` before evaluation |
| Synthetic graph could make fan-out a perfect giveaway | Jitter added so a scammer sometimes re-contacts a victim and a normal person sometimes meets someone new. Scammy senders average 19.6 unique recipients vs 3.7, but the unique-recipient ratios overlap heavily (scammy min 0.40 < normal max 1.00) |
| Results might not be reproducible | The whole dataset is byte-identical across separate runs (verified by md5). This required removing Python's `hash()` from the generator, since string hashing is salted per process |

## Results tables to include

1. text-only vs graph-only vs fused features (F3)
2. BBN alone vs feature model vs plain average vs stacked ensemble (F4)
3. expert vs learned structure: edges added/removed, held-out log-likelihood, F1 (F5)
4. adversarial recall drop under three defence configurations (F8)
