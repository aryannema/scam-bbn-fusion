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
| Random row splits would leak template wording across train/test and inflate scores | Split by **template family and sender ring**, so a family seen in training never appears in test |
| Time-ordered deployment differs from a random split | Also report a **time-based split** |
| Synthetic graphs are trivially separable | Inject label noise: ~12% of scam senders behave normally, ~5% of benign senders look odd |
| Hand-picked thresholds could be tuned to the test set | One threshold (`FRAUD_THRESHOLD = 0.50`) fixed in `src/config.py` before evaluation |

## Results tables to include

1. text-only vs graph-only vs fused features (F3)
2. BBN alone vs feature model vs plain average vs stacked ensemble (F4)
3. expert vs learned structure: edges added/removed, held-out log-likelihood, F1 (F5)
4. adversarial recall drop under three defence configurations (F8)
