# Walkthrough — one message through the whole pipeline

> **Stub — written in full in Phase 9**, once every stage exists and the file/line pointers are stable.

This document will follow a single message, end to end, in plain English, so that someone who
did not write the code can still understand it:

> `KYC band ho jayega! Turant verify karein: http://sbi-kyc-verify.xyz`

Planned sections, in pipeline order:

1. **The raw message arrives** — what we know: text, sender id, recipient id, timestamp.
2. **Normalization** (`preprocess.normalize`) — zero-width characters stripped, leetspeak undone.
3. **Text features** (F2) — code-mixing index, scam-lexicon hits, urgency, the `.xyz` URL.
4. **Phonetic codes** (F6) — why `turant` still matches the lexicon when spelled `turrant`.
5. **Graph features** (F3) — what the sender's fan-out and PageRank say about them.
6. **Discretization** — turning those numbers into `low` / `medium` / `high` for the network.
7. **The BBN** (F1) — the probability chain `0.10 → 0.34 → 0.61 → 0.93`.
8. **The ensemble** (F4) — how the feature model and the BBN are combined.
9. **The counterfactual** (F7) — the smallest change that would have made it SAFE.
10. **Under attack** (F8) — the same message perturbed, with and without our defences.

Each section will carry a small code excerpt and a `file.py:line` pointer.
