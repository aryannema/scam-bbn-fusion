# Viva Q&A

32 questions a professor is likely to ask, with short answers and the file to open.
Questions marked **[hard]** are the ones we would ask if we wanted to catch someone who
had not written the code.

---

## The idea

**1. What problem does this project solve?**
Scam SMS aimed at Indian users is written in Hinglish — Romanized Hindi mixed with
English, with no fixed spelling. Detectors trained on English spam miss it, and
black-box detectors cannot tell a user *why* a message was flagged. We combine wording,
sender behaviour and a Bayesian network that explains itself.

**2. What is a Bayesian Belief Network?**
A directed graph where each node is a variable and each arrow means "this directly
influences that". Each node carries a table of conditional probabilities. Given some
observed variables, you can compute the probability of any other — here, `Fraud`.
*Open `src/f1_bbn.py`, the diagram in the file docstring.*

**3. Why a Bayesian network instead of just a neural network?**
Because the explanation has to be the computation, not a story told afterwards. We add
evidence one piece at a time and record how the probability moves, producing
`0.15 → 0.38 → 0.77`. A neural network can be given a post-hoc explanation (SHAP, LIME),
but that is an approximation of the model, not the model itself.

**4. So is the Bayesian network your best detector?** **[hard]**
No, and we report that honestly. On the leak-safe split the BBN alone scores F1 0.856
while the gradient-boosting feature model scores 0.989. The BBN's value is
explainability and the counterfactuals in F7, not raw accuracy. The ensemble in F4 uses
both.

**5. Where does the project structure live?**
`docs/FUNCTIONALITY_MAP.md` — one row per functionality with its file, entry function
and how to demo it.

---

## Data

**6. Where does your data come from?**
Three sources: 5,171 real English SMS (UCI Spam Collection), 87 real hand-written
Hinglish rows (Hugging Face), and 2,500 Hinglish messages we generate from 9 scam and
7 benign template families. Total 7,758 messages. *Open `src/data_builder.py`.*

**7. Why is so much of it synthetic?**
No public dataset has Hinglish scam text *and* sender-network information. We needed
both to test fusion, so we generate what does not exist and say so plainly.

**8. Why is the sender graph simulated? Isn't that cheating?** **[hard]**
It is the single biggest limitation and it is stated in `docs/PAPER_NOTES.md`. The
graph shows that the *method* works on network structure of this kind; it is not
evidence about effect size on real telecom traffic. We deliberately blurred the
simulation so it cannot be solved trivially — see question 10.

**9. You used a public Hinglish dataset. Did you check it?** **[hard]**
Yes, and that is a finding. `bolewara/hinglish-scam-text-dataset` advertises 3,787 rows,
but 3,700 of them (97.7%) are byte-identical to UCI messages — it is largely a
re-upload. Only 87 rows are genuinely new. We deduplicate and keep those 87. Without
that step the same message would appear in both the train and test split.

**10. How do you know the simulated graph isn't trivially separable?**
Because we measured it and fixed it three times. Message volume alone once reached
F1 = 1.00, so sender activity is now drawn from an overlapping lognormal (volume alone
is AUC 0.70). Account age once had *zero* overlap between the groups, so the ranges now
overlap. And 12% of scammers behave normally while 5% of innocent senders look odd —
exact counts, not dice rolls. *See `tests/test_graph_builder.py`.*

**11. How do you know your results are not leakage?** **[hard]**
Four things, all tested in `tests/test_data_builder.py`:
(a) every sender ring runs exactly one template family, so holding out a ring and
holding out a wording are the same action; (b) 0 of 15 synthetic families, 0 of 53 rings
and 0 of 1,108 senders appear on both sides of the split; (c) no text is duplicated
anywhere, and 0 texts are shared between train and test; (d) feature selection and the
discretization bin edges are fitted on training rows only.

**12. Why split by template family instead of randomly?**
Our synthetic messages come from templates, so two rows of one family share most of
their words. A random split would put near-identical twins in train and test, and the
model would score ~99% by recognising wording it had already read.

**13. Why also report a time-based split?**
Because that is how a detector is really used: trained on the past, run on messages
that have not arrived yet. The two splits expose different weaknesses — see question 24.

**14. Is your dataset reproducible?**
Byte-identical across runs, verified by md5. This required removing Python's `hash()`
from the generator, because string hashing is salted per process, so the "fixed" seed
was producing a different dataset every run.

---

## Features (F2, F6)

**15. What is the code-mixing index (CMI)?**
`100 × (1 − most_common_language / language_bearing_words)`. Zero for a single language,
50 for a perfect mix. It is the standard published measure for code-mixed text.
*Open `code_mixing_index` in `src/f2_hinglish_features.py`.*

**16. Your word lists are tiny. Isn't that a weakness?**
Yes, and we expose it rather than hide it. A word we never listed is tagged `unknown`,
so CMI is a **lower bound** on the real mixing. `unknown_ratio` is published as a
feature (0.65 on benign, 0.37 on scam messages).

**17. Why compute some features on raw text and some on normalized text?** **[hard]**
Lexical features ("does it say kyc?") use normalized text so disguises are undone
first. Structural features ("is it SHOUTING?") must use the RAW text — `normalize()`
lowercases everything, so measuring caps after normalizing would always return 0.

**18. Why phonetic codes instead of a transformer?**
Because the problem is spelling variation, not semantics, and a 7-rule encoder we can
explain line by line beats a model we cannot. `turant`, `turrant`, `turat` and `tarant`
are one word; they all encode to `trt`.

**19. What is the key rule in your phonetic encoder?** **[hard]**
Dropping `n`/`m` before a consonant. Hindi marks a small nasal sound with a dot above
the letter (anusvara) and English letters have no dot, so people simply leave it out.
That rule is what makes `turant` and `turat` the same word, and `band` and `bnd`.

**20. Does the phonetic layer actually help, or is it decoration?**
Measured: scam messages get 1.39 phonetic hits on average against 0.12 for benign. The
feature that matters is `phonetic_only_hits` — hits that exact matching missed — at
0.129 for scam against 0.042 for benign, a 3.1x separation. In F8 it is also the only
defence that survives Hindi respellings.

**21. Why is the minimum phonetic code length 3?** **[hard]**
Measurement. At 2 characters, `lk` (from `link`) also matched `like`, `bd` (`band`)
matched `bad` and `bed`, and `bj` (`bhej`) matched `baje` — 1,865 false hits on benign
messages, and the phonetic-only signal fell to a useless 1.1x. The cost is that short
words (`kyc`→`kc`, `upi`→`up`) leave phonetic matching; F2 still catches those exactly.

**22. Do any innocent words still collide with scam words?**
Yes — `later` and `lottery` both encode to `ltr`. We keep it and document it: about
10.6% of benign messages pick up at least one phonetic hit. The hits are a *count* fed
to a model, not a rule that blocks a message. *See `tests/test_f6_phonetic.py`.*

---

## Graph and fusion (F3, F4)

**23. Which graph features do you use, and why those?**
In/out degree, unique-recipient ratio, PageRank, betweenness, clustering, burstiness
and a forwarding ratio. They come from the graph-fraud survey (Irofti et al.). The idea:
a campaign is a few accounts blasting strangers in a burst; a person texts the same few
people for weeks.

**24. Does fusion actually beat text alone?**
On the leak-safe split, yes: F1 0.963 (text) and 0.949 (graph) against **0.989 fused**.
On the time split, graph-only collapses to 0.527 because a sender never seen in training
has no history, and fused (0.938) matches text alone (0.939). So fusion helps when you
know the sender and does no harm when you do not.

**25. Why does graph-only fail so badly on the time split?** **[hard]**
Cold start. For the time split we compute graph features from training-period traffic
only — otherwise a sender's burstiness would be measured using messages from the future.
Senders who first appear after the cutoff therefore get zeros. That is realistic, and it
is why the text side matters.

**26. What is stacking?**
A small model that learns how much to trust each bigger model. Ours is a logistic
regression over two inputs — the feature model's probability and the BBN's. With two
inputs its coefficients are readable: it learned +9.6 for the feature model and +2.0
for the BBN.

**27. Why out-of-fold predictions?** **[hard]**
This is the easiest thing to get wrong. A model asked about rows it was trained on gives
memorised, over-confident answers; the stack would learn "always believe the feature
model" and fail on real data. So each training row is predicted by a model trained on
the other four folds. *See `test_out_of_fold_is_more_modest_than_asking_the_model_about_its_own_data`.*

**28. Did stacking beat a plain average?**
Yes: 0.9897 against 0.9836 on the leak-safe split. Against the feature model alone the
gain is marginal (0.9897 vs 0.9888), and on the time split stacking is slightly *worse*
(0.9310 vs 0.9384). We report both directions.

---

## The network itself (F1, F5, F7)

**29. Why discretize into low/medium/high? Doesn't that lose information?**
It does, and that is the price of a discrete Bayesian network, which needs each variable
to take one of a few values. Three bins at the 33rd/67th percentile is a compromise: two
throws away the middle ground, five splits the data so thinly the probability tables
become unreliable.

**30. One of your nodes had only two states. What happened?** **[hard]**
`LinkRisk` is a skewed count — most messages have no link — so both the 33rd and 67th
percentile were 0 and the bins collapsed, leaving `low`/`medium` only. Asking the
network about `LinkRisk = high` raised `KeyError`. `bin_edges()` now falls back to the
smallest distinct values when quantiles tie. *Open `bin_edges` in `src/f1_bbn.py`.*

**31. How is the explanation chain built?**
Start from the prior. Rank the evidence by how much each piece moves the probability on
its own, then add them one at a time, recording the probability after each. The list of
moves *is* the explanation. A test asserts the chain's last value equals the reported
score — otherwise it would not be an explanation of the verdict.

**32. What is PatternAnomaly, if you never measure it?**
A hidden middle layer meaning "is the wording anomalous overall?". We have no column for
it, so it is derived by counting how many of its three parents (urgency, credential
request, link risk) are `high`. Keeping it preserves the expert diagram instead of
quietly deleting a node.

**33. What does BIC do in structure learning?**
BIC = how well the structure explains the data, minus a penalty for each extra number
the structure needs. Without the penalty, the best structure would be the one with every
possible edge — perfect on training data, useless on new data.

**34. What did the data-learned structure disagree with you about?** **[hard]**
Both BIC and BDeu **dropped `PatternAnomaly → Fraud`** and wired `UrgencyCue → Fraud`
directly. The learned structures explain held-out data better (log-likelihood −3.92 vs
−4.67) but rank fraud slightly worse (ROC-AUC 0.942 vs 0.965). So our hidden layer costs
likelihood and buys ranking quality — a real, reportable trade-off.

**35. Why blacklist edges out of Fraud? Isn't that rigging the search?**
Search only sees correlation, so it will happily draw `Fraud → account age`, which reads
as "being a fraud makes your account younger". We forbid the *direction*, not which
variables matter, so the search still discovers the structure itself.

**36. What is a counterfactual explanation?**
The smallest change that would have flipped the verdict: *"If the account had looked
trustworthy (HIGH), this message would have been marked SAFE (1.00 → 0.04)."* Based on
Wachter et al. Because our network has only 3^5 = 243 evidence states, we enumerate all
of them and get the genuinely smallest answer rather than an approximation.

**37. Why does minimality matter?**
"If everything had been different it would be safe" is true and useless. We try all
one-variable changes first and only look at pairs if no single change works.

---

## Robustness (F8)

**38. How do you know your detector survives an attacker?**
We attack it ourselves five ways — homoglyphs, leetspeak, zero-width characters, typos
and Hindi respellings — and measure recall under three defences. Undefended recall
against homoglyphs falls from 0.932 to 0.707; with normalization it is 0.888, and with
the phonetic layer 0.937.

**39. Why do you attack only the text model, not the whole system?** **[hard]**
Because text attacks can only change text features. A scammer rewriting a message does
not change who they sent it to, so the graph features are untouched. Mixing them in
would hide the damage behind evidence the attacker cannot reach. That resilience is
itself an argument for fusion, and we say so rather than hiding the text model's weakness.

**40. Normalization lowers your clean recall (0.932 → 0.884). Why keep it?**
An honest trade-off: normalizing discards some signal (unusual characters are themselves
mildly predictive) but buys a large gain under attack. `normalize + phonetic` is best on
clean text *and* on 4 of 5 attacks, which is why it is the configuration we recommend.

---

## Engineering

**41. Why are all your functions so short?**
A hard rule for this project: one job per function, about 25 lines maximum, a plain
docstring saying what/why/example. The point is that any one of us can open a single
file and walk a professor through it.

**42. What does `pytest` cover?**
145 tests, including an app smoke test. Many are regression tests named after the bug
they prevent: `test_pure_numbers_are_never_touched`,
`test_digit_runs_never_collapse`, `test_host_is_parsed_not_substring_matched`,
`test_fan_out_is_informative_but_not_a_giveaway`,
`test_sender_number_is_stable_across_processes`.

**43. Tell me about a bug you found in your own code.** **[hard]**
Three good ones. (a) `collapse_repeats` turned `₹10,000` into `₹10,0` — three zeros are
a repeated character — corrupting every money amount. (b) Link risk was judged with
`"t.co" in text`, which is true for "flipkar**t.co**m", so every legitimate Flipkart
tracking link was labelled a URL shortener; it now parses the host. (c) Only 1 of 5,808
benign messages contained a URL, so "has a link ⇒ scam" was nearly a perfect rule —
we added legitimate links to benign templates, and `P(scam | risky link)` is now 0.757
instead of 1.0.

**44. Why is pgmpy pinned to an exact version?**
Because pgmpy 1.x renamed `BayesianNetwork` to `DiscreteBayesianNetwork`, `fit()` stopped
forwarding estimator options, and `black_list` became `ExpertKnowledge(forbidden_edges=)`.
A floating version would silently break F1 and F5. The differences are recorded in
`docs/API_NOTES.md`.
