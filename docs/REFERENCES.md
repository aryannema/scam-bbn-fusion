# References

Every link below was fetched and confirmed to resolve while writing this document
(11 October 2026). arXiv titles are quoted as the arXiv API returns them, not as we
remembered them. Where our original reading list had a wrong detail, the correction is
noted — the pgmpy paper in particular is JMLR 25(265), paper `23-0487`.

## Bayesian networks, and the library we use

1. **Ankan, A. and Textor, J. (2024).** "pgmpy: A Python Toolkit for Bayesian Networks."
   *Journal of Machine Learning Research* 25(265), 1–8. (MLOSS track.)
   <https://www.jmlr.org/papers/v25/23-0487.html> — the library behind F1, F5 and F7.
2. **Koller, D. and Friedman, N. (2009).** *Probabilistic Graphical Models: Principles
   and Techniques.* MIT Press. — structure learning and the BIC/BDeu scores used in F5.
3. **Russell, S. and Norvig, P.** *Artificial Intelligence: A Modern Approach*, the
   chapter on Bayesian networks. — the course text; our variable-elimination inference.
4. **Lappas, T. and Karampelas, P.** "Enhancement of Phishing Email Detection with
   Bayesian Networks." *European Conference on Cyber Warfare and Security (ECCWS)*,
   academic-conferences.org. — closest prior work: a Bayesian network for phishing.
5. **"Bayesian Modeling of Uncertainty in Anti-Phishing Training: A Serious Game for
   Adversarial Email Crafting."** *ECCWS.* — Bayesian treatment of adversarial email.

> Note on 4 and 5: both are ECCWS proceedings papers behind the
> academic-conferences.org paywall. We could confirm the publisher and venue but not
> fetch the full text, so we cite them as related work rather than quoting results.

## Graph-based fraud detection

6. **Cheng, D. et al. (2024).** "Graph Neural Networks for Financial Fraud Detection:
   A Review." arXiv:2411.05815. <https://arxiv.org/abs/2411.05815>
7. **Irofti, P. et al. (2019).** "Quick survey of graph-based fraud detection methods."
   arXiv:1910.11299. <https://arxiv.org/abs/1910.11299> — the feature families in
   `graph_builder.py` (degree, PageRank, clustering, burstiness) come from this survey.
8. **AI4Risk, "awesome-graph-based-fraud-detection."**
   <https://github.com/AI4Risk/awesome-graph-based-fraud-detection>

## Counterfactual explanation

9. **Wachter, S., Mittelstadt, B. and Russell, C. (2017).** "Counterfactual
   Explanations without Opening the Black Box: Automated Decisions and the GDPR."
   arXiv:1711.00399. <https://arxiv.org/abs/1711.00399> — the "smallest change to the
   world" idea that F7 implements. Because our network has only 3^5 = 243 possible
   evidence states, we can enumerate exhaustively where they had to approximate.

## Phonetics for code-mixed and Romanized Hindi text

10. **"Elevating Code-mixed Text Handling through Auditory Information of Words."**
    arXiv:2310.18155. <https://arxiv.org/abs/2310.18155> — SOUNDEX for Hinglish; the
    direct inspiration for F6.
11. **"Theedhum Nandrum@Dravidian-CodeMix-FIRE2020: A Sentiment Polarity Classifier
    for YouTube Comments..."** arXiv:2010.03189. <https://arxiv.org/abs/2010.03189> —
    language-specific Soundex for Indian code-mixed text.
12. **Mandal, S. and Nanmaran, K. (2018).** "Normalization of Transliterated Words in
    Code-Mixed Data Using Seq2Seq Model & Levenshtein Distance." arXiv:1805.08701.
    <https://arxiv.org/abs/1805.08701> — the problem F6 solves with rules instead.
13. **Proceedings of the Fifth Workshop on Computational Approaches to Linguistic
    Code-Switching (CALCS 2021).** <https://aclanthology.org/volumes/2021.calcs-1/> —
    normalization and back-transliteration for code-switched data.
14. **Kant, S. et al.** "SoundexHindi: A Phonetic Matching Algorithm for Hindi Written
    in English." — a published Soundex variant for exactly our setting. We wrote our
    own 7-rule encoder so that every rule is explainable, and cite this as the
    comparable published approach.
15. **libindic soundex.** <https://github.com/libindic/soundex> — an existing library
    for Indic phonetic matching. Named in F6 as the thing to compare against; we
    deliberately do **not** depend on it.

## SMS scams and adversarial robustness

16. **"An Empirical Analysis of SMS Scam Detection Systems."** arXiv:2210.10451.
    <https://arxiv.org/abs/2210.10451>
17. **"Every Character Counts: From Vulnerability to Defense in Phishing Detection."**
    arXiv:2509.20589. <https://arxiv.org/abs/2509.20589> — character-level attacks of
    the kind F8 implements.
18. **"Robust ML-based Detection of Conventional, LLM-Generated, and Adversarial
    Phishing Emails."** arXiv:2510.11915. <https://arxiv.org/abs/2510.11915>
19. **"A Robust and Explainable Transformer-Based Framework for Phishing Email
    Detection."** arXiv:2511.12085. <https://arxiv.org/abs/2511.12085> — the
    transformer alternative to our approach; useful for the "why not a transformer?"
    question.
20. **Google RETVec.** <https://github.com/google-research/retvec> — a vectorizer built
    to resist character perturbation; the industrial version of what F6 does by hand.

## Datasets

21. **UCI SMS Spam Collection.**
    <https://archive.ics.uci.edu/dataset/228/sms+spam+collection> — 5,574 real English
    SMS (747 spam / 4,827 ham). We keep 5,171 after removing 403 internal duplicates.
22. **`bolewara/hinglish-scam-text-dataset` (Hugging Face).**
    <https://huggingface.co/datasets/bolewara/hinglish-scam-text-dataset> — advertises
    3,787 rows. **We measured that 3,700 of them (97.7%) are byte-identical to UCI
    messages**; only 87 rows are new. We use those 87 and discard the rest. Anyone
    combining these two sources without checking will have duplicate rows spanning
    their train/test split. See `docs/PAPER_NOTES.md`.
23. **IEEE DataPort, "Spam SMS in Hindi Language."** Listed in our original reading
    list as a possible Hindi source. We did **not** use it: it needs an IEEE
    subscription to download, so we could not verify its contents, and we would rather
    generate data we can describe exactly than cite data we have not seen.

## Software

- pandas, NumPy, scikit-learn, networkx, Matplotlib, Streamlit — see
  `requirements.txt` for the exact pinned versions.
- networkx documentation: <https://networkx.org/>
- pgmpy documentation: <https://pgmpy.org/>
