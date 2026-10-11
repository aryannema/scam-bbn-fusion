# Functionality Map

**The most important document in this repo.** One row per functionality. If a professor asks
"how does X work, show us the code", find the row, open that one file, walk down it.

Status legend: ✅ done · 🚧 in progress · ⬜ planned

| F# | Name | File | Main function | How to demo it | 30-second explanation | Main library | Status |
|----|------|------|---------------|----------------|-----------------------|--------------|--------|
| — | Central config | [src/config.py](../src/config.py) | *(constants only)* | `python -c "from src.config import RANDOM_SEED; print(RANDOM_SEED)"` | Every path, the random seed and the 0.50 fraud threshold live in one file, so no two files can disagree. | — | ✅ |
| — | Dataset builder | `src/data_builder.py` | `build_dataset()` | `python -m src.data_builder` | Downloads real English spam (UCI) + a real Hinglish seed, generates ~2,500 synthetic Hinglish messages and a simulated sender network, writes one combined CSV. | pandas | ⬜ |
| — | Text normalization | `src/preprocess.py` | `normalize(text)` | `python -m src.preprocess` | Lowercases, strips zero-width characters, undoes leetspeak and homoglyphs. Shared by every other file, and it is defence config (b) in F8. | stdlib | ⬜ |
| **F1** | BBN reasoning + explanation chain | `src/f1_bbn.py` | `score_message(evidence)` | App section 1–2, or `python -m src.f1_bbn` | A hand-drawn Bayesian network (expert knowledge) computes P(Fraud). We then add evidence one variable at a time and print how the probability moves: `0.10 → 0.34 → 0.61 → 0.93`. That chain *is* the explanation. | pgmpy | ⬜ |
| **F2** | Hinglish code-mixed features | `src/f2_hinglish_features.py` | `extract_text_features(texts)` | `python -m src.f2_hinglish_features` | Tags each token as Hindi or English using word lists, then counts code-mixing index, scam-lexicon hits, urgency words, money amounts, URLs and CAPS ratio. | pandas | ⬜ |
| **F3** | Fused feature selection (graph + NLP) | `src/graph_builder.py`, `src/f3_fused_features.py` | `build_fused_features()` | `python -m src.f3_fused_features` | Builds a sender→recipient graph, takes PageRank/degree/burstiness per sender, glues them onto the text features, then asks mutual information which features actually matter. Produces the text-only vs graph-only vs fused table. | networkx, scikit-learn | ⬜ |
| **F4** | Graph + BBN ensemble fusion | `src/f4_ensemble.py` | `train_ensemble()` | `python -m src.f4_ensemble` | A gradient-boosting model reads the fused features, the BBN reads its own evidence, and a small logistic regression learns how much to trust each one (stacking on out-of-fold predictions). | scikit-learn | ⬜ |
| **F5** | Data-driven BBN structure learning | `src/f5_structure_learning.py` | `learn_structure()` | `python -m src.f5_structure_learning` | Instead of us drawing the network, hill-climbing search scores thousands of candidate graphs with BIC and keeps the best. We then compare it against our expert graph: which edges did the data add or remove? | pgmpy | ⬜ |
| **F6** | Phonetic layer for Romanized Hindi | `src/f6_phonetic.py` | `phonetic_code(word)` | App section 5, or `python -m src.f6_phonetic` | `turant`, `turrant`, `turat` are the same Hindi word spelled three ways, so exact matching fails. We collapse sound-alike letters (`v/w`, `ph/f`, `aa/a`) into one code, so all three match one lexicon entry. | stdlib, scikit-learn | ⬜ |
| **F7** | Counterfactual explanations | `src/f7_counterfactual.py` | `find_counterfactuals(evidence)` | App section 3, or `python -m src.f7_counterfactual` | Finds the *smallest* change that would have flipped the verdict: "if network centrality had been LOW, this would have been marked SAFE (0.93 → 0.31)". Brute force over 1, then 2, then 3 variables. | pgmpy | ⬜ |
| **F8** | Adversarial robustness | `src/f8_adversarial.py` | `run_attack_evaluation()` | App section 6, or `python -m src.f8_adversarial` | Attacks our own detector the way a real scammer would — homoglyphs, leetspeak, zero-width characters, respellings — and measures how much recall drops with no defence, with normalization, and with normalization + the phonetic layer. | stdlib, matplotlib | ⬜ |
| — | Demo app | `app/streamlit_app.py` | `main()` | `streamlit run app/streamlit_app.py` | One page with seven sections, each headed with its `F#` tag, so we can point at the exact functionality during the demo. | streamlit | ⬜ |

## Results tables the paper needs

| Table | Produced by | Status |
|-------|-------------|--------|
| text-only vs graph-only vs fused (F1 score) | F3 | ⬜ |
| BBN alone vs feature model vs average vs stacked ensemble | F4 | ⬜ |
| expert structure vs learned structure (edges, log-likelihood, F1) | F5 | ⬜ |
| adversarial recall drop, 3 defence configs | F8 | ⬜ |

*Updated at the end of every phase.*
