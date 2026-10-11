# Functionality Map

**The most important document in this repo.** One row per functionality. If a professor
asks "how does X work, show us the code", find the row, open that one file, walk down it.

All eight functionalities are complete, tested, and demoable from both
`scripts/run_all.sh` and the Streamlit app.

| F# | Name | File | Main function | How to demo it | 30-second explanation | Main library | Status |
|----|------|------|---------------|----------------|-----------------------|--------------|--------|
| **F1** | BBN reasoning + explanation chain | [src/f1_bbn.py](../src/f1_bbn.py) | `score_message(engine, evidence)` | App §1–2, or `python -m src.f1_bbn` | A hand-drawn Bayesian network computes P(Fraud) from 5 clues. We then add the clues one at a time and record how the probability moves: `0.16 → 0.28 → 0.73 → 0.50`. That chain **is** the explanation — it is the computation, not a story about it. | pgmpy | ✅ |
| **F2** | Hinglish code-mixed features | [src/f2_hinglish_features.py](../src/f2_hinglish_features.py) | `extract_text_features(texts)` | App §5, or `python -m src.f2_hinglish_features` | Tags every word as Hindi, English, brand-neutral or unknown from hand-written lists, then builds **19 features**: code-mixing index, script switches, scam-lexicon hits, urgency and credential words, money amounts, link risk, digit and CAPS ratios. | pandas | ✅ |
| **F3** | Fused feature selection (graph + NLP) | [src/graph_builder.py](../src/graph_builder.py), [src/f3_fused_features.py](../src/f3_fused_features.py) | `build_fused_table()`, `run_comparison()` | App §4 and §7, or `python -m src.f3_fused_features` | Builds a sender→recipient graph, takes degree / PageRank / clustering / burstiness / forwarding per sender, glues them onto the 21 text features, then asks mutual information which of the 34 actually matter. Produces the text-vs-graph-vs-fused table. | networkx, scikit-learn | ✅ |
| **F4** | Graph + BBN ensemble fusion | [src/f4_ensemble.py](../src/f4_ensemble.py) | `train_ensemble()` | App §1 and §7, or `python -m src.f4_ensemble` | Gradient boosting reads all 34 features; the BBN reads its 5 clues; a small logistic regression learns how much to trust each, trained on **out-of-fold** predictions so it never sees memorised answers. | scikit-learn | ✅ |
| **F5** | Data-driven BBN structure learning | [src/f5_structure_learning.py](../src/f5_structure_learning.py) | `learn_structure()`, `run_comparison()` | App §7, or `python -m src.f5_structure_learning` | Instead of us drawing the network, hill-climbing search scores thousands of candidate graphs with BIC (and BDeu) and keeps the best, with edges out of `Fraud` forbidden. We then compare it to our expert graph: which edges did the data add or remove? | pgmpy | ✅ |
| **F6** | Phonetic layer for Romanized Hindi | [src/f6_phonetic.py](../src/f6_phonetic.py) | `phonetic_code(word)` | App §5, or `python -m src.f6_phonetic` | `turant`, `turrant`, `turat`, `tarant` are one Hindi word spelled four ways. Seven rules collapse sound-alike spellings — including Hindi's often-unwritten nasal — so all four encode to `trt` and match one lexicon entry. Also emits a bag-of-phonetic-codes matrix. | stdlib, scikit-learn | ✅ |
| **F7** | Counterfactual explanations | [src/f7_counterfactual.py](../src/f7_counterfactual.py) | `find_counterfactuals(engine, evidence)` | App §3, or `python -m src.f7_counterfactual` | Finds the **smallest** change that would flip the verdict: *"If the account had looked trustworthy, this would have been SAFE (1.00 → 0.04)."* With only 3⁵ = 243 possible evidence states we enumerate all of them, so the answer is genuinely minimal. | pgmpy | ✅ |
| **F8** | Adversarial robustness | [src/f8_adversarial.py](../src/f8_adversarial.py) | `run_attack_evaluation()` | App §6, or `python -m src.f8_adversarial` | Attacks our own detector five ways — homoglyphs, leetspeak, zero-width characters, typos, Hindi respellings — and measures recall under three defences: none, normalize, normalize + phonetic. | stdlib, matplotlib | ✅ |

## Supporting files (not numbered functionalities)

| Name | File | Main function | How to demo it | What it does | Status |
|------|------|---------------|----------------|--------------|--------|
| Central config | [src/config.py](../src/config.py) | *(constants only)* | `python -c "from src.config import RANDOM_SEED; print(RANDOM_SEED)"` | Every path, the seed, the 0.50 fraud threshold and every tuning constant in one file, so no two files can disagree. | ✅ |
| Dataset builder | [src/data_builder.py](../src/data_builder.py) | `build_dataset()` | `python -m src.data_builder` | Downloads real English spam (UCI, 5,171 kept) + the real Hinglish seed (87 rows, deduplicated against UCI), generates 2,500 synthetic Hinglish messages and a simulated network of 1,108 senders. Writes one CSV of 7,758 messages plus both leak-safe splits. | ✅ |
| Text normalization | [src/preprocess.py](../src/preprocess.py) | `normalize(text)` | `python -m src.preprocess` | Folds look-alike letters, deletes invisible characters, lowercases, undoes leetspeak (never inside pure numbers, so amounts survive) and collapses padded letter runs. Shared by everything; it is defence (b) in F8. | ✅ |
| Demo app | [app/streamlit_app.py](../app/streamlit_app.py) | `main()` | `streamlit run app/streamlit_app.py` | One page, seven sections, each headed with its F-number. | ✅ |
| Run everything | [scripts/run_all.sh](../scripts/run_all.sh) | — | `bash scripts/run_all.sh` | Rebuilds the dataset and runs all eight functionalities plus the tests, saving every table and chart into `outputs/`. | ✅ |

## Results tables the paper needs

| Table | Produced by | File in `outputs/` | Headline |
|-------|-------------|--------------------|----------|
| text-only vs graph-only vs fused | F3 | `f3_comparison_split_leaksafe.csv`, `f3_comparison_split_time.csv` | fused F1 **0.989** vs text 0.963, graph 0.949 | ✅ |
| BBN vs feature model vs average vs stacked | F4 | `f4_ensemble_split_leaksafe.csv`, `f4_ensemble_split_time.csv` | stacked **0.9897**, average 0.9836, BBN 0.8555 | ✅ |
| expert vs learned structure | F5 | `f5_structure_comparison.csv` + 3 PNGs | learned wins log-likelihood (−3.92 vs −4.67), expert wins AUC (0.965 vs 0.942) | ✅ |
| adversarial, 3 defences × 5 attacks | F8 | `f8_adversarial.csv`, `f8_adversarial.png` | homoglyphs: 0.707 → 0.888 → **0.937** | ✅ |

## Tests

One test file per functionality, 145 tests total, all passing.

| Functionality | Test file |
|---|---|
| preprocess | [tests/test_preprocess.py](../tests/test_preprocess.py) |
| data builder | [tests/test_data_builder.py](../tests/test_data_builder.py) |
| F2 | [tests/test_f2_hinglish_features.py](../tests/test_f2_hinglish_features.py) |
| F6 | [tests/test_f6_phonetic.py](../tests/test_f6_phonetic.py) |
| graph builder | [tests/test_graph_builder.py](../tests/test_graph_builder.py) |
| F3 | [tests/test_f3_fused_features.py](../tests/test_f3_fused_features.py) |
| F1 | [tests/test_f1_bbn.py](../tests/test_f1_bbn.py) |
| F5 | [tests/test_f5_structure_learning.py](../tests/test_f5_structure_learning.py) |
| F4 | [tests/test_f4_ensemble.py](../tests/test_f4_ensemble.py) |
| F7 | [tests/test_f7_counterfactual.py](../tests/test_f7_counterfactual.py) |
| F8 | [tests/test_f8_adversarial.py](../tests/test_f8_adversarial.py) |
| demo app | [tests/test_streamlit_app.py](../tests/test_streamlit_app.py) |
