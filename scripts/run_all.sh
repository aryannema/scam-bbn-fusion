#!/usr/bin/env bash
#
# Runs every stage of the project end to end, in dependency order.
#
# WHAT: builds the dataset, then runs all eight functionalities, saving every results
#       table and chart into outputs/.
# WHY:  one command reproduces every number in the report from a clean checkout.
# HOW:  bash scripts/run_all.sh
#
# set -e  : stop at the first failure instead of carrying on with broken data.
# set -u  : treat an unset variable as an error.
# pipefail: a failure anywhere in a pipeline fails the whole pipeline.
set -euo pipefail

# Always run from the project root, whichever directory the user invoked us from.
cd "$(dirname "$0")/.."

# Prefer the project's own virtual environment if it exists, so the pinned versions
# in requirements.txt are the ones actually used.

if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON=".venv/Scripts/python.exe"
else
  PYTHON="python3"
  echo "NOTE: .venv not found, using system python3. See docs/SETUP.md."
fi


banner() {
  echo
  echo "================================================================"
  echo "  $1"
  echo "================================================================"
}

banner "STAGE 0  Dataset: download, generate, simulate the network"
$PYTHON -m src.data_builder

banner "STAGE 1  Shared text normalization (used by everything)"
$PYTHON -m src.preprocess

banner "STAGE 2  F2  Hinglish code-mixed features"
$PYTHON -m src.f2_hinglish_features

banner "STAGE 3  F6  Phonetic layer for Romanized Hindi"
$PYTHON -m src.f6_phonetic

banner "STAGE 4  Sender graph"
$PYTHON -m src.graph_builder

banner "STAGE 5  F3  Fused feature selection (KEY RESULT: text vs graph vs fused)"
$PYTHON -m src.f3_fused_features

banner "STAGE 6  F1  Expert Bayesian network and explanation chain"
$PYTHON -m src.f1_bbn

banner "STAGE 7  F5  Structure learning (expert vs learned)"
$PYTHON -m src.f5_structure_learning

banner "STAGE 8  F4  Ensemble fusion (MAIN RESULT)"
$PYTHON -m src.f4_ensemble

banner "STAGE 9  F7  Counterfactual explanations"
$PYTHON -m src.f7_counterfactual

banner "STAGE 10 F8  Adversarial robustness"
$PYTHON -m src.f8_adversarial

banner "STAGE 11 Tests"
$PYTHON -m pytest -q

banner "DONE"
echo "Result tables and charts are in outputs/"
echo "Start the demo app with:  streamlit run app/streamlit_app.py"
