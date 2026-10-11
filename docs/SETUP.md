# Setup

How to get this project running from a fresh clone. Takes about 5 minutes.

## 1. Requirements

- **Python 3.10 – 3.14.** We developed on Python 3.13.
  (`pgmpy`, our Bayesian-network library, declares `>=3.10,<3.15`.)
- About 600 MB of disk space for the virtual environment.
- Internet access the first time, to download the two public datasets.

Check your version:

```bash
python3 --version
```

## 2. Create the virtual environment

A *virtual environment* is a private folder of Python packages that belongs to this
project only. It keeps our pinned versions from clashing with anything else on the machine.

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt now starts with `(.venv)`. That means it worked.

> **Never commit `.venv/`.** It is hundreds of megabytes of machine-specific binaries and
> it is already listed in `.gitignore`. The thing we *do* commit is `requirements.txt`,
> which is the recipe to rebuild it.

## 3. Install the dependencies

```bash
pip install -r requirements.txt
```

Every version is pinned exactly, so you get the same versions we tested against.

Verify:

```bash
python -c "import pandas, numpy, sklearn, networkx, pgmpy, matplotlib, streamlit; print('ok')"
```

## 4. Run the tests

```bash
pytest
```

## 5. Build the dataset and run everything

```bash
bash scripts/run_all.sh
```

This downloads the public datasets into `data/raw/`, generates the synthetic Hinglish
messages and sender network into `data/processed/`, then runs all eight functionalities
end to end.

## 6. Run the demo app

```bash
streamlit run app/streamlit_app.py
```

Then open the URL it prints (usually <http://localhost:8501>).

## Troubleshooting

**A package fails to build a wheel on Python 3.13.**
Use Python 3.12 instead — it is the most widely supported version for this stack:

```bash
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**`ModuleNotFoundError: No module named 'src'`.**
Run commands from the repository root (the folder containing `pytest.ini`), not from
inside `src/`. `pytest.ini` sets `pythonpath = .`, which only applies from the root.

**`streamlit: command not found`.**
Your virtual environment is not active. Re-run the `activate` command from step 2.
