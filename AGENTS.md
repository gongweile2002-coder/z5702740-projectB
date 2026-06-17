# AGENTS.md - FINS5545 Part B Working Instructions

This project builds systematic multi-asset funds and a Streamlit dashboard from
the hosted FINS5545 project data. Raw parquet files must be loaded only through
`src/data_access.py`; raw data files should not be committed. Derived app inputs
under `results/` are expected to be committed because the deployed app reads
precomputed outputs.

Project rules for AI/code assistance:

- Keep portfolio backtests walk-forward and out-of-sample. Weights may use only
  information available on or before the rebalance date and should apply from the
  next trading day.
- Compute crypto returns on the crypto calendar before aligning them to the
  equity trading calendar for combined funds.
- Deduplicate headlines by `ticker`, `date`, and `title`. Sentiment signals must
  be lagged before they affect any portfolio weights.
- The Streamlit app must not import NLTK or recompute sentiment/backtests. It
  should load `results/data/*.csv` and `results/tables/*.csv`.
- Prefer clear pandas/numpy code, labelled figures, and reproducible scripts.
  Required outputs use the filenames in `PROJECT_BRIEF.md`.
- Treat AI-written interpretation as a draft only. The final report wording and
  investment interpretation should be reviewed and rewritten by the student.

Verification steps:

1. Run `python scripts/run_part_b.py` from the project root.
2. Run `python scripts/check_handin.py`.
3. Run `streamlit run streamlit_app.py` and check that the dashboard loads.
4. Review `report/report.pdf` and edit the wording before submission.
