# FINS5545 Project Part B Report Draft

This PDF was generated from `scripts/run_part_b.py`. Review the wording, replace the placeholder zID folder name, and re-export from Word if required by the course.

- The backtest is monthly, long-only and walk-forward with weights estimated from trailing data only.
- The sentiment model uses de-duplicated headlines aligned to the next trading day, smoothed five days and lagged one day.
- The deployed app reads precomputed CSV artifacts and does not import NLTK or recompute backtests.