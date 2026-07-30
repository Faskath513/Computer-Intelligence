# Phase 1 — Competition Selection

## Eligibility rules (from the brief — non-negotiable)

- Must be **active**, or **concluded within 2 months** of your submission deadline.
- Must have **clearly defined start/end dates** — no indefinite rolling-leaderboard
  playground competitions (Titanic, House Prices Advanced Regression, etc. are OUT).
- Must fit at least one allowed technique: Regression, SVM, Neural Networks,
  Clustering, Ensemble methods.

## How to search

1. Go to `kaggle.com/competitions`
2. Filter: **Active only**
3. Sort by "Recently Launched" to find ones with real end dates
4. Open each candidate's "Overview" tab and confirm it explicitly states a
   deadline (not "ongoing")

## What makes a competition good for THIS project (not just eligible)

Prefer:
- Tabular data (CSV) over image/audio/text-heavy — much faster EDA → model → app pipeline
- Single, clear target column
- Dataset under ~500MB (trains fine on a laptop or free Colab)
- A supported metric (accuracy, F1, AUC, RMSE — anything sklearn/Keras report natively)

Avoid unless you have strong prior experience:
- Multi-modal data (image + text + tabular combined)
- Very large datasets (multi-GB) — will blow your time budget on infra, not modeling
- Competitions with heavy custom evaluation code

## Decision record — LOCKED & CONFIRMED

```
Competition name:      Forecasting Sticker Sales
Kaggle URL:             kaggle.com/competitions/playground-series-s5e1
Series:                 Kaggle Playground Series — Season 5, Episode 1
Task type:              Time series regression
Target column:          num_sold
Evaluation metric:      Mean Absolute Percentage Error (MAPE)
Competition start:      January 1, 2025
Competition deadline:   February 1, 2025 (11:59 PM UTC)
Your submission deadline: ____________________
Eligible? (Y/N + why):  Y — fixed start/close dates, tabular time series,
                         fits regression + ensemble techniques allowed
                         by the brief. Late submission accepted by university.
```

**Important modeling implication:** the metric is **MAPE**, which penalizes
errors on low-volume series more heavily. Pay special attention to countries
like Kenya where sales are very low (5-18 units) — small absolute errors = large
MAPE contributions.

## Immediate next actions

- [x] Competition chosen: Playground Series S5E1 — Forecasting Sticker Sales
- [x] Target column, metric, and deadline confirmed
- [x] Create a free Kaggle account if you don't have one
- [x] Join the competition ("Enroll") — read and accept the rules
- [x] Download `train.csv`, `test.csv`, `sample_submission.csv`
- [x] Skim the competition's "Discussion" and public notebooks tab — see what
      others are already trying (don't copy — just get oriented)

Decision record is fully filled in — move to `02_environment_setup.md`.