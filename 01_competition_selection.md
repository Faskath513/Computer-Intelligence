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
Competition name:      Predicting Student Health Risk
Kaggle URL:             kaggle.com/competitions/playground-series-s6e7
Series:                 Kaggle Playground Series — Season 6, Episode 7
Task type:              Multi-class classification (3 classes)
Target column:          health_condition
Class labels:           at-risk / unhealthy / fit
Evaluation metric:      Balanced accuracy
Competition start:      July 1, 2026
Competition deadline:   July 31, 2026 (11:59 PM UTC)
Your submission deadline: ____________________
Eligible? (Y/N + why):  Y — active competition, fixed start/close dates
                         (not a rolling leaderboard), tabular data, fits
                         classification + ensemble/NN techniques allowed
                         by the brief.
```

**Important modeling implication:** the metric is **balanced accuracy**, not
plain accuracy. This means the classes may be imbalanced — check class
distribution of `health_condition` early in EDA (Phase 3), and if imbalanced,
plan to use `class_weight="balanced"` in sklearn models and weighted loss (or
oversampling) in the neural net, so your validation metric actually matches
what Kaggle scores you on.

## Immediate next actions

- [x] Competition chosen: Playground Series S6E7 — Predicting Student Health Risk
- [x] Target column, metric, and deadline confirmed
- [x] Create a free Kaggle account if you don't have one
- [x] Join the competition ("Enroll") — read and accept the rules
- [x] Download `train.csv`, `test.csv`, `sample_submission.csv`
- [x] Skim the competition's "Discussion" and public notebooks tab — see what
      others are already trying (don't copy — just get oriented)

Decision record is fully filled in — move to `02_environment_setup.md`.