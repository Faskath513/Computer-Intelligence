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

## Decision record — fill this in and don't revisit it

```
Competition name:      ____________________
Kaggle URL:             ____________________
Task type:              classification / regression / other: ______
Target column:          ____________________
Evaluation metric:      ____________________
Competition deadline:   ____________________
Your submission deadline: ____________________
Eligible? (Y/N + why):  ____________________
```

## Immediate next actions

- [ ] Create a free Kaggle account if you don't have one
- [ ] Join the competition ("Enroll") — read and accept the rules
- [ ] Download `train.csv`, `test.csv`, `sample_submission.csv`
- [ ] Skim the competition's "Discussion" and public notebooks tab — see what
      others are already trying (don't copy — just get oriented)

Once this file's decision record is filled in, move to `02_environment_setup.md`.
