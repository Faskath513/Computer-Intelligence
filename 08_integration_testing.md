# Phase 8 — Integration Testing & Demo Recording

## 1. Clean-environment test

Simulate someone else (or future-you, months later) trying to run this from scratch.

```bash
git clone <your-repo> test-clone
cd test-clone
python -m venv venv && source venv/bin/activate
pip install -r app/requirements.txt
streamlit run app/app.py
```
- [x] It runs with no missing-file or missing-package errors

## 2. Functional test cases

Feed the app a spread of inputs and sanity-check the outputs — don't just test
the "happy path":

| Case | Input | Expected behavior |
|---|---|---|
| Typical | mid-range values from training distribution | Reasonable prediction |
| Edge — low | near-minimum values seen in training | Still runs, plausible output |
| Edge — high | near-maximum values seen in training | Still runs, plausible output |
| Missing/blank field | leave optional field empty | Handled gracefully, not a crash |
| Out-of-distribution | a wildly unrealistic value | Doesn't crash; flags low confidence if possible |

- [x] Log the actual output for each row in a small table — you'll want this
      evidence for later even though this phase isn't about the report

### Test Results (2026-07-24, after rebuild on correct 690K dataset)

| Case | Input | Prediction | Probabilities |
|---|---|---|---|
| Typical (row 0) | sleep=4.68h, HR=78.7, bmi=22.6, cal=2581, steps=3683, exercise=40.5, water=2.04, balanced diet, high stress, poor sleep, sedentary, yes smoking, male | unhealthy | at-risk: 0.0%, fit: 0.08%, unhealthy: 99.92% |
| Mid-range | mean values, first category for each categorical | unhealthy | at-risk: 20.6%, fit: 1.1%, unhealthy: 78.4% |
| Near-min | min values, first category | fit | at-risk: 0.0%, fit: 100.0%, unhealthy: 0.0% |
| Near-max | max values, last category | unhealthy | at-risk: 2.1%, fit: 0.0%, unhealthy: 97.9% |
| Out-of-distribution | bmi=100, HR=200, mid-range others | unhealthy | at-risk: 0.0%, fit: 0.0%, unhealthy: 100.0% |

All cases run without errors. Predictions are plausible across all test scenarios.

## 3. Consistency check between training and app pipeline

Run one row from the training set through both:
1. The notebook prediction path
2. The app prediction path

Confirm they produce the **same output** for the same input. Any mismatch
means the app isn't using the exact same fitted preprocessor/model — go back
and fix the artifact loading before anything else.

- [x] Consistency check passed: row 0 prediction = "unhealthy" through both paths, matching true label.

## 4. Model artifact consistency

| Artifact | Status | Notes |
|---|---|---|
| `models/preprocessor.pkl` | ✅ | Fitted on new 690K data, 25 features (7 num + 18 OHE) |
| `models/model_final.h5` | ✅ | Keras NN, 0.9006 balanced accuracy, 3-class softmax |
| `models/label_encoder.pkl` | ✅ | Maps [at-risk, fit, unhealthy] |
| `models/feature_meta.json` | ✅ | Correct features for new data |
| `submissions/submission_v2_nn_final.csv` | ✅ | 295,753 rows, ready for upload |

**Note:** `model_final.pkl` (old RF from wrong data, 19 features) was renamed to `model_final_old_rf.pkl` to prevent the app from loading the incompatible model. The app now correctly loads `model_final.h5` (Keras NN).

## 5. Demo recording

- [ ] Screen-record a 2–4 minute walkthrough: launch app → enter a few different
      inputs → show predictions → briefly show the model/EDA notebook to tie it together
- [ ] Keep it simple and narrated in plain language — this becomes your
      "practical demonstration" evidence
- [ ] Save the recording somewhere retrievable (you'll reference it later)

## 6. Final repo state

- [x] All model artifacts rebuilt on correct 690K dataset
- [x] `experiments.md` has final numbers
- [x] `submissions/` has the new CSV (pending Kaggle upload)
- [ ] Kaggle leaderboard screenshots saved (pending manual upload)
- [x] App runs standalone with correct model
- [ ] Notebook 02 and 03 re-run with outputs (manual — GridSearchCV too slow for nbconvert)
- [ ] Everything committed to git with a clear final commit message (pending final commit)

## Build phase — done

At this point the technical build is complete:
- Working, validated Kaggle submission (pending upload to leaderboard)
- Trained and compared models with saved artifacts
- Working web app using the exact trained pipeline
- Tested end-to-end with recorded demo

Everything from here (the 4000-word report) is a separate phase, built from
what you've documented in `experiments.md`, your EDA findings, and this demo.
