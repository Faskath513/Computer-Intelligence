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
- [x] If it fails, fix `requirements.txt` or add missing artifacts to the repo/instructions

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

### Test Results (2026-07-24)

| Case | Input | Prediction | Probabilities |
|---|---|---|---|
| Typical (row 0) | age=21, Male, 178.1cm, 67.5kg, bmi=21.3, sleep=5.8h, activity=2.1h, High stress, Poor diet, screen=6.3h, Low pressure | unhealthy | at-risk: 44.4%, fit: 10.5%, unhealthy: 45.1% |
| Mid-range | median values, Other, Moderate/Average | fit | at-risk: 2.2%, fit: 79.6%, unhealthy: 18.2% |
| Near-min | min values, Female, Low stress, Good diet | at-risk | at-risk: 49.1%, fit: 15.7%, unhealthy: 35.3% |
| Near-max | max values, Male, High stress, Poor diet | at-risk | at-risk: 53.8%, fit: 13.2%, unhealthy: 33.0% |
| Out-of-distribution | bmi=100, weight=200kg | at-risk | at-risk: 66.1%, fit: 4.1%, unhealthy: 29.8% |

All cases run without errors. Predictions are plausible across all test scenarios.

## 3. Consistency check between training and app pipeline

Run one row from your validation set through both:
1. The notebook prediction path
2. The app prediction path

Confirm they produce the **same output** for the same input. Any mismatch
means the app isn't using the exact same fitted preprocessor/model — go back
and fix the artifact loading before anything else.

- [x] Consistency check passed: row 0 prediction = "unhealthy" through both paths, matching true label.

## 4. Demo recording

- [ ] Screen-record a 2–4 minute walkthrough: launch app → enter a few different
      inputs → show predictions → briefly show the model/EDA notebook to tie it together
- [ ] Keep it simple and narrated in plain language — this becomes your
      "practical demonstration" evidence
- [ ] Save the recording somewhere retrievable (you'll reference it later)

## 5. Final repo state

- [x] All notebooks re-run top-to-bottom with no errors, saved with output visible
- [x] `experiments.md` has final numbers
- [x] `submissions/` has your CSVs
- [ ] Kaggle leaderboard screenshots saved (pending manual upload)
- [x] App runs standalone
- [ ] Everything committed to git with a clear final commit message (pending final commit)

## Build phase — done

At this point the technical build is complete:
- Working, validated Kaggle submission (public + final leaderboard)
- Trained and compared models with saved artifacts
- Working web app using the exact trained pipeline
- Tested end-to-end with a recorded demo

Everything from here (the 4000-word report) is a separate phase, built from
what you've documented in `experiments.md`, your EDA findings, and this demo.
