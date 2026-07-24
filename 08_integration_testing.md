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
- [ ] It runs with no missing-file or missing-package errors
- [ ] If it fails, fix `requirements.txt` or add missing artifacts to the repo/instructions

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

- [ ] Log the actual output for each row in a small table — you'll want this
      evidence for later even though this phase isn't about the report

## 3. Consistency check between training and app pipeline

Run one row from your validation set through both:
1. The notebook prediction path
2. The app prediction path

Confirm they produce the **same output** for the same input. Any mismatch
means the app isn't using the exact same fitted preprocessor/model — go back
and fix the artifact loading before anything else.

## 4. Demo recording

- [ ] Screen-record a 2–4 minute walkthrough: launch app → enter a few different
      inputs → show predictions → briefly show the model/EDA notebook to tie it together
- [ ] Keep it simple and narrated in plain language — this becomes your
      "practical demonstration" evidence
- [ ] Save the recording somewhere retrievable (you'll reference it later)

## 5. Final repo state

- [ ] All notebooks re-run top-to-bottom with no errors, saved with output visible
- [ ] `experiments.md` has final numbers
- [ ] `submissions/` has your CSVs
- [ ] Kaggle leaderboard screenshots saved
- [ ] App runs standalone
- [ ] Everything committed to git with a clear final commit message

## Build phase — done

At this point the technical build is complete:
- Working, validated Kaggle submission (public + final leaderboard)
- Trained and compared models with saved artifacts
- Working web app using the exact trained pipeline
- Tested end-to-end with a recorded demo

Everything from here (the 4000-word report) is a separate phase, built from
what you've documented in `experiments.md`, your EDA findings, and this demo.
