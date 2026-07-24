# Phase 2 — Environment & Repo Setup

## Folder structure

Create this exact structure — later files assume it exists:

```
project-root/
├── data/
│   ├── raw/              # original Kaggle files, untouched, gitignored
│   └── processed/        # cleaned/feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_final_model.ipynb
├── src/
│   ├── data_prep.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   └── (trained model artifacts go here, gitignored if large)
├── app/
│   ├── app.py
│   └── requirements.txt
├── submissions/
│   └── (Kaggle submission CSVs)
├── .gitignore
└── README.md
```

## Commands

```bash
mkdir -p project-root/{data/raw,data/processed,notebooks,src,models,app,submissions}
cd project-root
git init

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install pandas numpy scikit-learn matplotlib seaborn jupyter \
            tensorflow streamlit kaggle joblib xgboost
pip freeze > requirements.txt
```

## Kaggle API setup (to download data + submit from CLI)

1. Kaggle account → Settings → "Create New API Token" → downloads `kaggle.json`
2. Place it:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
3. `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)
4. Test:
   ```bash
   kaggle competitions download -c <competition-slug> -p data/raw
   ```

## `.gitignore` (minimum)

```
venv/
data/raw/
data/processed/
models/*.pkl
models/*.h5
__pycache__/
.ipynb_checkpoints/
```

## Sanity check before moving on

- [ ] `python -c "import pandas, sklearn, tensorflow, streamlit; print('ok')"` runs clean
- [ ] `data/raw/` contains `train.csv`, `test.csv`, `sample_submission.csv`
- [ ] Repo committed with initial structure

Next: `03_eda.md`
