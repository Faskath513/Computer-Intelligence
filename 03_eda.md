# Phase 3 — Exploratory Data Analysis

Notebook: `notebooks/01_eda.ipynb`

Goal: understand the data well enough that every later modeling decision can
be justified by something you found here.

## Step-by-step

### 1. Load & first look
```python
import pandas as pd

train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")

train.shape, test.shape
train.info()
train.head()
```

### 2. Missing values
```python
train.isnull().sum().sort_values(ascending=False)
(train.isnull().mean() * 100).round(1)  # % missing per column
```
- [ ] Decide per column: drop / impute mean-median-mode / impute with a flag /
      leave as "missing" category

### 3. Target distribution
```python
train["<target_col>"].value_counts(normalize=True)   # classification
# or
train["<target_col>"].describe()                     # regression
train["<target_col>"].hist(bins=50)
```
- [ ] Note class imbalance (classification) or skew (regression) — this drives
      whether you need `class_weight`, resampling, or a log-transform later

### 4. Univariate exploration
```python
import matplotlib.pyplot as plt
import seaborn as sns

num_cols = train.select_dtypes(include="number").columns
train[num_cols].hist(figsize=(15, 10), bins=30)
plt.tight_layout()
```
- [ ] Flag features with extreme skew or obvious outliers

### 5. Correlation / relationship to target
```python
corr = train[num_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
```
```python
train.corr(numeric_only=True)["<target_col>"].sort_values(ascending=False)
```

### 6. Categorical features
```python
cat_cols = train.select_dtypes(include="object").columns
for c in cat_cols:
    print(c, train[c].nunique())
```
- [ ] High-cardinality columns → note for target/frequency encoding instead of one-hot

### 7. Outlier detection
```python
Q1 = train[num_cols].quantile(0.25)
Q3 = train[num_cols].quantile(0.75)
IQR = Q3 - Q1
outliers = ((train[num_cols] < (Q1 - 1.5*IQR)) | (train[num_cols] > (Q3 + 1.5*IQR))).sum()
outliers.sort_values(ascending=False)
```

## Deliverable checklist

- [ ] Notebook runs top-to-bottom without errors
- [ ] Write 3–5 concrete findings as markdown cells in the notebook, e.g.:
  - "Feature X is 40% missing → will impute with median + missing-flag"
  - "Target is right-skewed → will log-transform for regression models"
  - "Feature Y has 95% correlation with feature Z → will drop one"
- [ ] These findings become your feature engineering decisions in the next phase

Next: `04_feature_engineering.md`
