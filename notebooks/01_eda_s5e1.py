import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('data/raw/train_s5e1.csv', parse_dates=['date'])
test = pd.read_csv('data/raw/test_s5e1.csv', parse_dates=['date'])

print("=== TRAIN ===")
print(f"Shape: {train.shape}")
print(f"Date range: {train['date'].min()} to {train['date'].max()}")
print(f"Memory: {train.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nNulls:\n{train.isnull().sum()}")
print(f"\n=== num_sold ===")
print(train['num_sold'].describe())
print(f"\nSkew: {train['num_sold'].skew():.2f}")
print(f"Kurtosis: {train['num_sold'].kurtosis():.2f}")

print("\n=== CATEGORICAL ===")
for col in ['country', 'store', 'product']:
    print(f"\n{col}:")
    print(train[col].value_counts())

print("\n=== TEST ===")
print(f"Shape: {test.shape}")
print(f"Date range: {test['date'].min()} to {test['date'].max()}")

# Nulls: only Holographic Goose has all NaN
null_mask = train['num_sold'].isnull()
print(f"\nNull num_sold rows breakdown:")
print(train[null_mask][['country', 'store', 'product']].value_counts())

# Time series structure: 90 unique combos
combos = train.groupby(['country', 'store', 'product']).size()
print(f"\nUnique time series: {len(combos)}")
print(f"Expected: 6*3*5 = {6*3*5}")

# Check each series completeness
print("\nSeries date coverage:")
coverage = train.groupby(['country', 'store', 'product']).agg(
    min_date=('date', 'min'),
    max_date=('date', 'max'),
    n_rows=('num_sold', 'count'),
    n_null=('num_sold', lambda x: x.isnull().sum())
).reset_index()
print(coverage.head(10))

# Seasonality check by month
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
monthly = train.groupby('month')['num_sold'].mean()
print(f"\nMonthly avg num_sold:\n{monthly}")

# Yearly trend
yearly = train.groupby('year')['num_sold'].mean()
print(f"\nYearly avg num_sold:\n{yearly}")

# Country X store X product avg
agg = train.groupby(['country', 'store', 'product'])['num_sold'].agg(['mean', 'std', 'count'])
print(f"\nGroup stats:\n{agg.describe()}")

# Top combinations by volume
top_combos = train.groupby(['country', 'store', 'product'])['num_sold'].mean().sort_values(ascending=False)
print(f"\nTop 10 by avg sales:\n{top_combos.head(10)}")
print(f"\nBottom 10 by avg sales:\n{top_combos.tail(10)}")

print("\n=== PLOTS ===")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

train.groupby('date')['num_sold'].sum().plot(ax=axes[0,0], title='Total Sales Over Time', alpha=0.7)
train.groupby('date')['num_sold'].mean().plot(ax=axes[0,1], title='Mean Sales Over Time', alpha=0.7)
train.boxplot(column='num_sold', by='country', ax=axes[0,2])
axes[0,2].set_title('num_sold by Country')
train.boxplot(column='num_sold', by='store', ax=axes[1,0])
axes[1,0].set_title('num_sold by Store')
train.boxplot(column='num_sold', by='product', ax=axes[1,1])
axes[1,1].set_title('num_sold by Product')
train.hist(column='num_sold', ax=axes[1,2], bins=50)
axes[1,2].set_title('num_sold Distribution')

plt.tight_layout()
plt.savefig('data/processed/eda_s5e1_overview.png', dpi=150)

# Monthly seasonality plot
fig2, ax2 = plt.subplots(figsize=(12, 4))
monthly_pivot = train.groupby(['year', 'month'])['num_sold'].mean().unstack(0)
monthly_pivot.T.plot(ax=ax2, marker='o', alpha=0.7)
ax2.set_title('Monthly Seasonality by Year')
ax2.set_xlabel('Month')
ax2.set_ylabel('Mean num_sold')
plt.tight_layout()
plt.savefig('data/processed/eda_s5e1_seasonality.png', dpi=150)

print("Saved plots to data/processed/")
print("\n=== EDA COMPLETE ===")
