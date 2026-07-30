import pandas as pd
import numpy as np

def create_date_features(df):
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    return df

def add_trend(group):
    group = group.sort_values('date').copy()
    group['days_since_start'] = (group['date'] - group['date'].min()).dt.days
    group['elapsed_years'] = group['days_since_start'] / 365.0
    return group

def compute_series_stats(train):
    series_stats = train.groupby(['country', 'store', 'product']).agg(
        overall_mean=('num_sold', 'mean'),
        overall_std=('num_sold', 'std'),
        overall_median=('num_sold', 'median'),
        overall_min=('num_sold', 'min'),
        overall_max=('num_sold', 'max')
    ).reset_index()

    monthly_stats = train.groupby(['country', 'store', 'product', 'month']).agg(
        monthly_mean=('num_sold', 'mean'),
        monthly_std=('num_sold', 'std'),
        monthly_median=('num_sold', 'median')
    ).reset_index()

    yearly_stats = train.groupby(['country', 'store', 'product', 'year']).agg(
        yearly_mean=('num_sold', 'mean')
    ).reset_index()
    yearly_stats['yearly_lag_1'] = yearly_stats.groupby(['country', 'store', 'product'])['yearly_mean'].shift(1)
    yearly_stats['yearly_change'] = yearly_stats['yearly_mean'] - yearly_stats['yearly_lag_1']

    last_year = train[train['year'] == train['year'].max()]
    recent_stats = last_year.groupby(['country', 'store', 'product']).agg(
        recent_mean=('num_sold', 'mean'),
        recent_std=('num_sold', 'std'),
        recent_median=('num_sold', 'median')
    ).reset_index()

    last_90 = train.sort_values('date').groupby(['country', 'store', 'product']).tail(90)
    recent90_stats = last_90.groupby(['country', 'store', 'product']).agg(
        last90_mean=('num_sold', 'mean'),
        last90_std=('num_sold', 'std'),
        last90_median=('num_sold', 'median'),
        last90_min=('num_sold', 'min'),
        last90_max=('num_sold', 'max')
    ).reset_index()

    return series_stats, monthly_stats, yearly_stats, recent_stats, recent90_stats

def add_yearly_lags(combined, lags=[365, 730, 1095]):
    for lag in lags:
        combined[f'lag_yearly_{lag}'] = combined.groupby(['country', 'store', 'product'])['num_sold'].transform(
            lambda x: x.shift(lag)
        )
    return combined

def preprocess(train, test):
    train = train.copy()
    test = test.copy()
    train['date'] = pd.to_datetime(train['date'])
    test['date'] = pd.to_datetime(test['date'])

    train = create_date_features(train)
    test = create_date_features(test)

    series_stats, monthly_stats, yearly_stats, recent_stats, recent90_stats = compute_series_stats(train)

    train = train.groupby(['country', 'store', 'product'], group_keys=False).apply(add_trend)
    test = test.groupby(['country', 'store', 'product'], group_keys=False).apply(add_trend)

    train['_is_train_row'] = True
    test['_is_train_row'] = False
    test['num_sold'] = np.nan
    combined = pd.concat([train, test], ignore_index=True).sort_values(
        ['country', 'store', 'product', 'date']
    ).reset_index(drop=True)
    combined = add_yearly_lags(combined)
    train = combined[combined['_is_train_row'] == True].copy()
    test = combined[combined['_is_train_row'] == False].copy()
    train = train.drop(columns=['_is_train_row'])
    test = test.drop(columns=['_is_train_row', 'num_sold'])

    train = train.merge(series_stats, on=['country', 'store', 'product'], how='left')
    test = test.merge(series_stats, on=['country', 'store', 'product'], how='left')

    train = train.merge(monthly_stats, on=['country', 'store', 'product', 'month'], how='left')
    test = test.merge(monthly_stats, on=['country', 'store', 'product', 'month'], how='left')

    train = train.merge(recent_stats, on=['country', 'store', 'product'], how='left')
    test = test.merge(recent_stats, on=['country', 'store', 'product'], how='left')

    train = train.merge(recent90_stats, on=['country', 'store', 'product'], how='left')
    test = test.merge(recent90_stats, on=['country', 'store', 'product'], how='left')

    train = train.merge(yearly_stats[['country', 'store', 'product', 'year', 'yearly_mean', 'yearly_change']],
                        on=['country', 'store', 'product', 'year'], how='left')
    test = test.merge(yearly_stats[['country', 'store', 'product', 'year', 'yearly_mean', 'yearly_change']],
                      on=['country', 'store', 'product', 'year'], how='left')

    from sklearn.preprocessing import LabelEncoder
    for col in ['country', 'store', 'product']:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]).unique())
        train[f'{col}_encoded'] = le.transform(train[col])
        test[f'{col}_encoded'] = le.transform(test[col])

    train = train.dropna(subset=['num_sold']).reset_index(drop=True)

    fill_cols = [c for c in train.columns if train[c].dtype in ['float64', 'int64'] and c not in ['id', 'num_sold']]
    for col in fill_cols:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        test[col] = test[col].fillna(med)

    return train, test

if __name__ == '__main__':
    train = pd.read_csv('data/raw/train_s5e1.csv')
    test = pd.read_csv('data/raw/test_s5e1.csv')
    train, test = preprocess(train, test)
    train.to_parquet('data/processed/train_s5e1_fe.parquet', index=False)
    test.to_parquet('data/processed/test_s5e1_fe.parquet', index=False)
    print(f'Train: {train.shape}, Test: {test.shape}')
    print(f'Train nulls: {train.isnull().sum().sum()}')
    print(f'Test nulls: {test.isnull().sum().sum()}')
    print(f'Columns: {list(train.columns)}')
