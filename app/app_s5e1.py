import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Sticker Sales Forecaster",
    page_icon="📊",
    layout="wide",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

@st.cache_data
def load_data():
    train = pd.read_csv('data/raw/train_s5e1.csv', parse_dates=['date'])
    test = pd.read_csv('data/raw/test_s5e1.csv', parse_dates=['date'])
    sub = pd.read_csv('data/raw/sample_submission_s5e1.csv')
    return train, test, sub

@st.cache_resource
def load_model():
    lgb_path = os.path.join(MODEL_DIR, 's5e1', 'lgb_s5e1_full.pkl')
    xgb_path = os.path.join(MODEL_DIR, 's5e1', 'xgb_s5e1_full.pkl')
    if os.path.exists(lgb_path):
        return joblib.load(lgb_path), 'lgb'
    elif os.path.exists(xgb_path):
        return joblib.load(xgb_path), 'xgb'
    return None, None

train, test, sample = load_data()
model, model_name = load_model()

st.title("📊 Sticker Sales Forecaster")
st.caption("Kaggle Playground Series S5E1 — Forecasting Sticker Sales")
st.markdown("Predict sticker sales across 6 countries, 3 stores, and 5 products (90 time series).")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Train Rows", f"{len(train):,}", "2010-2016")
col2.metric("Test Rows", f"{len(test):,}", "2017-2019")
col3.metric("Countries", f"{train['country'].nunique()}")
col4.metric("Products", f"{train['product'].nunique()}")

st.subheader("Historical Sales by Country")
daily_sales = train.groupby(['date', 'country'])['num_sold'].sum().reset_index()
fig = px.line(daily_sales, x='date', y='num_sold', color='country',
              title='Total Daily Sticker Sales by Country')
st.plotly_chart(fig, use_container_width=True)

st.subheader("Sales Distribution")
col1, col2 = st.columns(2)
with col1:
    fig2 = px.box(train, x='country', y='num_sold', color='country',
                  title='num_sold by Country')
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    fig3 = px.box(train, x='product', y='num_sold', color='product',
                  title='num_sold by Product')
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("Monthly Seasonality")
train['month'] = train['date'].dt.month
monthly = train.groupby(['month', 'country'])['num_sold'].mean().reset_index()
fig4 = px.line(monthly, x='month', y='num_sold', color='country',
               title='Average Monthly Sales by Country',
               markers=True)
st.plotly_chart(fig4, use_container_width=True)

st.subheader("Explore a Time Series")
countries = ['All'] + sorted(train['country'].unique().tolist())
stores = ['All'] + sorted(train['store'].unique().tolist())
products = ['All'] + sorted(train['product'].unique().tolist())

sel_country = st.selectbox("Country", countries, index=0)
sel_store = st.selectbox("Store", stores, index=0)
sel_product = st.selectbox("Product", products, index=0)

filtered = train.copy()
if sel_country != 'All':
    filtered = filtered[filtered['country'] == sel_country]
if sel_store != 'All':
    filtered = filtered[filtered['store'] == sel_store]
if sel_product != 'All':
    filtered = filtered[filtered['product'] == sel_product]

series_data = filtered.groupby('date')['num_sold'].sum().reset_index()
fig5 = px.line(series_data, x='date', y='num_sold',
               title=f'Sales: {sel_country} / {sel_store} / {sel_product}')
st.plotly_chart(fig5, use_container_width=True)

if model is not None:
    st.subheader("Model Predictions")
    if os.path.exists('data/processed/test_s5e1_fe.parquet'):
        test_fe = pd.read_parquet('data/processed/test_s5e1_fe.parquet')
        feature_cols = [c for c in test_fe.columns if c not in ['id', 'date',
            'country', 'store', 'product']]
        test_pred = model.predict(test_fe[feature_cols])
        test_pred = np.maximum(test_pred, 0)
    else:
        sub = pd.read_csv('submissions/submission_s5e1_ensemble.csv')
        test_pred = sub['num_sold'].values
    test['prediction'] = test_pred
    test['num_sold'] = test_pred

    sel_c2 = sel_country if sel_country != 'All' else 'Canada'
    sel_s2 = sel_store if sel_store != 'All' else 'Discount Stickers'
    sel_p2 = sel_product if sel_product != 'All' else 'Kaggle'
    viz_test = test[(test['country'] == sel_c2) &
                    (test['store'] == sel_s2) &
                    (test['product'] == sel_p2)]

    if len(viz_test) > 0:
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(
            x=viz_test['date'], y=viz_test['prediction'],
            mode='lines+markers', name=f'Forecast',
            line=dict(color='orange', width=2)
        ))
        train_viz = train[(train['country'] == sel_c2) &
                          (train['store'] == sel_s2) &
                          (train['product'] == sel_p2)]
        if len(train_viz) > 0:
            fig6.add_trace(go.Scatter(
                x=train_viz['date'], y=train_viz['num_sold'],
                mode='lines', name='Historical',
                line=dict(color='blue', width=1)
            ))
        fig6.update_layout(title=f'Forecast: {sel_c2} / {sel_s2} / {sel_p2}')
        st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Submission Preview")
    sub = sample.copy()
    sub['num_sold'] = test_pred
    st.dataframe(sub.head(10), use_container_width=True, hide_index=True)

    csv = sub.to_csv(index=False).encode('utf-8')
    st.download_button("Download Submission CSV", csv, "submission_s5e1.csv", "text/csv")
else:
    st.warning("No trained model found. Run train_s5e1.py first.")

with st.expander("About this model"):
    st.write(f"- **Model:** {model_name.upper() if model else 'N/A'} Regressor")
    st.write("- **Metric:** Mean Absolute Percentage Error (MAPE)")
    st.write("- **Features:** Date, trend, yearly lags, target encodings, monthly stats")
    st.write("- **Validation MAPE:** ~8.6% (2016 holdout)")
    st.write("- **90 time series:** 6 countries × 3 stores × 5 products")
    st.write("- **Competition:** Kaggle Playground Series S5E1, Jan 2025")
