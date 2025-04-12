import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Sales Forecast App", layout="centered")

st.title("📈 Sales Forecast using Random Forest & XGBoost")
st.markdown("Upload a CSV file with `Date` and `Sales` columns.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")

    # Resample monthly
    df = df.resample("ME").sum()

    # Split
    train_size = int(len(df) * 0.75)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    X = np.arange(len(train)).reshape(-1, 1)
    y = train["Sales"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_forecast = rf.predict(np.arange(len(test)).reshape(-1, 1))

    xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
    xgb_model.fit(X_train, y_train)
    xgb_forecast = xgb_model.predict(np.arange(len(test)).reshape(-1, 1))

    avg_forecast = (rf_forecast + xgb_forecast) / 2

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test.index, test["Sales"], label="Actual Sales", marker="o", linestyle="dashed")
    ax.plot(test.index, rf_forecast, label="Random Forest Forecast", marker="s")
    ax.plot(test.index, xgb_forecast, label="XGBoost Forecast", marker="^")
    ax.plot(test.index, avg_forecast, label="Average Forecast", marker="*", linestyle="dashed", linewidth=2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.set_title("Forecast Results")
    st.pyplot(fig)

    # Show raw forecasts
    forecast_df = pd.DataFrame({
        "Date": test.index[:len(avg_forecast)],
        "Actual": test["Sales"].values[:len(avg_forecast)],
        "RandomForest": rf_forecast,
        "XGBoost": xgb_forecast,
        "Average": avg_forecast
    }).set_index("Date")

    st.subheader("📊 Forecast Table")
    st.dataframe(forecast_df)
