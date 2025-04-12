# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
# from sklearn.model_selection import train_test_split

# st.set_page_config(page_title="Sales Forecast App", layout="centered")

# st.title("📈 Sales Forecast using Random Forest & XGBoost")
# st.markdown("Upload a CSV file with `Date` and `Sales` columns.")

# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file:
#     df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")

#     # Resample monthly
#     df = df.resample("ME").sum()

#     # Split
#     train_size = int(len(df) * 0.75)
#     train, test = df.iloc[:train_size], df.iloc[train_size:]

#     X = np.arange(len(train)).reshape(-1, 1)
#     y = train["Sales"].values

#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train models
#     rf = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf.fit(X_train, y_train)
#     rf_forecast = rf.predict(np.arange(len(test)).reshape(-1, 1))

#     xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
#     xgb_model.fit(X_train, y_train)
#     xgb_forecast = xgb_model.predict(np.arange(len(test)).reshape(-1, 1))

#     avg_forecast = (rf_forecast + xgb_forecast) / 2

#     # Plotting
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.plot(test.index, test["Sales"], label="Actual Sales", marker="o", linestyle="dashed")
#     ax.plot(test.index, rf_forecast, label="Random Forest Forecast", marker="s")
#     ax.plot(test.index, xgb_forecast, label="XGBoost Forecast", marker="^")
#     ax.plot(test.index, avg_forecast, label="Average Forecast", marker="*", linestyle="dashed", linewidth=2)

#     ax.set_xlabel("Date")
#     ax.set_ylabel("Sales")
#     ax.legend()
#     ax.set_title("Forecast Results")
#     st.pyplot(fig)

#     # Show raw forecasts
#     forecast_df = pd.DataFrame({
#         "Date": test.index[:len(avg_forecast)],
#         "Actual": test["Sales"].values[:len(avg_forecast)],
#         "RandomForest": rf_forecast,
#         "XGBoost": xgb_forecast,
#         "Average": avg_forecast
#     }).set_index("Date")

#     st.subheader("📊 Forecast Table")
#     st.dataframe(forecast_df)


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split

# App config
st.set_page_config(page_title="Sales Forecast App", layout="centered")
st.title("📈 Sales Forecast using Random Forest & XGBoost")
st.markdown("Upload a CSV file with `Date` and `Sales` columns (e.g. daily or monthly data).")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")

    # Resample to month-end sales
    df = df.resample("M").sum()

    # Split the data
    train_size = int(len(df) * 0.75)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # Prepare training data
    X = np.arange(len(train)).reshape(-1, 1)
    y = train["Sales"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
    xgb_model.fit(X_train, y_train)

    # Predict future 3 months
    future_steps = 3
    total_length = len(df)
    X_future = np.arange(total_length, total_length + future_steps).reshape(-1, 1)

    rf_forecast = rf.predict(X_future)
    xgb_forecast = xgb_model.predict(X_future)
    avg_forecast = (rf_forecast + xgb_forecast) / 2

    # Future dates
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=future_steps, freq="M")

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "RandomForest": rf_forecast,
        "XGBoost": xgb_forecast,
        "Average": avg_forecast
    }).set_index("Date")

    # 📊 Show the forecast table
    st.subheader("📆 Next 3-Month Sales Forecast")
    st.dataframe(forecast_df.style.format("{:.2f}"))

    # 📉 Plot forecast
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(forecast_df.index, forecast_df["RandomForest"], label="Random Forest", marker="s")
    ax.plot(forecast_df.index, forecast_df["XGBoost"], label="XGBoost", marker="^")
    ax.plot(forecast_df.index, forecast_df["Average"], label="Average", marker="*", linestyle="dashed", linewidth=2)

    ax.set_title("Next 3 Months Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Forecasted Sales")
    ax.legend()
    st.pyplot(fig)
