import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# App Config
st.set_page_config(
    page_title="VisionSales: AI Sales Forecast",
    layout="centered",
    page_icon="📊",
    initial_sidebar_state="auto"
)

# Custom Background and Styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("background.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    .css-18e3th9 {
        background-color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 10px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #1e1e2f;
            color: #f1f1f1;
        }
        .main {
            background-color: #2c2f4a;
            padding: 2rem;
            border-radius: 10px;
        }
        .css-18e3th9 {
            padding-top: 2rem;
        }
        .title-style {
            text-align: center;
            font-size: 2.5rem;
            color: #ffffff;
        }
        .subheader-style {
            color: #cccccc;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo
st.image("https://i.imgur.com/5cLsmcU.png", width=100)

# App Title and Description
st.markdown("<h1 class='title-style'>VisionSales: AI Sales Forecast</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader-style'>Upload your historical sales data to forecast the next 3 months using Random Forest & XGBoost.</p>", unsafe_allow_html=True)

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your sales CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
    df = df.resample("ME").sum()

    # Train/Test split
    train_size = int(len(df) * 0.75)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    X = np.arange(len(train)).reshape(-1, 1)
    y = train["Sales"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    xgb = XGBRegressor(objective="reg:squarederror")
    xgb.fit(X_train, y_train)

    # Forecast next 3 months
    future_dates = [df.index[-1] + pd.DateOffset(months=i) for i in range(1, 4)]
    future_idx = np.arange(len(df), len(df) + 3).reshape(-1, 1)

    rf_forecast = rf.predict(future_idx)
    xgb_forecast = xgb.predict(future_idx)
    avg_forecast = (rf_forecast + xgb_forecast) / 2

    # Forecast Table
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "RandomForest": rf_forecast,
        "XGBoost": xgb_forecast,
        "Average": avg_forecast
    })

    st.subheader("📅 Next 3-Month Sales Forecast")
    st.dataframe(forecast_df.set_index("Date"))

    # Plot
    fig, ax = plt.subplots()
    ax.plot(forecast_df["Date"], forecast_df["RandomForest"], label="Random Forest", marker="s")
    ax.plot(forecast_df["Date"], forecast_df["XGBoost"], label="XGBoost", marker="^")
    ax.plot(forecast_df["Date"], forecast_df["Average"], label="Average", marker="*", linestyle="--")

    ax.set_xlabel("Date")
    ax.set_ylabel("Forecasted Sales")
    ax.set_title("Next 3 Months Forecast")
    ax.legend()
    st.pyplot(fig)

    # Textual summary
    max_month = forecast_df.loc[forecast_df['Average'].idxmax(), 'Date'].strftime('%B %Y')
    st.success(f"🔮 Peak forecasted month is **{max_month}** with sales around **{forecast_df['Average'].max():.2f}**")

    # Suggestion
    st.info("💡 **Tip to boost sales**: Consider launching a promotional campaign or discount in months with predicted lower sales to maximize revenue.")











# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.ensemble import RandomForestRegressor
# # import xgboost as xgb
# # from sklearn.model_selection import train_test_split

# # st.set_page_config(page_title="Sales Forecast App", layout="centered")

# # st.title("📈 Sales Forecast using Random Forest & XGBoost")
# # st.markdown("Upload a CSV file with `Date` and `Sales` columns.")

# # uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# # if uploaded_file:
# #     df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")

# #     # Resample monthly
# #     df = df.resample("ME").sum()

# #     # Split
# #     train_size = int(len(df) * 0.75)
# #     train, test = df.iloc[:train_size], df.iloc[train_size:]

# #     X = np.arange(len(train)).reshape(-1, 1)
# #     y = train["Sales"].values

# #     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# #     # Train models
# #     rf = RandomForestRegressor(n_estimators=100, random_state=42)
# #     rf.fit(X_train, y_train)
# #     rf_forecast = rf.predict(np.arange(len(test)).reshape(-1, 1))

# #     xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
# #     xgb_model.fit(X_train, y_train)
# #     xgb_forecast = xgb_model.predict(np.arange(len(test)).reshape(-1, 1))

# #     avg_forecast = (rf_forecast + xgb_forecast) / 2

# #     # Plotting
# #     fig, ax = plt.subplots(figsize=(10, 5))
# #     ax.plot(test.index, test["Sales"], label="Actual Sales", marker="o", linestyle="dashed")
# #     ax.plot(test.index, rf_forecast, label="Random Forest Forecast", marker="s")
# #     ax.plot(test.index, xgb_forecast, label="XGBoost Forecast", marker="^")
# #     ax.plot(test.index, avg_forecast, label="Average Forecast", marker="*", linestyle="dashed", linewidth=2)

# #     ax.set_xlabel("Date")
# #     ax.set_ylabel("Sales")
# #     ax.legend()
# #     ax.set_title("Forecast Results")
# #     st.pyplot(fig)

# #     # Show raw forecasts
# #     forecast_df = pd.DataFrame({
# #         "Date": test.index[:len(avg_forecast)],
# #         "Actual": test["Sales"].values[:len(avg_forecast)],
# #         "RandomForest": rf_forecast,
# #         "XGBoost": xgb_forecast,
# #         "Average": avg_forecast
# #     }).set_index("Date")

# #     st.subheader("📊 Forecast Table")
# #     st.dataframe(forecast_df)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
# from sklearn.model_selection import train_test_split

# # App config
# st.set_page_config(page_title="Sales Forecast App", layout="centered")
# st.title("📈 Sales Forecast using Random Forest & XGBoost")
# st.markdown("Upload a CSV file with `Date` and `Sales` columns (e.g. daily or monthly data).")

# # File upload
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file:
#     df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")

#     # Resample to month-end sales
#     df = df.resample("M").sum()

#     # Split the data
#     train_size = int(len(df) * 0.75)
#     train, test = df.iloc[:train_size], df.iloc[train_size:]

#     # Prepare training data
#     X = np.arange(len(train)).reshape(-1, 1)
#     y = train["Sales"].values
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train models
#     rf = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf.fit(X_train, y_train)

#     xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
#     xgb_model.fit(X_train, y_train)

#     # Predict future 3 months
#     future_steps = 3
#     total_length = len(df)
#     X_future = np.arange(total_length, total_length + future_steps).reshape(-1, 1)

#     rf_forecast = rf.predict(X_future)
#     xgb_forecast = xgb_model.predict(X_future)
#     avg_forecast = (rf_forecast + xgb_forecast) / 2

#     # Future dates
#     last_date = df.index[-1]
#     future_dates = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=future_steps, freq="M")

#     forecast_df = pd.DataFrame({
#         "Date": future_dates,
#         "RandomForest": rf_forecast,
#         "XGBoost": xgb_forecast,
#         "Average": avg_forecast
#     }).set_index("Date")

#     # 📊 Show the forecast table
#     st.subheader("📆 Next 3-Month Sales Forecast")
#     st.dataframe(forecast_df.style.format("{:.2f}"))

#     # 📉 Plot forecast
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.plot(forecast_df.index, forecast_df["RandomForest"], label="Random Forest", marker="s")
#     ax.plot(forecast_df.index, forecast_df["XGBoost"], label="XGBoost", marker="^")
#     ax.plot(forecast_df.index, forecast_df["Average"], label="Average", marker="*", linestyle="dashed", linewidth=2)

#     ax.set_title("Next 3 Months Forecast")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Forecasted Sales")
#     ax.legend()
#     st.pyplot(fig)
