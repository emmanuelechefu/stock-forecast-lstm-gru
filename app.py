# app.py
# Streamlit app: Stock forecasting with LSTM/GRU (robust version)

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from contextlib import redirect_stdout
from datetime import date, timedelta

# Silence optional protobuf warnings (harmless but noisy)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# ---- Project modules ----
from src.data_loader import download_stock_data
from src.model_builder import build_lstm_model, build_gru_model
from src.train import train_model
from src.preprocessing import preprocess_data

# ----------------- Streamlit page config -----------------
st.set_page_config(page_title="Stock Forecast (LSTM/GRU)", layout="wide")
st.title("📈 Stock Price Forecast — LSTM / GRU")
st.write(
    "This app downloads recent data from Yahoo Finance, trains a small recurrent network, "
    "evaluates on a hold-out set, and forecasts forward."
)

# ----------------- Helpers (robust) -----------------
@st.cache_data(show_spinner=False)
def get_data(ticker: str, years_back: int) -> pd.DataFrame | None:
    """
    Return tidy DataFrame with ['Close','Volume'] or None if invalid/unavailable.
    """
    t = (ticker or "").strip().upper()
    if not t.isalnum() or len(t) > 10:
        return None

    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=years_back * 365)).isoformat()

    try:
        df = download_stock_data(t, start_date, end_date)  # may return MultiIndex columns
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # Flatten MultiIndex defensively
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns.name = None
    df = df.dropna()

    # Keep expected columns only
    keep = [c for c in df.columns if c.lower() in ("close", "volume")]
    if "Close" not in keep:
        return None

    df = df[keep].copy()

    # Basic sanity: constant / near-constant series is not useful
    if df["Close"].nunique() < 5:
        return None

    return df


def validate_ready(data: pd.DataFrame | None, window: int, min_rows: int = 200) -> tuple[bool, str]:
    """Quick checks before training/forecast."""
    if data is None:
        return False, "Invalid ticker or no data returned for the selected period."
    if window <= 1 or window > 250:
        return False, "Lookback window must be between 2 and 250."
    if len(data) < max(window + 10, min_rows):
        return False, f"Not enough rows ({len(data)}) for window={window}. Try more years of history or a smaller window."
    return True, ""


# sklearn-version-safe RMSE
from sklearn.metrics import mean_squared_error as _mse, mean_absolute_error
def rmse_compat(y_true, y_pred) -> float:
    try:
        return float(_mse(y_true, y_pred, squared=False))  # newer sklearn
    except TypeError:
        return float(np.sqrt(_mse(y_true, y_pred)))        # older sklearn


def forecast_future(model, close_series: pd.Series, scaler, window: int, steps: int) -> np.ndarray:
    """
    Recursive one-step-ahead forecast beyond the last known date.
    Returns array of length `steps` with prices in original scale.
    """
    scaled_full = scaler.transform(close_series.values.reshape(-1, 1))   # (N,1)
    window_seq = scaled_full[-window:].copy()                            # (window,1)
    preds_scaled = []
    for _ in range(steps):
        x_input = window_seq.reshape(1, window, 1)  # (1, window, features)
        next_scaled = model.predict(x_input, verbose=0)[0, 0]
        preds_scaled.append(next_scaled)
        window_seq = np.vstack([window_seq[1:], [[next_scaled]]])
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel()
    return preds


# ----------------- Sidebar controls -----------------
with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Ticker", value="AAPL", help="e.g. AAPL, MSFT, TSLA, NVDA").upper()
    years_back = st.slider("Years of history", 1, 10, 5)
    window = st.slider("Lookback window (days)", 10, 180, 60, step=5)
    model_type = st.selectbox("Model type", ["LSTM", "GRU"])
    epochs = st.slider("Training epochs", 5, 200, 25, step=5)
    batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128], value=32)
    forecast_steps = st.slider("Future forecast (trading days)", 5, 252, 30, step=5)
    run_btn = st.button("▶️ Train & Forecast")

# ----------------- Main run -----------------
if run_btn:
    try:
        with st.spinner("📥 Downloading data..."):
            data = get_data(ticker, years_back)

        ok, msg = validate_ready(data, window)
        if not ok:
            st.error(msg)
            st.info("Tips: check the ticker (e.g., AAPL, MSFT, TSLA), increase years of history, or reduce the lookback window.")
            st.stop()

        st.success(f"Loaded {ticker}: {data.index.min().date()} → {data.index.max().date()} ({len(data)} rows)")
        st.line_chart(data[['Close']].rename(columns={'Close': f'{ticker} Close'}))

        # --- Preprocess (Close only for clean inverse scaling) ---
        X, y, scaler = preprocess_data(data[['Close']], sequence_length=window)

        if len(X) < 100:
            st.warning(f"Very small training set after windowing (samples={len(X)}). Results may be poor.")

        # --- Train/test split (chronological) ---
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # --- Build model ---
        if model_type.lower() == "gru":
            model = build_gru_model((X.shape[1], X.shape[2]))
        else:
            model = build_lstm_model((X.shape[1], X.shape[2]))

        st.write("### Model Summary")
        s = io.StringIO()
        with redirect_stdout(s):
            model.summary()
        st.code(s.getvalue(), language="text")

        # --- Train ---
        st.write("### Training")
        with st.spinner("🧠 Training model..."):
            history = train_model(model, X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)

        # Loss curves
        fig_loss, ax_loss = plt.subplots(figsize=(6, 3))
        ax_loss.plot(history.history['loss'], label='train')
        ax_loss.plot(history.history['val_loss'], label='val')
        ax_loss.set_title('Loss over epochs')
        ax_loss.set_xlabel('Epoch'); ax_loss.set_ylabel('MSE'); ax_loss.legend()
        st.pyplot(fig_loss)

        # --- Evaluate on hold-out test ---
        st.write("### Evaluation (hold-out test)")
        pred_test = model.predict(X_test)
        pred_test_inv = scaler.inverse_transform(pred_test)[:, 0]
        y_test_inv   = scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]

        # Metrics (sklearn-version safe)
        rmse = rmse_compat(y_test_inv, pred_test_inv)
        mae  = float(mean_absolute_error(y_test_inv, pred_test_inv))
        st.write(f"**RMSE:** {rmse:.2f}   |   **MAE:** {mae:.2f}")

        # Plot actual vs predicted
        fig_eval, ax_eval = plt.subplots(figsize=(10, 4))
        ax_eval.plot(y_test_inv, label='Actual', linewidth=1)
        ax_eval.plot(pred_test_inv, label='Predicted', linewidth=1)
        ax_eval.set_title(f'{ticker} — Test Set: Actual vs Predicted')
        ax_eval.set_xlabel('Test Samples (chronological)'); ax_eval.set_ylabel('Price')
        ax_eval.legend()
        st.pyplot(fig_eval)

        # --- Future forecast ---
        st.write("### Future Forecast")
        if forecast_steps < 1 or forecast_steps > 252:
            st.warning("Forecast steps should be between 1 and 252 trading days. Capping to range.")
            forecast_steps = max(1, min(forecast_steps, 252))

        future_prices = forecast_future(model, data['Close'], scaler, window, forecast_steps)
        last_date = data.index[-1]
        future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=forecast_steps)
        future_df = pd.DataFrame({'Forecast': future_prices}, index=future_dates)

        fig_fc, ax_fc = plt.subplots(figsize=(12, 4))
        tail = data['Close'].iloc[-250:]  # last ~250 trading days for context
        ax_fc.plot(tail.index, tail.values, label='History')
        ax_fc.plot(future_df.index, future_df['Forecast'], label='Forecast')
        ax_fc.set_title(f'{ticker} — {model_type.upper()} {forecast_steps}-Day Forecast')
        ax_fc.set_xlabel('Date'); ax_fc.set_ylabel('Price'); ax_fc.legend()
        st.pyplot(fig_fc)

        with st.expander("Show raw data"):
            st.dataframe(data.tail(20))
        with st.expander("Show future forecast table"):
            st.dataframe(future_df)

        st.success("Done ✅")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.exception(e)
else:
    st.info("Set parameters in the sidebar and press **Train & Forecast**.")
