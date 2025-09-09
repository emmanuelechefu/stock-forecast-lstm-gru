import yfinance as yf
import pandas as pd

def download_stock_data(ticker, start_date, end_date):
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        group_by='column',  # <--- THIS FIXES the MultiIndex issue
        auto_adjust=True    # Optional: adjusts prices for splits/dividends
    )

    # Now you can safely access flat columns
    if 'Close' in data.columns and 'Volume' in data.columns:
        data = data[['Close', 'Volume']]
    else:
        print("[ERROR] 'Close' or 'Volume' not found in the data columns!")
        print("Available columns:", data.columns)
        raise KeyError("Missing expected columns in the downloaded data.")
    
    return data
