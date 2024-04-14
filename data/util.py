import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


tickers = ["SPY", "AAPL", "INTC", "CSCO", "KO"]
start_date = "1993-01-01"
end_date = "2023-12-31"
interval = '1d'

fig, axs = plt.subplots(len(tickers), 1, figsize=(10, 10))

if len(tickers) == 1:
    axs = [axs]

for i, ticker in enumerate(tickers):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    
    data.to_csv(f"{ticker}_data.csv")
    
    axs[i].plot(data.index, data['Close'], label=f"{ticker} Close Price")
    axs[i].set_title(f"{ticker} Stock Closing Prices")
    axs[i].set_xlabel('Date')
    axs[i].set_ylabel('Price')
    axs[i].legend()

plt.tight_layout()

plt.show()
