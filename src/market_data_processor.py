import pandas as pd
import matplotlib.pyplot as plt
import talib
import pynance as pn


class MarketDataProcessor:
    """
    A reusable class for loading stock data, preparing it,
    applying TA-Lib indicators, calculating financial metrics,
    and visualizing results.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None

    # --------------------------------------------------
    # LOAD & PREPARE DATA
    # --------------------------------------------------
    def load_data(self):
        """Load stock price data into a pandas DataFrame."""
        self.data = pd.read_csv(self.file_path)

        # Basic validation
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert Date to datetime
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.data.sort_values("Date", inplace=True)
        self.data.set_index("Date", inplace=True)
        return self.data

    # --------------------------------------------------
    # APPLY TA-LIB INDICATORS
    # --------------------------------------------------
    def apply_ta_indicators(self):
        """
        Apply commonly-used technical indicators using TA-Lib.
        Includes SMA, EMA, RSI, MACD.
        """

        close = self.data["Close"]

        # Moving Averages
        self.data["SMA_20"] = talib.SMA(close, timeperiod=20)
        self.data["EMA_20"] = talib.EMA(close, timeperiod=20)

        # RSI
        self.data["RSI_14"] = talib.RSI(close, timeperiod=14)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close)
        self.data["MACD"] = macd
        self.data["MACD_Signal"] = macd_signal
        self.data["MACD_Hist"] = macd_hist

        return self.data

    # --------------------------------------------------
    # FINANCIAL METRICS (PyNance)
    # --------------------------------------------------
    def calculate_financial_metrics(self):
     """
     Computes financial metrics like daily returns, volatility,
     and cumulative returns using pandas.
     """

     # Ensure 'Close' exists
     if "Close" not in self.data.columns:
        raise ValueError("Column 'Close' not found in data")

     # Daily returns
     self.data["Returns"] = self.data["Close"].pct_change()

     # 20-day rolling volatility (annualized)
     self.data["Volatility_20"] = (
        self.data["Returns"].rolling(20).std() * (252 ** 0.5)
     )

    # Cumulative returns
     self.data["Cumulative_Returns"] = (1 + self.data["Returns"]).cumprod()

     return self.data

    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------
    def visualize(self, show=True, save_path=None):
        """Plot stock price and key indicators."""

        plt.figure(figsize=(14, 8))
        plt.plot(self.data["Close"], label="Close Price")
        plt.plot(self.data["SMA_20"], label="SMA 20")
        plt.plot(self.data["EMA_20"], label="EMA 20")
        plt.title("Close Price with Moving Averages")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()

        # RSI Plot
        plt.figure(figsize=(14, 4))
        plt.plot(self.data["RSI_14"], label="RSI 14", color="orange")
        plt.axhline(70, color="red", linestyle="--")
        plt.axhline(30, color="green", linestyle="--")
        plt.title("RSI Indicator")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path.replace(".png", "_rsi.png"))

        if show:
            plt.show()

        # MACD Plot
        plt.figure(figsize=(14, 4))
        plt.plot(self.data["MACD"], label="MACD", color="blue")
        plt.plot(self.data["MACD_Signal"], label="Signal", color="red")
        plt.bar(self.data.index, self.data["MACD_Hist"], label="Histogram")
        plt.title("MACD Indicator")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path.replace(".png", "_macd.png"))

        if show:
            plt.show()
