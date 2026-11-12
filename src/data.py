"""Load Henry Hub natural gas prices and turn them into daily log returns."""

from pathlib import Path

import numpy as np
import pandas as pd

# FRED series DHHNGSP: Henry Hub Natural Gas Spot Price, US$ per MMBtu, daily.
# FRED republishes it from the U.S. Energy Information Administration (EIA).
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP"
DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "henry_hub_daily.csv"


def download_prices(path=DEFAULT_CSV):
    """Download the latest Henry Hub prices from FRED and save them as a CSV.

    Only needed to refresh the data. The notebook reads the saved CSV,
    so it runs offline.
    """
    prices = pd.read_csv(FRED_URL)
    prices.to_csv(path, index=False)
    return path


def load_prices(path=DEFAULT_CSV, start="2008-01-01"):
    """Read the cached CSV and return a clean daily price Series.

    FRED marks days with no trade (holidays) as blank, so those rows are dropped.
    Before 2008 about half of the days are missing, which would turn many
    "daily" returns into multi-day returns, so the sample starts in 2008.
    """
    df = pd.read_csv(path, parse_dates=["observation_date"], index_col="observation_date")
    prices = pd.to_numeric(df["DHHNGSP"], errors="coerce").dropna()
    prices = prices[prices > 0]  # log returns need positive prices
    prices.name = "price"
    return prices.loc[start:]


def log_returns(prices):
    """Daily log return: r_t = ln(P_t / P_{t-1})."""
    returns = np.log(prices).diff().dropna()
    returns.name = "return"
    return returns
