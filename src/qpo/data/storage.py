import sqlite3
import pandas as pd
from pathlib import Path
from typing import cast


# Function to initialize the database and create the prices table if it doesn't exist
def init_db(db_path: str | Path = 'data/prices.db') -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        'CREATE TABLE IF NOT EXISTS prices (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, PRIMARY KEY(ticker, date))')
    conn.commit()
    conn.close()


# Function to save price data for a given ticker into the database
def save_prices(ticker: str, df: pd.DataFrame, db_path: str | Path = 'data/prices.db') -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for date, row in df.iterrows():
        cursor.execute(
            "INSERT OR IGNORE INTO prices (ticker, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ticker, str(cast(pd.Timestamp, date).date()),
             float(row['Open']), float(row['High']), float(row['Low']), float(row['Close']), int(row['Volume']))
        )

    conn.commit()
    conn.close()


# Function to retrieve price data for a given ticker from the database, end exclusive
def load_prices(ticker: str, start: str, end: str, db_path: str | Path = 'data/prices.db') -> pd.DataFrame:
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM prices WHERE ticker = ? AND date >= ? AND date < ? ORDER BY date",
        conn, params=(ticker, start, end), parse_dates=['date'])

    conn.close()
    return df
