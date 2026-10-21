from qpo.data.storage import init_db, save_prices, load_prices
import os
import sqlite3
import pandas as pd


def test_init_db_creates_missing_parent_folders(tmp_path):
    db_path = tmp_path / "nested" / "folder" / "prices.db"
    assert not db_path.parent.exists()

    init_db(db_path)

    assert db_path.exists()


# Function to test the init_db function
def test_init_db_creates_table(tmp_path):
    db_path = tmp_path / "prices.db"
    init_db(db_path)

    assert os.path.exists(db_path)
    assert run_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='prices'", db_path)


# Function to test the save_prices and load_prices functions
def test_save_and_load_prices(tmp_path):
    db_path = tmp_path / "prices.db"
    init_db(db_path)

    save_prices("AAPL", pd.DataFrame({
        'Open': [100, 101],
        'High': [102, 103],
        'Low': [99, 100],
        'Close': [101, 102],
        'Volume': [1000, 1100]
    }, index=pd.to_datetime(['2023-01-03', '2023-01-04'])), db_path)

    result = load_prices("AAPL", "2023-01-01", "2023-01-05", db_path)
    assert result['close'].iloc[0] == 101
    assert len(result) == 2


# Function to run a query on the database
def run_query(sql_text, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(sql_text)
    conn.commit()
    result = cursor.fetchall()

    conn.close()
    return result
