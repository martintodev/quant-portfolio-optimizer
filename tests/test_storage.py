from qpo.data.storage import init_db
import os
import sqlite3


# Function to test the init_db function
def test_init_db_creates_table():
    init_db()

    assert os.path.exists('data/prices.db')
    assert run_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='prices'")


# Function to run a query on the database
def run_query(sql_text):
    conn = sqlite3.connect('data/prices.db')
    cursor = conn.cursor()

    cursor.execute(sql_text)
    conn.commit()
    result = cursor.fetchall()

    conn.close()
    return result
