import sqlite3


# Function to initialize the database and create the prices table if it doesn't exist
def init_db():
    conn = sqlite3.connect('data/prices.db')
    cursor = conn.cursor()
    cursor.execute(
        'CREATE TABLE IF NOT EXISTS prices (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, PRIMARY KEY(ticker, date))')
    conn.commit()
    conn.close()
