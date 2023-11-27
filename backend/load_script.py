import os
import pandas as pd
import sqlite3
import subprocess

# Local paths
current_dir = os.path.dirname(os.path.realpath(__file__))
zip_file_path = current_dir
csv_file_path = os.path.join(current_dir, 'all_stocks_5yr.csv')
sqlite_db_path = os.path.join(current_dir, "db.sqlite3")

os.makedirs(zip_file_path, exist_ok=True)

# Download the file
subprocess.run(['kaggle', 'datasets', 'download', 'camnugent/sandp500/4', '--unzip'], cwd=zip_file_path)

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Connect to SQLite database
conn = sqlite3.connect(sqlite_db_path)

# Check if the table is empty before inserting data
cursor = conn.cursor()
cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='stocks'")
if cursor.fetchone()[0] == 0:
    # Create a table and insert data
    conn.execute('''
    CREATE TABLE IF NOT EXISTS stocks (
        Date TEXT,
        Open REAL,
        High REAL,
        Low REAL,
        Close REAL,
        Volume INTEGER,
        Name TEXT
    )
    ''')
    df = pd.read_csv(csv_file_path)
    df.to_sql('stocks', conn, if_exists='replace', index=False)
else:
    print("Data already present in the table!")

# Close the connection
conn.close()