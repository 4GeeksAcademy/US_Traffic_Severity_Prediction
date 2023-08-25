import sqlite3
import pandas as pd

# Connect to the SQLite database
sqlite_db_path = r'../data/raw/us_accidents_2019.db'
conn = sqlite3.connect(sqlite_db_path)
cursor = conn.cursor()

# Define the SQL query to select all columns from the table
query = "SELECT * FROM US_Accidents_2019"

# Load the results into a DataFrame
df = pd.read_sql(query, conn)

# Close the database connection
conn.close()

# Save the DataFrame as a Parquet file
df.to_parquet(r'../data/raw/US_Accidents_2019_V2.parquet', index=False, engine='pyarrow')