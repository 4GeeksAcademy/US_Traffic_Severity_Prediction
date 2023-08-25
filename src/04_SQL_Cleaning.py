import sqlite3

# Connect to the SQLite database
sqlite_db_path = r'../data/raw/us_accidents_2019.db'
conn = sqlite3.connect(sqlite_db_path)
cursor = conn.cursor()

# List of columns to be deleted
columns_to_delete = ['id', 'description', 'distancemi', 'endtime', 'endlat', 'endlng']

# Generate SQL commands to drop the specified columns
table_name = 'US_Accidents_2019'
for column in columns_to_delete:
    drop_column_sql = f"ALTER TABLE {table_name} DROP COLUMN {column}"
    cursor.execute(drop_column_sql)

# Commit the changes and close the connection
conn.commit()
conn.close()