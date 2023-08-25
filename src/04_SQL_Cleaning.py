import sqlite3
import pandas as pd

# Define the path to your SQLite database
database_path = r'../data/raw/us_accidents_2019.db'

# Connect to the database
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# Define the table name
table_name = 'usaccidents_table'

# Define the column names to drop
columns_to_drop = ['ID', 'TMC', 'Description', 'Distance(mi)', 'End_Time', 'Duration', 'End_Lat', 'End_Lng']

# Generate the SQL query to drop the specified columns
drop_columns_query = f"ALTER TABLE {table_name} DROP COLUMN {', '.join(columns_to_drop)}"

# Execute the query to drop columns
cursor.execute(drop_columns_query)
conn.commit()

# Define the SQL query to select all remaining columns from the table
select_remaining_columns_query = f"SELECT * FROM {table_name}"

# Load the results into a DataFrame
df = pd.read_sql(select_remaining_columns_query, conn)

# Close the database connection
conn.close()

# Save the DataFrame as a Parquet file
df.to_parquet(r'../data/raw/US_Accidents_2019_V2.parquet', index=False)