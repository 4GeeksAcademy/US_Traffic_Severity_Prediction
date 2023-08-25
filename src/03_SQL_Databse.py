# Import Packages
import sqlite3
import pandas as pd

# Create the Database
database = '../data/raw/us_accidents_2019.db'

# Connect to the Database
conn = sqlite3.connect(database)
cursor = conn.cursor()

# Load the Raw Original Dataset 
df = pd.read_parquet('../data/raw/US_Accidents_2019.parquet')

# Create new table for the data
table = 'US_Accidents_2019'

# Export the Dataset to SQL
df.to_sql(table, conn, if_exists='replace', index=False)

# Define the SQL query to retrieve the first 10 rows from your table
query = "SELECT * FROM US_Accidents_2019 LIMIT 10"

# Execute the query and load the results into a DataFrame
df = pd.read_sql(query, conn)
                 
# Close the database connection
conn.close()

# Print the first 10 rows
print(df.head(10))