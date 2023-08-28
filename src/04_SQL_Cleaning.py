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


# The column 'ID' doesn't provide any useful information for predictions so we elimate it.
# 'TMC', 'Distance(mi)', 'End_Time', 'Duration', 'End_Lat', 'End_Lng' also have to be eliminated because 
# they don't provide any information as predictors, this data only happens afterwards the accident occurs.
# 'Description' also can be eliminated because the Data inside this variable has already been extracted 
# from the dataset by the creators.