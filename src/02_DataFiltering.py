import pandas as pd

# Read the CSV dataset
df = pd.read_csv(r'../data/raw/US_Accidents_March23.csv')

# Convert Start_Time and End_Time to datetypes
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

# Convert Start_Time and End_Time to datetypes
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

# Filter the data for years 2019
df2 = df[(df['Start_Time'].dt.year == 2019)]

# Replace symbols in column names with spaces and lower case
df2.columns = df2.columns.str.lower()
df2.columns = df2.columns.str.replace('[^a-zA-Z0-9]', '', regex=True)

# Convert and save to Parquet format
df2.to_parquet(r'../data/raw/US_Accidents_2019.parquet', engine='pyarrow')