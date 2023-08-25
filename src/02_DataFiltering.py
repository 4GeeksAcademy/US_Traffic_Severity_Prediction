import pandas as pd

# Read the CSV dataset
df = pd.read_csv('../data/raw/US_Accidents_March23.csv')

# Convert Start_Time and End_Time to datetypes
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

# Extract year, month, day, hour and weekday
df['Year']=df['Start_Time'].dt.year
df['Month']=df['Start_Time'].dt.strftime('%b')
df['Day']=df['Start_Time'].dt.day
df['Hour']=df['Start_Time'].dt.hour
df['Weekday']=df['Start_Time'].dt.strftime('%a')

# Filter the data for years 2019
filtered_data2 = df[(df['Start_Time'].dt.year == 2019)]

# Convert and save to Parquet format
df.to_parquet('../data/raw/US_Accidents_2019.parquet', engine='pyarrow')