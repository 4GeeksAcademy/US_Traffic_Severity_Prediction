# Import Packages and DataSet

import pandas as pd
import numpy as np
import re
from collections import Counter


# Import the dfSet
df = pd.read_parquet('../data/raw/US_Accidents_2019_for_EDA.parquet')

# Count the number of duplicates before removing them
num_duplicates = df.duplicated().sum()

# Remove duplicates from the entire dfFrame
df_without_duplicates = df.drop_duplicates()

# Print the number of duplicates and the shape of the DataFrame
print('Number of duplicates before removing:', num_duplicates)
print('Shape before removing duplicates:', df.shape)
print('Shape after removing duplicates:', df_without_duplicates.shape)

# Organize Road Data

# create a list of top 40 most common words in street name
st_type =' '.join(df['street'].unique().tolist()) # flat the array of street name
st_type = re.split(' |-', st_type) # split the long string by space and hyphen
st_type = [x[0] for x in Counter(st_type).most_common(40)] # select the 40 most common words
print('the 40 most common words')
print(*st_type, sep = ', ') 

# Remove some irrelevant words and add spaces and hyphen back
st_type= [' Rd', ' St', ' Dr', ' Ave', ' Blvd', ' Ln', ' Highway', ' Pkwy', ' Hwy', 
          ' Way', ' Ct', 'Pl', ' Road', 'US-', 'Creek', ' Cir',  'Route', 
          'I-', 'Trl', 'Pike', ' Fwy']
print(*st_type, sep = ', ')  

# for each word create a boolean column
for i in st_type:
  df[i.strip()] = np.where(df['street'].str.contains(i, case=True, na = False), True, False)
df.loc[df['Road']==1,'Rd'] = True
df.loc[df['Highway']==1,'Hwy'] = True

df['severity'] = df['severity'].astype(int)
street_corr  = df.loc[:,['severity']+[x.strip() for x in st_type]].corr()

# Drop low correlation Road Types
drop_list = street_corr.index[street_corr['severity'].abs()<0.05].to_list()
df = df.drop(drop_list, axis=1)
df = df.drop(['street'], axis=1)

# Simplyfing Wind Direction Data

# Group and Simplify the various attributes
df.loc[df['winddirection']=='Calm','winddirection'] = 'CALM'
df.loc[(df['winddirection']=='West')|(df['winddirection']=='WSW')|(df['winddirection']=='WNW'),'winddirection'] = 'W'
df.loc[(df['winddirection']=='South')|(df['winddirection']=='SSW')|(df['winddirection']=='SSE'),'winddirection'] = 'S'
df.loc[(df['winddirection']=='North')|(df['winddirection']=='NNW')|(df['winddirection']=='NNE'),'winddirection'] = 'N'
df.loc[(df['winddirection']=='East')|(df['winddirection']=='ESE')|(df['winddirection']=='ENE'),'winddirection'] = 'E'
df.loc[df['winddirection']=='Variable','winddirection'] = 'VAR'
print('Wind Direction after simplification: ', df['winddirection'].unique())

# Drop NaN from Weather Conditions
df = df.dropna(subset=['weathercondition'])

# One Hot Encode the Weather Data
df['clear'] = np.where(df['weathercondition'].str.contains('Clear|Fair', case=False, na = False), True, False)
df['cloud'] = np.where(df['weathercondition'].str.contains('Cloud|Overcast|Cloudy', case=False, na = False), True, False)
df['rain'] = np.where(df['weathercondition'].str.contains('Rain|storm', case=False, na = False), True, False)
df['heavyrain'] = np.where(df['weathercondition'].str.contains('Heavy Rain|Rain Shower|Heavy T-Storm|Heavy Thunderstorms', case=False, na = False), True, False)
df['snow'] = np.where(df['weathercondition'].str.contains('Snow|Sleet|Ice|Wintry|Hail', case=False, na = False), True, False)
df['heavysnow'] = np.where(df['weathercondition'].str.contains('Heavy Snow|Heavy Sleet|Heavy Ice Pellets|Snow Showers|Squalls', case=False, na = False), True, False)
df['fog'] = np.where(df['weathercondition'].str.contains('Fog|Haze|Mist|Smoke|Sand|Dust', case=False, na = False), True, False)

df = df.drop(['weathercondition'], axis=1)

# Compare Start Time and Weather Time Stamp to check if there are differences

#Connvert Weather Timestamp to Date Type
df['weathertimestamp'] = pd.to_datetime(df['weathertimestamp'], errors='coerce')
print((df.weathertimestamp - df.starttime).mean())
# Delete Weather Timestamp 
df = df.drop(['weathertimestamp'], axis=1)

# Extract Minutes (in a day) Information from our Start Time feature
df['minute']=df['hour']*60.0+df['starttime'].dt.minute
df.loc[:4, ['starttime', 'year', 'month', 'day', 'hour', 'weekday', 'minute']]

# Drop NaN Values
missing = pd.DataFrame(df.isnull().sum()).reset_index()
missing.columns = ['Feature', 'Missing_Percent(%)']
missing['Missing_Percent(%)'] = missing['Missing_Percent(%)'].apply(lambda x: x / df.shape[0] * 100)
print(missing.loc[missing['Missing_Percent(%)']>0,:])

# Drop NaN Values from all Dataset
df = df.dropna()

# Drop Outliers
df[df['precipitationin'] < 5]
df[df['windspeedmph'] < 120]
df[df['temperaturef'] < 125]

# Factorize the Object Data
df['source_n'] = pd.factorize(df['source'])[0]
df['city_n'] = pd.factorize(df['city'])[0]
df['county_n'] = pd.factorize(df['county'])[0]
df['state_n'] = pd.factorize(df['state'])[0]
df['zipcode_n'] = pd.factorize(df['zipcode'])[0]
df['timezone_n'] = pd.factorize(df['timezone'])[0]
df['airportcode_n'] = pd.factorize(df['airportcode'])[0]
df['winddirection_n'] = pd.factorize(df['winddirection'])[0]
df['sunrisesunset_n'] = pd.factorize(df['sunrisesunset'])[0]
df['civiltwilight_n'] = pd.factorize(df['civiltwilight'])[0]
df['nauticaltwilight_n'] = pd.factorize(df['nauticaltwilight'])[0]
df['astronomicaltwilight_n'] = pd.factorize(df['astronomicaltwilight'])[0]
df['month_n'] = pd.factorize(df['month'])[0]
df['weekday_n'] = pd.factorize(df['weekday'])[0]

# Drop unecessary features
df = df.drop(
    ['source', 'source_n', 'county', 'county_n', 'zipcode', 'zipcode_n', 'timezone', 'timezone_n', 'airportcode', 'airportcode_n', 
     'windchillf', 'year', 'civiltwilight', 'civiltwilight_n', 'nauticaltwilight', 
     'nauticaltwilight_n','astronomicaltwilight','astronomicaltwilight_n', 'hour'], axis=1)

# Save Clean DataSet 
df.to_csv('../data/processed/US_Accidents_2019_Clean.csv')
