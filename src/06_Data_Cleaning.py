# Import Packages and DataSet

import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

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

# Standard Scalling Data:

num_variables = ['city_n', 'state_n', 
       'winddirection_n', 'sunrisesunset_n', 'month_n',
       'weekday_n', "startlat", "startlng", 'temperaturef', 'humidity', 'pressurein', 'visibilitymi',
       'windspeedmph', 'precipitationin', 'amenity', 'bump',
       'crossing', 'giveway', 'junction', 'noexit', 'railway', 'roundabout',
       'station', 'stop', 'trafficcalming', 'trafficsignal', 'day', 'minute', 'clear', 'cloud', 'rain',
       'heavyrain', 'snow', 'heavysnow', 'fog', 'Rd', 'I-', 'St', 'Dr', 'Ave', 'Blvd'
       ]

scaler = StandardScaler()
norm_features = scaler.fit_transform(df[num_variables])
df_norm = pd.DataFrame(norm_features, index = df.index, columns = num_variables)
df_norm['severity'] = df['severity']
df_norm.head()

# Save the Scaler
with open('../models/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Test / Train Split
# Create Datasets for each type
data1 = df_norm[df_norm['severity'] == 1]
data2 = df_norm[df_norm['severity'] == 2]
data3 = df_norm[df_norm['severity'] == 3]
data4 = df_norm[df_norm['severity'] == 4]

X = data1.drop('severity', axis = 1)
y = data1['severity']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.3, random_state = 42)

X = data2.drop('severity', axis = 1)
y = data2['severity']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size = 0.3, random_state = 42)

X = data3.drop('severity', axis = 1)
y = data3['severity']

X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size = 0.3, random_state = 42)

X = data4.drop('severity', axis = 1)
y = data4['severity']

X_train4, X_test4, y_train4, y_test4 = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Concatenate the Datasets
X_train = pd.concat([X_train1, X_train2, X_train3, X_train4], ignore_index=True)
X_test = pd.concat([X_test1, X_test2, X_test3, X_test4], ignore_index=True)
y_train = pd.concat([y_train1, y_train2, y_train3, y_train4], ignore_index=True)
y_test = pd.concat([y_test1, y_test2, y_test3, y_test4], ignore_index=True)

# Save Train and Test Data 
X_train.to_csv('../data/processed/usaccidents_xtrain.csv', index=False)
X_test.to_csv('../data/processed/usaccidents_xtest.csv', index=False)
y_train.to_csv('../data/processed/usaccidents_ytrain.csv', index=False)
y_test.to_csv('../data/processed/usaccidents_ytest.csv', index=False)

# Save Uniques Cat-Factorized
unique_states = df[['state', 'state_n']].drop_duplicates()
unique_states.to_csv('../data/processed/unique_states.csv')

unique_cities = df[['city', 'city_n']].drop_duplicates()
unique_cities.to_csv('../data/processed/unique_cities.csv')

unique_weekdays = df[['weekday', 'weekday_n']].drop_duplicates()
unique_weekdays.to_csv('../data/processed/unique_weekdays.csv')

unique_months = df[['month', 'month_n']].drop_duplicates()
unique_months.to_csv('../data/processed/unique_months.csv')

unique_sunrisesunset = df[['sunrisesunset', 'sunrisesunset_n']].drop_duplicates()
unique_sunrisesunset.to_csv('../data/processed/unique_sunrisesunset.csv')

unique_winddirection = df[['winddirection', 'winddirection_n']].drop_duplicates()
unique_winddirection.to_csv('../data/processed/unique_winddirection.csv')

unique_days = df[['day']].drop_duplicates()
unique_days.to_csv('../data/processed/unique_days.csv')
