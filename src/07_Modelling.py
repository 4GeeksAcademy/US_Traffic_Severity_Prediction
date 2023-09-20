# Import Packages and DataSet
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import gc
from pickle import dump

# Import the dfSet
df = pd.read_csv('../data/raw/US_Accidents_2019_Clean.csv')

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

# Clean garbage
gc.collect()

# Logistic Regression Model Training
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train.values.ravel())
y_pred = lr_model.predict(X_test)
print(f'The Accuracy of the Logistic Regression is: {accuracy_score(y_test, y_pred)*100}%')

# Random Forest 
# Create the Base Models
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(f'The Accuracy of the Random Forest is: {accuracy_score(y_test, y_pred)*100}%')

# Save the Model 
dump(rf_model, open('../models/randomforest_default_42.sav', 'wb'))

# Random Forest Boosting
# Define the hyperparameter grid for each model
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
}

# Create the GridSearchCV object
grid_search = GridSearchCV(rf_model, param_grid=param_grid, cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(best_params)
print(best_model)

# Retrain Model with HyperParam
# Create the Base Models
rf_model = RandomForestClassifier(random_state=42, n_estimators=200)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(f'The Accuracy of the Random Forest is: {accuracy_score(y_test, y_pred)*100}%')

# Save Model 
dump(rf_model, open('../models/randomforest_boost_default_42.sav', 'wb'))

# Train XGBClassifier Model
xg_model = XGBClassifier(random_state = 42, num_class=4)

# Label Encode Data
le = LabelEncoder()
y_train2 = le.fit_transform(y_train)
y_test2 = le.fit_transform(y_test)

# Train the Model
xg_model.fit(X_train, y_train2)
y_pred = xg_model.predict(X_test)
print(f'The Accuracy of the XGBClassifier is: {accuracy_score(y_test2, y_pred)*100}%')

# Save Model 
xg_model.save_model('../models/xgb_default_42.json')
