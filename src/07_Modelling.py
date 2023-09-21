# Import Packages and DataSet
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import gc
from pickle import dump

# Import the dfSet
X_test = pd.read_csv('../data/processed/usaccidents_xtest.csv')
X_train = pd.read_csv('../data/processed/usaccidents_xtrain.csv')
y_test = pd.read_csv('../data/processed/usaccidents_ytest.csv')
y_train = pd.read_csv('../data/processed/usaccidents_ytrain.csv')

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
dump(rf_model, open('../models/randomforest_default_42.pkl', 'wb'))

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
dump(rf_model, open('../models/randomforest_boost_default_42.pkl', 'wb'))

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
