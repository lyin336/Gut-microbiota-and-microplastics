########################Random Forest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load data
features = pd.read_csv('alpha.csv')

# Separate features and target variable
features_true = features.drop(columns=["alpha"])
X = features_true
y = features["alpha"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log10 transform the target variable
y_train = np.log10(y_train).ravel()
y_test = np.log10(y_test).ravel()

# Perform one-hot encoding on the training and test sets separately to avoid data leakage
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Ensure training and test sets have the same columns (in case some categorical variables are missing in the test set)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Define the preprocessing pipeline (imputation and scaling)
preprocessing = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with the median
    ('scaler', StandardScaler())  # Standardize the data (Z-score normalization)
])

# Combine preprocessing and RandomForest model into a final pipeline
final_pipeline = Pipeline([
    ('preprocessing', preprocessing),  # Preprocessing steps
    ('model', RandomForestRegressor(random_state=42))  # Random Forest model
])

# Define parameter grid for hyperparameter tuning
param_grid = {
    'model__n_estimators': [100, 200, 300, 400, 500],
    'model__max_depth': [10, 20, 30, 40, 50],
    'model__min_samples_split': [2, 4, 6, 8, 10]
}

# Use GridSearchCV for hyperparameter tuning and perform cross-validation during the process
grid_search = GridSearchCV(
    estimator=final_pipeline,
    param_grid=param_grid,
    cv=10,
    scoring={'r2': 'r2', 'rmse': make_scorer(mean_squared_error, squared=False)},
    refit='r2',
    n_jobs=2,
    verbose=2
)

# Fit the GridSearchCV with the training data
grid_search.fit(X_train, y_train)

# Output the best parameter combination
print("Best parameter combination:", grid_search.best_params_)

# Extract all cross-validation results
cv_results = grid_search.cv_results_

# Get the index of the best parameter combination
best_index = grid_search.best_index_

# Extract the 10-fold cross-validation results corresponding to the best parameter combination
best_r2_scores = [cv_results[f'split{i}_test_r2'][best_index] for i in range(10)]
best_rmse_scores = [cv_results[f'split{i}_test_rmse'][best_index] for i in range(10)]

# Compute the mean and standard deviation of the R² and RMSE scores from 10-fold cross-validation
best_r2_mean = np.mean(best_r2_scores)
best_r2_std = np.std(best_r2_scores)
best_rmse_mean = np.mean(best_rmse_scores)
best_rmse_std = np.std(best_rmse_scores)

print(f"\nMean ± Standard Deviation of R² from 10-fold cross-validation: {best_r2_mean:.4f} ± {best_r2_std:.4f}")
print(f"Mean ± Standard Deviation of RMSE from 10-fold cross-validation: {best_rmse_mean:.4f} ± {best_rmse_std:.4f}")

# Use the best parameter combination to predict on the test set
y_pred = grid_search.predict(X_test)

# Compute the R² and RMSE scores on the test set
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\nTest set R² score:", r2)
print("Test set RMSE score:", rmse)

######################XGBoost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor 

# Load data
features = pd.read_csv('alpha.csv')

# Separate features and target variable
features_true = features.drop(columns=["alpha"])
X = features_true
y = features["alpha"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log10 transform the target variable
y_train = np.log10(y_train).ravel()
y_test = np.log10(y_test).ravel()

# Perform one-hot encoding on the training and test sets separately to avoid data leakage
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Ensure training and test sets have the same columns (in case some categorical variables are missing in the test set)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Define the preprocessing pipeline (imputation and scaling)
preprocessing = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with the median
    ('scaler', StandardScaler())  # Standardize the data (Z-score normalization)
])

# Combine preprocessing and XGBoost model into a final pipeline
final_pipeline = Pipeline([
    ('preprocessing', preprocessing),  # Preprocessing steps
    ('model', XGBRegressor(random_state=42))  # XGBoost model
])

# Define parameter grid for hyperparameter tuning
param_grid = {
    'model__n_estimators': [100, 200, 300, 400, 500],
    'model__max_depth': [10, 20, 30, 40, 50],
    'model__learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25]
}

# Use GridSearchCV for hyperparameter tuning and perform cross-validation during the process
grid_search = GridSearchCV(
    estimator=final_pipeline,
    param_grid=param_grid,
    cv=10,
    scoring={'r2': 'r2', 'rmse': make_scorer(mean_squared_error, squared=False)},
    refit='r2',
    n_jobs=2,
    verbose=2
)

# Fit the GridSearchCV with the training data
grid_search.fit(X_train, y_train)

# Output the best parameter combination
print("Best parameter combination:", grid_search.best_params_)

# Extract all cross-validation results
cv_results = grid_search.cv_results_

# Get the index of the best parameter combination
best_index = grid_search.best_index_

# Extract the 10-fold cross-validation results corresponding to the best parameter combination
best_r2_scores = [cv_results[f'split{i}_test_r2'][best_index] for i in range(10)]
best_rmse_scores = [cv_results[f'split{i}_test_rmse'][best_index] for i in range(10)]

# Compute the mean and standard deviation of the R² and RMSE scores from 10-fold cross-validation
best_r2_mean = np.mean(best_r2_scores)
best_r2_std = np.std(best_r2_scores)
best_rmse_mean = np.mean(best_rmse_scores)
best_rmse_std = np.std(best_rmse_scores)

print(f"\nMean ± Standard Deviation of R² from 10-fold cross-validation: {best_r2_mean:.4f} ± {best_r2_std:.4f}")
print(f"Mean ± Standard Deviation of RMSE from 10-fold cross-validation: {best_rmse_mean:.4f} ± {best_rmse_std:.4f}")

# Use the best parameter combination to predict on the test set
y_pred = grid_search.predict(X_test)

# Compute the R² and RMSE scores on the test set
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\nTest set R² score:", r2)
print("Test set RMSE score:", rmse)

#######################ANN (MLP)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor  

# Load data
features = pd.read_csv('alpha.csv')

# Separate features and target variable
features_true = features.drop(columns=["alpha"])
X = features_true
y = features["alpha"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log10 transform the target variable
y_train = np.log10(y_train).ravel()
y_test = np.log10(y_test).ravel()

# Perform one-hot encoding on the training and test sets separately to avoid data leakage
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Ensure training and test sets have the same columns (in case some categorical variables are missing in the test set)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Define the preprocessing pipeline (imputation and scaling)
preprocessing = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with the median
    ('scaler', StandardScaler())  # Standardize the data (Z-score normalization)
])

# Combine preprocessing and MLP model into a final pipeline
final_pipeline = Pipeline([
    ('preprocessing', preprocessing),  # Preprocessing steps
    ('model', MLPRegressor(random_state=42))  # MLPRegressor model
])

# Define parameter grid for hyperparameter tuning
param_grid = {
    'model__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'model__alpha': [0.0001, 0.001, 0.01],  # Regularization term
    'model__learning_rate_init': [0.01,0.05,0.1],
    'model__max_iter': [500, 1000, 1500]
}

# Use GridSearchCV for hyperparameter tuning and perform cross-validation during the process
grid_search = GridSearchCV(
    estimator=final_pipeline,
    param_grid=param_grid,
    cv=10,
    scoring={'r2': 'r2', 'rmse': make_scorer(mean_squared_error, squared=False)},
    refit='r2',
    n_jobs=2,
    verbose=2
)

# Fit the GridSearchCV with the training data
grid_search.fit(X_train, y_train)

# Output the best parameter combination
print("Best parameter combination:", grid_search.best_params_)

# Extract all cross-validation results
cv_results = grid_search.cv_results_

# Get the index of the best parameter combination
best_index = grid_search.best_index_

# Extract the 10-fold cross-validation results corresponding to the best parameter combination
best_r2_scores = [cv_results[f'split{i}_test_r2'][best_index] for i in range(10)]
best_rmse_scores = [cv_results[f'split{i}_test_rmse'][best_index] for i in range(10)]

# Compute the mean and standard deviation of the R² and RMSE scores from 10-fold cross-validation
best_r2_mean = np.mean(best_r2_scores)
best_r2_std = np.std(best_r2_scores)
best_rmse_mean = np.mean(best_rmse_scores)
best_rmse_std = np.std(best_rmse_scores)

print(f"\nMean ± Standard Deviation of R² from 10-fold cross-validation: {best_r2_mean:.4f} ± {best_r2_std:.4f}")
print(f"Mean ± Standard Deviation of RMSE from 10-fold cross-validation: {best_rmse_mean:.4f} ± {best_rmse_std:.4f}")

# Use the best parameter combination to predict on the test set
y_pred = grid_search.predict(X_test)

# Compute the R² and RMSE scores on the test set
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\nTest set R² score:", r2)
print("Test set RMSE score:", rmse)

##############ElasticNet
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet  

# Load data
features = pd.read_csv('alpha.csv')

# Separate features and target variable
features_true = features.drop(columns=["alpha"])
X = features_true
y = features["alpha"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log10 transform the target variable
y_train = np.log10(y_train).ravel()
y_test = np.log10(y_test).ravel()

# Perform one-hot encoding on the training and test sets separately to avoid data leakage
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Ensure training and test sets have the same columns (in case some categorical variables are missing in the test set)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Define the preprocessing pipeline (imputation and scaling)
preprocessing = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with the median
    ('scaler', StandardScaler())  # Standardize the data (Z-score normalization)
])

# Combine preprocessing and ElasticNet model into a final pipeline
final_pipeline = Pipeline([
    ('preprocessing', preprocessing),  # Preprocessing steps
    ('model', ElasticNet(random_state=42))  # ElasticNet model
])

# Define parameter grid for hyperparameter tuning
param_grid = {
    'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1],  # Regularization strength
    'model__l1_ratio': [0.2, 0.4, 0.6, 0.8, 1],  # Balance between L1 and L2 regularization
    'model__max_iter': [20000, 40000, 60000]  # Maximum number of iterations
}

# Use GridSearchCV for hyperparameter tuning and perform cross-validation during the process
grid_search = GridSearchCV(
    estimator=final_pipeline,
    param_grid=param_grid,
    cv=10,
    scoring={'r2': 'r2', 'rmse': make_scorer(mean_squared_error, squared=False)},
    refit='r2',
    n_jobs=-1,
    verbose=2
)

# Fit the GridSearchCV with the training data
grid_search.fit(X_train, y_train)

# Output the best parameter combination
print("Best parameter combination:", grid_search.best_params_)

# Extract all cross-validation results
cv_results = grid_search.cv_results_

# Get the index of the best parameter combination
best_index = grid_search.best_index_

# Extract the 10-fold cross-validation results corresponding to the best parameter combination
best_r2_scores = [cv_results[f'split{i}_test_r2'][best_index] for i in range(10)]
best_rmse_scores = [cv_results[f'split{i}_test_rmse'][best_index] for i in range(10)]

# Compute the mean and standard deviation of the R² and RMSE scores from 10-fold cross-validation
best_r2_mean = np.mean(best_r2_scores)
best_r2_std = np.std(best_r2_scores)
best_rmse_mean = np.mean(best_rmse_scores)
best_rmse_std = np.std(best_rmse_scores)

print(f"\nMean ± Standard Deviation of R² from 10-fold cross-validation: {best_r2_mean:.4f} ± {best_r2_std:.4f}")
print(f"Mean ± Standard Deviation of RMSE from 10-fold cross-validation: {best_rmse_mean:.4f} ± {best_rmse_std:.4f}")

# Use the best parameter combination to predict on the test set
y_pred = grid_search.predict(X_test)

# Compute the R² and RMSE scores on the test set
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\nTest set R² score:", r2)
print("Test set RMSE score:", rmse)

######################Linear regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression 

# Load data
features = pd.read_csv('alpha.csv')

# Separate features and target variable
features_true = features.drop(columns=["alpha"])
X = features_true
y = features["alpha"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log10 transform the target variable
y_train = np.log10(y_train).ravel()
y_test = np.log10(y_test).ravel()

# Perform one-hot encoding on the training and test sets separately to avoid data leakage
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Ensure training and test sets have the same columns (in case some categorical variables are missing in the test set)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Define the preprocessing pipeline (imputation and scaling)
preprocessing = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with the median
    ('scaler', StandardScaler())  # Standardize the data (Z-score normalization)
])

# Combine preprocessing and LinearRegression model into a final pipeline
final_pipeline = Pipeline([
    ('preprocessing', preprocessing),  # Preprocessing steps
    ('model', LinearRegression())  # LinearRegression model
])

# Perform 10-fold cross-validation using cross_validate
scoring = {'r2': 'r2', 'rmse': make_scorer(mean_squared_error, squared=False)}
cv_results = cross_validate(final_pipeline, X_train, y_train, cv=10, scoring=scoring, return_train_score=False)

# Output the mean and standard deviation of the R² and RMSE scores from 10-fold cross-validation
mean_r2 = np.mean(cv_results['test_r2'])
std_r2 = np.std(cv_results['test_r2'])
mean_rmse = np.mean(cv_results['test_rmse'])
std_rmse = np.std(cv_results['test_rmse'])

print(f"\nMean ± Standard Deviation of R² from 10-fold cross-validation: {mean_r2:.4f} ± {std_r2:.4f}")
print(f"Mean ± Standard Deviation of RMSE from 10-fold cross-validation: {mean_rmse:.4f} ± {std_rmse:.4f}")

# Train the final model on the full training set
final_pipeline.fit(X_train, y_train)

# Use the trained model to make predictions on the test set
y_pred = final_pipeline.predict(X_test)

# Compute the R² and RMSE scores on the test set
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\nTest set R² score:", r2)
print("Test set RMSE score:", rmse)