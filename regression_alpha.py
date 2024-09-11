#####################Random Forest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler 

# Load data
features = pd.read_csv('alpha.csv')

# Separate features and target variable
features_true = features.drop(columns=["alpha"])
X = features_true
y = features["alpha"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform one-hot encoding on the training and test sets separately to avoid data leakage
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Ensure training and test sets have the same columns (in case some categorical variables are missing in the test set)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Impute missing values in the training set
X_train["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_train["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_train["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_train["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# Use the median values from the training set to impute missing values in the test set to avoid data leakage
X_test["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_test["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_test["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_test["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# Perform Z-score normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit the scaler on the training set and transform the training set
X_test_scaled = scaler.transform(X_test)  # Use the same scaler to transform the test set, avoiding data leakage

# Define Random Forest model
model = RandomForestRegressor(random_state=42)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [2, 4, 6, 8, 10]
}

# Use GridSearchCV for hyperparameter tuning and perform cross-validation during the process
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=10,
    scoring={'r2': 'r2', 'rmse': make_scorer(mean_squared_error, squared=False)},
    refit='r2',
    n_jobs=2,
    verbose=2
)
grid_search.fit(X_train_scaled, y_train)

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
y_pred = grid_search.predict(X_test_scaled)

# Compute the R² and RMSE scores on the test set
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\nTest set R² score:", r2)
print("Test set RMSE score:", rmse)

###############XGBoost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Load data
features = pd.read_csv('alpha.csv')

# Separate features and target variable
features_true = features.drop(columns=["alpha"])
X = features_true
y = features["alpha"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform one-hot encoding on the training and test sets separately
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Ensure training and test sets have the same columns (in case some categorical variables are missing in the test set)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Impute missing values in the training set
X_train["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_train["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_train["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_train["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# Use the median values from the training set to impute missing values in the test set
X_test["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_test["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_test["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_test["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# Perform Z-score normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit the scaler on the training set and transform the training set
X_test_scaled = scaler.transform(X_test)  # Use the same scaler to transform the test set, avoiding data leakage

# Define XGBoost model
model = XGBRegressor(random_state=42)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, 50],
    'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25]
}

# Use GridSearchCV for hyperparameter tuning and perform cross-validation during the process
scoring = {'r2': 'r2', 'rmse': make_scorer(mean_squared_error, squared=False)}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring=scoring, refit='r2', n_jobs=2, verbose=2)
grid_search.fit(X_train_scaled, y_train)

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
y_pred = grid_search.predict(X_test_scaled)

# Compute the R² and RMSE scores on the test set
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\nTest set R² score:", r2)
print("Test set RMSE score:", rmse)

###############MLP
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor  # Use MLP for regression
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
import shap  # Import SHAP library

# Load data
features = pd.read_csv('alpha.csv')

# Separate features and target variable
features_true = features.drop(columns=["alpha"])
X = features_true
y = features["alpha"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform one-hot encoding on the training and test sets separately
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Ensure training and test sets have the same columns (in case some categorical variables are missing in the test set)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Impute missing values in the training set
X_train["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_train["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_train["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_train["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# Use the median values from the training set to impute missing values in the test set
X_test["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_test["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_test["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_test["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# Perform Z-score normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit the scaler on the training set and transform the training set
X_test_scaled = scaler.transform(X_test)  # Use the same scaler to transform the test set, avoiding data leakage

# Define MLP regression model
model = MLPRegressor(random_state=42)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],  # Hidden layer configuration
    'alpha': [0.0001, 0.001, 0.01],  # L2 regularization parameter
    'learning_rate_init': [0.01, 0.05, 0.1],
    'max_iter': [500, 1000, 1500]  # Max iterations
}

# Use GridSearchCV for hyperparameter tuning and perform cross-validation during the process
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=10,
    scoring={'r2': 'r2', 'rmse': make_scorer(mean_squared_error, squared=False)},
    refit='r2',
    n_jobs=2,
    verbose=2
)
grid_search.fit(X_train_scaled, y_train)

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
y_pred = grid_search.predict(X_test_scaled)

# Compute the R² and RMSE scores on the test set
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\nTest set R² score:", r2)
print("Test set RMSE score:", rmse)

#############Elastic Net
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet  # Use ElasticNet for regression
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
import shap  # Import SHAP library

# Load data
features = pd.read_csv('alpha.csv')

# Separate features and target variable
features_true = features.drop(columns=["alpha"])
X = features_true
y = features["alpha"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform one-hot encoding on the training and test sets separately
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Ensure training and test sets have the same columns (in case some categorical variables are missing in the test set)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Impute missing values in the training set
X_train["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_train["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_train["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_train["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# Use the median values from the training set to impute missing values in the test set
X_test["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_test["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_test["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_test["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# Perform Z-score normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit the scaler on the training set and transform the training set
X_test_scaled = scaler.transform(X_test)  # Use the same scaler to transform the test set, avoiding data leakage

# Define ElasticNet regression model
model = ElasticNet()

# Define parameter grid for hyperparameter tuning
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],  # Adjust alpha parameter range
    'l1_ratio': [0.2, 0.4, 0.6, 0.8, 1],  # Range for L1 ratio
    'max_iter': [20000, 40000, 60000]
}

# Define RMSE scoring function
scoring = {'r2': 'r2', 'rmse': make_scorer(mean_squared_error, squared=False)}

# Use GridSearchCV for hyperparameter tuning and perform cross-validation during the process
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring=scoring, refit='r2', n_jobs=2, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Output the best parameter combination
print("Best parameter combination:", grid_search.best_params_)

# Extract all cross-validation results
cv_results = grid_search.cv_results_

# Get the index of the best parameter combination
best_index = grid_search.best_index_

# Extract the 10-fold cross-validation results corresponding to the best parameter combination
best_r2_scores = [cv_results[f'split{i}_test_r2'][best_index] for i in range(10)]
best_rmse_scores = [cv_results[f'split{i}_test_rmse'][best_index] for i in range(10)]

# Compute mean and standard deviation of R² and RMSE scores across the 10 folds
best_r2_mean = np.mean(best_r2_scores)
best_r2_std = np.std(best_r2_scores)
best_rmse_mean = np.mean(best_rmse_scores)
best_rmse_std = np.std(best_rmse_scores)

print(f"\nMean ± Standard Deviation of R² from 10-fold cross-validation: {best_r2_mean:.4f} ± {best_r2_std:.4f}")
print(f"Mean ± Standard Deviation of RMSE from 10-fold cross-validation: {best_rmse_mean:.4f} ± {best_rmse_std:.4f}")

# Predict on the test set using the best hyperparameters
y_pred = grid_search.predict(X_test_scaled)

# Compute R² and RMSE on the test set
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\nTest set R² score:", r2)
print("Test set RMSE score:", rmse)

#############Linear Regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression  # Use Linear Regression
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler

# Load data
features = pd.read_csv('alpha.csv')

# Separate features and target variable
features_true = features.drop(columns=["alpha"])
X = features_true
y = features["alpha"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform one-hot encoding on the training and test sets separately
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Ensure training and test sets have the same columns (in case some categorical variables are missing in the test set)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Impute missing values in the training set
X_train["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_train["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_train["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_train["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# Use the median values from the training set to impute missing values in the test set
X_test["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_test["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_test["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_test["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# Perform Z-score normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit the scaler on the training set and transform the training set
X_test_scaled = scaler.transform(X_test)  # Use the same scaler to transform the test set, avoiding data leakage

# Define Linear Regression model
model = LinearRegression()

# Define a custom RMSE scoring function
rmse_scorer = make_scorer(mean_squared_error, squared=False)

# Use cross_val_score for 10-fold cross-validation, calculating R² and RMSE
r2_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='r2')
rmse_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring=rmse_scorer)

# Compute the mean and standard deviation of R² from cross-validation
mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)

# Compute the mean and standard deviation of RMSE from cross-validation
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)

print(f"Mean R² score from 10-fold cross-validation: {mean_r2:.4f} ± {std_r2:.4f}")
print(f"Mean RMSE score from 10-fold cross-validation: {mean_rmse:.4f} ± {std_rmse:.4f}")

# Fit the model on the entire training set
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Compute R² and RMSE scores on the test set
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\nTest set R² score:", r2)
print("Test set RMSE score:", rmse)
