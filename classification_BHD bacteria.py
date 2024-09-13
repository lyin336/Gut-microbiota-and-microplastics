######################Random Forest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
auc = pd.read_csv('beneficial bacteria.csv')

# Separate features and target variable
X = auc.drop(columns=["Exposure"])
y = auc["Exposure"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Pipeline to combine feature selection and model training
pipe = Pipeline([
    ('feature_selection', SelectFromModel(Lasso(alpha=0.00001, max_iter=50000))),  # Lasso for feature selection
    ('rf', RandomForestClassifier(random_state=42))  # Random forest classifier
])

# Define the parameter grid
param_grid = {
    'rf__n_estimators': [100, 200, 300, 400, 500],  # Number of trees in the forest
    'rf__max_depth': [10, 20, 30, 40, 50],  # Maximum depth of the tree
    'rf__min_samples_split': [2, 4, 6, 8, 10]  # Minimum number of samples required to split an internal node
}

# Use GridSearchCV for hyperparameter tuning, with 10-fold cross-validation
grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Output the best parameter combination
print("Best Parameters from GridSearchCV:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Extract the 10-fold cross-validation results for the best parameter combination
cv_results = grid_search.cv_results_
best_index = grid_search.best_index_

# Extract the 10-fold cross-validation AUC scores for the best parameter combination
best_auc_scores = [cv_results[f'split{i}_test_score'][best_index] for i in range(10)]

# Calculate the mean and standard deviation of the 10-fold cross-validation AUC scores
best_auc_mean = np.mean(best_auc_scores)
best_auc_std = np.std(best_auc_scores)

print(f"\nMean ± Standard Deviation of the 10-fold cross-validation AUC score for the best parameter combination: {best_auc_mean:.4f} ± {best_auc_std:.4f}")

# Use the best model to make predictions on the test set
y_probs_test = best_model.predict_proba(X_test)[:, 1]

# Calculate the AUC score on the test set
auc_score_rf_la = roc_auc_score(y_test, y_probs_test)
print(f"Test AUC Score: {auc_score_rf_la:.4f}")

# Calculate FPR and TPR
fpr_rf_la, tpr_rf_la, thresholds = roc_curve(y_test, y_probs_test)

# Plot the ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr_rf_la, tpr_rf_la, label='ROC curve (AUC = %0.2f)' % auc_score_rf_la)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.axis('square')
plt.show()

##############XGBoost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from xgboost import XGBClassifier  # Import XGBoost classifier

# Load data
auc = pd.read_csv('beneficial bacteria.csv')

# Separate features and target variable
X = auc.drop(columns=["Exposure"])
y = auc["Exposure"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Pipeline to combine feature selection and model training
pipe = Pipeline([
    ('feature_selection', SelectFromModel(Lasso(alpha=0.00001, max_iter=50000))),  # Lasso for feature selection
    ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))  # Use XGBoost classifier
])

# Define the parameter grid
param_grid = {
    'xgb__n_estimators': [100, 200, 300, 400, 500],  # Number of trees in XGBoost
    'xgb__max_depth': [10, 20, 30, 40, 50],  # Maximum depth of the tree
    'xgb__learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25],  # Learning rate
}

# Use GridSearchCV for hyperparameter tuning, with 10-fold cross-validation
grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Output the best parameter combination
print("Best Parameters from GridSearchCV:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Extract the 10-fold cross-validation results for the best parameter combination
cv_results = grid_search.cv_results_
best_index = grid_search.best_index_

# Extract the 10-fold cross-validation AUC scores for the best parameter combination
best_auc_scores = [cv_results[f'split{i}_test_score'][best_index] for i in range(10)]

# Calculate the mean and standard deviation of the 10-fold cross-validation AUC scores
best_auc_mean = np.mean(best_auc_scores)
best_auc_std = np.std(best_auc_scores)

print(f"\nMean ± Standard Deviation of the 10-fold cross-validation AUC score for the best parameter combination: {best_auc_mean:.4f} ± {best_auc_std:.4f}")

# Use the best model to make predictions on the test set
y_probs_test = best_model.predict_proba(X_test)[:, 1]

# Calculate the AUC score on the test set
auc_score_rf_la = roc_auc_score(y_test, y_probs_test)
print(f"Test AUC Score: {auc_score_rf_la:.4f}")

# Calculate FPR and TPR
fpr_rf_la, tpr_rf_la, thresholds = roc_curve(y_test, y_probs_test)

# Plot the ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr_rf_la, tpr_rf_la, label='ROC curve (AUC = %0.2f)' % auc_score_rf_la)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.axis('square')
plt.show()

###########MLP
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier  # Import MLP classifier
import matplotlib.pyplot as plt

# Load data
auc = pd.read_csv('beneficial bacteria.csv')

# Separate features and target variable
X = auc.drop(columns=["Exposure"])
y = auc["Exposure"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Pipeline to combine feature selection and model training
pipe = Pipeline([
    ('feature_selection', SelectFromModel(Lasso(alpha=0.00001, max_iter=50000))),  # Lasso for feature selection
    ('mlp', MLPClassifier(random_state=42, max_iter=5000))  # Use MLP classifier
])

# Define the parameter grid
param_grid = {
    'mlp__hidden_layer_sizes': [(50,), (100,), (100, 100), (50, 50)],  # Hidden layer sizes
    'mlp__learning_rate_init': [0.02, 0.04, 0.06, 0.08, 0.1],  # Initial learning rate
    'mlp__alpha': [0.0001, 0.001, 0.01],  # L2 regularization parameter
}

# Use GridSearchCV for hyperparameter tuning, with 10-fold cross-validation
grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Output the best parameter combination
print("Best Parameters from GridSearchCV:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Extract the 10-fold cross-validation results for the best parameter combination
cv_results = grid_search.cv_results_
best_index = grid_search.best_index_

# Extract the 10-fold cross-validation AUC scores for the best parameter combination
best_auc_scores = [cv_results[f'split{i}_test_score'][best_index] for i in range(10)]

# Calculate the mean and standard deviation of the 10-fold cross-validation AUC scores
best_auc_mean = np.mean(best_auc_scores)
best_auc_std = np.std(best_auc_scores)

print(f"\nMean ± Standard Deviation of the 10-fold cross-validation AUC score for the best parameter combination: {best_auc_mean:.4f} ± {best_auc_std:.4f}")

# Use the best model to make predictions on the test set
y_probs_test = best_model.predict_proba(X_test)[:, 1]

# Calculate the AUC score on the test set
auc_score_rf_la = roc_auc_score(y_test, y_probs_test)
print(f"Test AUC Score: {auc_score_rf_la:.4f}")

# Calculate FPR and TPR
fpr_rf_la, tpr_rf_la, thresholds = roc_curve(y_test, y_probs_test)

# Plot the ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr_rf_la, tpr_rf_la, label='ROC curve (AUC = %0.2f)' % auc_score_rf_la)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.axis('square')
plt.show()

##########SVM
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # Import SVM classifier
import matplotlib.pyplot as plt

# Load data
auc = pd.read_csv('beneficial bacteria.csv')

# Separate features and target variable
X = auc.drop(columns=["Exposure"])
y = auc["Exposure"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Pipeline to combine feature selection and model training
pipe = Pipeline([
    ('feature_selection', SelectFromModel(Lasso(alpha=0.00001, max_iter=50000))),  # Lasso for feature selection
    ('svm', SVC(probability=True, random_state=42))  # Use SVM classifier
])

# Define the parameter grid
param_grid = {
    'svm__C': [0.01, 0.1, 1, 10],  # Regularization parameter
    'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel type
    'svm__gamma': [0.0001, 0.001, 0.01],  # Kernel coefficient
}

# Use GridSearchCV for hyperparameter tuning, with 10-fold cross-validation
grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Output the best parameter combination
print("Best Parameters from GridSearchCV:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Extract the 10-fold cross-validation results for the best parameter combination
cv_results = grid_search.cv_results_
best_index = grid_search.best_index_

# Extract the 10-fold cross-validation AUC scores for the best parameter combination
best_auc_scores = [cv_results[f'split{i}_test_score'][best_index] for i in range(10)]

# Calculate the mean and standard deviation of the 10-fold cross-validation AUC scores
best_auc_mean = np.mean(best_auc_scores)
best_auc_std = np.std(best_auc_scores)

print(f"\nMean ± Standard Deviation of the 10-fold cross-validation AUC score for the best parameter combination: {best_auc_mean:.4f} ± {best_auc_std:.4f}")

# Use the best model to make predictions on the test set
y_probs_test = best_model.predict_proba(X_test)[:, 1]

# Calculate the AUC score on the test set
auc_score_rf_la = roc_auc_score(y_test, y_probs_test)
print(f"Test AUC Score: {auc_score_rf_la:.4f}")

# Calculate FPR and TPR
fpr_rf_la, tpr_rf_la, thresholds = roc_curve(y_test, y_probs_test)

# Plot the ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr_rf_la, tpr_rf_la, label='ROC curve (AUC = %0.2f)' % auc_score_rf_la)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.axis('square')
plt.show()

###########Naive Bayes
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB  # Import Naive Bayes classifier
import matplotlib.pyplot as plt

# Load data
auc = pd.read_csv('beneficial bacteria.csv')

# Separate features and target variable
X = auc.drop(columns=["Exposure"])
y = auc["Exposure"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Pipeline to combine feature selection and model training
pipe = Pipeline([
    ('feature_selection', SelectFromModel(Lasso(alpha=0.00001, max_iter=50000))),  # Lasso for feature selection
    ('nb', GaussianNB())  # Use Naive Bayes classifier
])

# Use 10-fold cross-validation to evaluate model performance
cv_scores = cross_val_score(pipe, X_train, y_train, cv=10, scoring='roc_auc')

# Output the mean and standard deviation of the 10-fold cross-validation AUC scores
cv_mean_auc = np.mean(cv_scores)
cv_std_auc = np.std(cv_scores)
print(f"Mean ± Standard Deviation of the 10-fold cross-validation AUC score: {cv_mean_auc:.4f} ± {cv_std_auc:.4f}")

# Refit the model on the entire training set
pipe.fit(X_train, y_train)

# Make predictions on the test set
y_probs_test = pipe.predict_proba(X_test)[:, 1]

# Calculate the AUC score on the test set
auc_score_nb = roc_auc_score(y_test, y_probs_test)
print(f"Test AUC Score: {auc_score_nb:.4f}")

# Calculate FPR and TPR
fpr_nb, tpr_nb, thresholds = roc_curve(y_test, y_probs_test)

# Plot the ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr_nb, tpr_nb, label='ROC curve (AUC = %0.2f)' % auc_score_nb)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.axis('square')
plt.show()
