#####################Random forest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
import shap  # 导入SHAP库

# 加载数据
features = pd.read_csv('Carbohydrate metabolism.csv')

# 分离特征和目标变量
features_true = features.drop(columns=["Carbohydrate metabolism"])
X = features_true
y = features["Carbohydrate metabolism"]

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 在训练集和测试集上分别进行独热编码
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# 确保训练集和测试集具有相同的列（以防某些分类变量在测试集中缺失）
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 在训练集上进行缺失值填补
X_train["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_train["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_train["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_train["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# 使用训练集中的中位数来填补测试集的缺失值
X_test["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_test["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_test["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_test["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# 对特征进行Z-score标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 使用训练集数据拟合scaler并转换训练集
X_test_scaled = scaler.transform(X_test)  # 直接使用训练集的scaler对测试集进行转换，避免数据泄漏

# 对目标变量进行log10转换
y_train = np.log10(y_train).ravel()
y_test = np.log10(y_test).ravel()

# 定义随机森林模型
model = RandomForestRegressor(random_state=42)

# 定义超参数网格
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [2, 4, 6, 8, 10]
}

# 使用GridSearchCV进行超参数调整，并在调整过程中执行交叉验证
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=10,
    scoring={'r2': 'r2', 'rmse': make_scorer(mean_squared_error, squared=False)},
    refit='r2',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train_scaled, y_train)

# 输出最佳参数组合
print("最佳参数组合:", grid_search.best_params_)

# 提取所有交叉验证结果
cv_results = grid_search.cv_results_

# 获取最佳参数组合的索引
best_index = grid_search.best_index_

# 提取与最佳参数组合对应的十折交叉验证结果
best_r2_scores = [cv_results[f'split{i}_test_r2'][best_index] for i in range(10)]
best_rmse_scores = [cv_results[f'split{i}_test_rmse'][best_index] for i in range(10)]

# 计算最佳参数组合的十折交叉验证R²和RMSE得分的平均值和标准差
best_r2_mean = np.mean(best_r2_scores)
best_r2_std = np.std(best_r2_scores)
best_rmse_mean = np.mean(best_rmse_scores)
best_rmse_std = np.std(best_rmse_scores)

print(f"\n最佳参数组合的十折交叉验证R²得分的平均值±标准差：{best_r2_mean:.4f} ± {best_r2_std:.4f}")
print(f"最佳参数组合的十折交叉验证RMSE得分的平均值±标准差：{best_rmse_mean:.4f} ± {best_rmse_std:.4f}")

# 直接使用最佳参数组合在测试集上进行预测
y_pred = grid_search.predict(X_test_scaled)

# 计算测试集的R²和RMSE得分
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\n测试集R²得分:", r2)
print("测试集RMSE得分:", rmse)

# 使用SHAP计算特征重要性
explainer = shap.TreeExplainer(grid_search.best_estimator_)
shap_values = explainer.shap_values(X_train_scaled)

# 绘制SHAP值的柱状图（特征重要性）
shap.summary_plot(shap_values, X_train, plot_type="bar")

####################xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import shap  # 导入SHAP库

# 加载数据
features = pd.read_csv('Carbohydrate metabolism.csv')

# 分离特征和目标变量
features_true = features.drop(columns=["Carbohydrate metabolism"])
X = features_true
y = features["Carbohydrate metabolism"]

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 在训练集和测试集上分别进行独热编码
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# 确保训练集和测试集具有相同的列（以防某些分类变量在测试集中缺失）
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 在训练集上进行缺失值填补
X_train["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_train["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_train["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_train["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# 使用训练集中的中位数来填补测试集的缺失值
X_test["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_test["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_test["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_test["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# 对特征进行Z-score标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 使用训练集数据拟合scaler并转换训练集
X_test_scaled = scaler.transform(X_test)  # 直接使用训练集的scaler对测试集进行转换，避免数据泄漏

# 对目标变量进行log10转换
y_train = np.log10(y_train).ravel()
y_test = np.log10(y_test).ravel()

# 定义XGBoost模型
model = XGBRegressor(random_state=42)

# 定义超参数网格
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, 50],
    'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25]
}

# 使用GridSearchCV进行超参数调整，并在调整过程中执行交叉验证
scoring = {'r2': 'r2', 'rmse': make_scorer(mean_squared_error, squared=False)}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring=scoring, refit='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# 输出最佳参数组合
print("最佳参数组合:", grid_search.best_params_)

# 提取所有交叉验证结果
cv_results = grid_search.cv_results_

# 获取最佳参数组合的索引
best_index = grid_search.best_index_

# 提取与最佳参数组合对应的十折交叉验证结果
best_r2_scores = [cv_results[f'split{i}_test_r2'][best_index] for i in range(10)]
best_rmse_scores = [cv_results[f'split{i}_test_rmse'][best_index] for i in range(10)]

# 计算最佳参数组合的十折交叉验证R²和RMSE得分的平均值和标准差
best_r2_mean = np.mean(best_r2_scores)
best_r2_std = np.std(best_r2_scores)
best_rmse_mean = np.mean(best_rmse_scores)
best_rmse_std = np.std(best_rmse_scores)

print(f"\n最佳参数组合的十折交叉验证R²得分的平均值±标准差：{best_r2_mean:.4f} ± {best_r2_std:.4f}")
print(f"最佳参数组合的十折交叉验证RMSE得分的平均值±标准差：{best_rmse_mean:.4f} ± {best_rmse_std:.4f}")

# 直接使用最佳参数组合在测试集上进行预测
y_pred = grid_search.predict(X_test_scaled)

# 计算测试集的R²和RMSE得分
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\n测试集R²得分:", r2)
print("测试集RMSE得分:", rmse)

# 计算 SHAP 值
explainer = shap.Explainer(grid_search.best_estimator_, X_train_scaled)  # 使用新的SHAP解释器
shap_values = explainer(X_train_scaled)  # 计算 SHAP 值

# 计算每个特征的平均绝对 SHAP 值
feature_importance = np.mean(np.abs(shap_values.values), axis=0)
feature_names = X_train.columns  # 在转换前使用原始的 DataFrame 获取列名

# 将特征名称和它们的重要性值打印出来
feature_importance_df = pd.DataFrame(list(zip(feature_names, feature_importance)), columns=['Feature', 'Importance'])
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("特征的重要性:")
print(feature_importance_df)

# 可视化所有特征的 SHAP 值（重要性）
shap.summary_plot(shap_values, X_train, plot_type="bar")

#########################MLP
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor  # 使用MLP回归
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
import shap  # 导入SHAP库

# 加载数据
features = pd.read_csv('Carbohydrate metabolism.csv')

# 分离特征和目标变量
features_true = features.drop(columns=["Carbohydrate metabolism"])
X = features_true
y = features["Carbohydrate metabolism"]

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 在训练集和测试集上分别进行独热编码
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# 确保训练集和测试集具有相同的列（以防某些分类变量在测试集中缺失）
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 在训练集上进行缺失值填补
X_train["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_train["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_train["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_train["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# 使用训练集中的中位数来填补测试集的缺失值
X_test["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_test["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_test["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_test["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# 对特征进行Z-score标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 使用训练集数据拟合scaler并转换训练集
X_test_scaled = scaler.transform(X_test)  # 直接使用训练集的scaler对测试集进行转换，避免数据泄漏

# 对目标变量进行log10转换
y_train = np.log10(y_train).ravel()
y_test = np.log10(y_test).ravel()

# 定义MLP回归模型
model = MLPRegressor(random_state=42)

# 定义超参数网格
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],  # 隐藏层配置
    'alpha': [0.0001, 0.001, 0.01],  # L2正则化参数
    'learning_rate_init': [0.01, 0.05, 0.1],
    'max_iter': [500, 1000, 1500]  # 学习率
}

# 使用GridSearchCV进行超参数调整，并在调整过程中执行交叉验证
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=10,
    scoring={'r2': 'r2', 'rmse': make_scorer(mean_squared_error, squared=False)},
    refit='r2',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train_scaled, y_train)

# 输出最佳参数组合
print("最佳参数组合:", grid_search.best_params_)

# 提取所有交叉验证结果
cv_results = grid_search.cv_results_

# 获取最佳参数组合的索引
best_index = grid_search.best_index_

# 提取与最佳参数组合对应的十折交叉验证结果
best_r2_scores = [cv_results[f'split{i}_test_r2'][best_index] for i in range(10)]
best_rmse_scores = [cv_results[f'split{i}_test_rmse'][best_index] for i in range(10)]

# 计算最佳参数组合的十折交叉验证R²和RMSE得分的平均值和标准差
best_r2_mean = np.mean(best_r2_scores)
best_r2_std = np.std(best_r2_scores)
best_rmse_mean = np.mean(best_rmse_scores)
best_rmse_std = np.std(best_rmse_scores)

print(f"\n最佳参数组合的十折交叉验证R²得分的平均值±标准差：{best_r2_mean:.4f} ± {best_r2_std:.4f}")
print(f"最佳参数组合的十折交叉验证RMSE得分的平均值±标准差：{best_rmse_mean:.4f} ± {best_rmse_std:.4f}")

# 直接使用最佳参数组合在测试集上进行预测
y_pred = grid_search.predict(X_test_scaled)

# 计算测试集的r2和rmse得分
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\n测试集r2得分:", r2)
print("测试集rmse得分:", rmse)

#############Elastic net
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet  # 使用ElasticNet回归
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
import shap  # 导入SHAP库

# 加载数据
features = pd.read_csv('Carbohydrate metabolism.csv')

# 分离特征和目标变量
features_true = features.drop(columns=["Carbohydrate metabolism"])
X = features_true
y = features["Carbohydrate metabolism"]

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 在训练集和测试集上分别进行独热编码
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# 确保训练集和测试集具有相同的列（以防某些分类变量在测试集中缺失）
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 在训练集上进行缺失值填补
X_train["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_train["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_train["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_train["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# 使用训练集中的中位数来填补测试集的缺失值
X_test["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_test["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_test["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_test["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# 对特征进行Z-score标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 使用训练集数据拟合scaler并转换训练集
X_test_scaled = scaler.transform(X_test)  # 直接使用训练集的scaler对测试集进行转换，避免数据泄漏

# 对目标变量进行log10转换
y_train = np.log10(y_train).ravel()
y_test = np.log10(y_test).ravel()

# 定义ElasticNet回归模型
model = ElasticNet()

# 定义超参数网格
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],  # 调整超参数Alpha的范围，避免过小的alpha值
    'l1_ratio': [0.2, 0.4, 0.6, 0.8, 1],  # L1 Ratio的范围，去除l1_ratio=0的情况
    'max_iter': [20000, 40000, 60000]
}

# 定义RMSE评分函数
scoring = {'r2': 'r2', 'rmse': make_scorer(mean_squared_error, squared=False)}

# 使用GridSearchCV进行超参数调整，并在调整过程中执行交叉验证
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring=scoring, refit='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# 输出最佳参数组合
print("最佳参数组合:", grid_search.best_params_)

# 提取所有交叉验证结果
cv_results = grid_search.cv_results_

# 获取最佳参数组合的索引
best_index = grid_search.best_index_

# 提取与最佳参数组合对应的十折交叉验证结果
best_r2_scores = [cv_results[f'split{i}_test_r2'][best_index] for i in range(10)]
best_rmse_scores = [cv_results[f'split{i}_test_rmse'][best_index] for i in range(10)]

# 计算最佳参数组合的十折交叉验证R²和RMSE得分的平均值和标准差
best_r2_mean = np.mean(best_r2_scores)
best_r2_std = np.std(best_r2_scores)
best_rmse_mean = np.mean(best_rmse_scores)
best_rmse_std = np.std(best_rmse_scores)

print(f"\n最佳参数组合的十折交叉验证R²得分的平均值±标准差：{best_r2_mean:.4f} ± {best_r2_std:.4f}")
print(f"最佳参数组合的十折交叉验证RMSE得分的平均值±标准差：{best_rmse_mean:.4f} ± {best_rmse_std:.4f}")

# 直接使用最佳参数组合在测试集上进行预测
y_pred = grid_search.predict(X_test_scaled)

# 计算测试集的R²和RMSE得分
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\n测试集R²得分:", r2)
print("测试集RMSE得分:", rmse)

#############Linear regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression  # 使用线性回归
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler

# 加载数据
features = pd.read_csv('Carbohydrate metabolism.csv')

# 分离特征和目标变量
features_true = features.drop(columns=["Carbohydrate metabolism"])
X = features_true
y = features["Carbohydrate metabolism"]

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 在训练集和测试集上分别进行独热编码
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# 确保训练集和测试集具有相同的列（以防某些分类变量在测试集中缺失）
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 在训练集上进行缺失值填补
X_train["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_train["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_train["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_train["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# 使用训练集中的中位数来填补测试集的缺失值
X_test["pH"].replace(np.nan, np.nanmedian(X_train["pH"]), inplace=True)
X_test["Light"].replace(np.nan, np.nanmedian(X_train["Light"]), inplace=True)
X_test["Concentration"].replace(np.nan, np.nanmedian(X_train["Concentration"]), inplace=True)
X_test["Size"].replace(np.nan, np.nanmedian(X_train["Size"]), inplace=True)

# 对特征进行Z-score标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 使用训练集数据拟合scaler并转换训练集
X_test_scaled = scaler.transform(X_test)  # 直接使用训练集的scaler对测试集进行转换，避免数据泄漏

# 对目标变量进行log10转换
y_train = np.log10(y_train).ravel()
y_test = np.log10(y_test).ravel()

# 定义线性回归模型
model = LinearRegression()

# 定义自定义的RMSE评分函数
rmse_scorer = make_scorer(mean_squared_error, squared=False)

# 使用cross_val_score进行十折交叉验证，计算R^2和RMSE
r2_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='r2')
rmse_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring=rmse_scorer)

# 计算交叉验证的R^2平均值和标准差
mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)

# 计算交叉验证的RMSE平均值和标准差
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)

print(f"十折交叉验证的平均R^2得分: {mean_r2:.4f} ± {std_r2:.4f}")
print(f"十折交叉验证的平均RMSE得分: {mean_rmse:.4f} ± {std_rmse:.4f}")

# 在整个训练集上拟合模型
model.fit(X_train_scaled, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test_scaled)

# 计算测试集的R^2和RMSE得分
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\n测试集R^2得分:", r2)
print("测试集RMSE得分:", rmse)



