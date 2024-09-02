######################Randdom forest
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

# 读取数据
auc = pd.read_csv('beneficial bacteria.csv')

# 分离特征和目标变量
X = auc.drop(columns=["Exposure"])
y = auc["Exposure"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用Pipeline将特征选择和模型训练结合起来
pipe = Pipeline([
    ('feature_selection', SelectFromModel(Lasso(alpha=0.00001, max_iter=50000))),  # Lasso用于特征选择
    ('rf', RandomForestClassifier(random_state=42))  # 随机森林分类器
])

# 定义超参数网格
param_grid = {
    'rf__n_estimators': [100, 200, 300, 400, 500],  # 随机森林树的数量
    'rf__max_depth': [10, 20, 30, 40, 50],  # 树的最大深度
    'rf__min_samples_split': [2, 4, 6, 8, 10]  # 内部节点再划分所需的最小样本数
}

# 使用GridSearchCV进行超参数调整，使用十折交叉验证
grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数组合
print("Best Parameters from GridSearchCV:", grid_search.best_params_)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 提取与最佳参数组合对应的十折交叉验证结果
cv_results = grid_search.cv_results_
best_index = grid_search.best_index_

# 提取与最佳参数组合对应的10折交叉验证AUC结果
best_auc_scores = [cv_results[f'split{i}_test_score'][best_index] for i in range(10)]

# 计算最佳参数组合的十折交叉验证AUC得分的平均值和标准差
best_auc_mean = np.mean(best_auc_scores)
best_auc_std = np.std(best_auc_scores)

print(f"\n最佳参数组合的十折交叉验证AUC得分的平均值±标准差：{best_auc_mean:.4f} ± {best_auc_std:.4f}")

# 使用最佳模型在测试集上进行预测
y_probs_test = best_model.predict_proba(X_test)[:, 1]

# 计算测试集的AUC值
auc_score_rf_la = roc_auc_score(y_test, y_probs_test)
print(f"Test AUC Score: {auc_score_rf_la:.4f}")

# 计算FPR和TPR
fpr_rf_la, tpr_rf_la, thresholds = roc_curve(y_test, y_probs_test)

# 绘制ROC曲线
plt.figure(figsize=(6, 6))
plt.plot(fpr_rf_la, tpr_rf_la, label='ROC curve (AUC = %0.2f)' % auc_score_rf_la)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.axis('square')
plt.show()

##############xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from xgboost import XGBClassifier  # 导入XGBoost分类器

# 读取数据
auc = pd.read_csv('beneficial bacteria.csv')

# 分离特征和目标变量
X = auc.drop(columns=["Exposure"])
y = auc["Exposure"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用Pipeline将特征选择和模型训练结合起来
pipe = Pipeline([
    ('feature_selection', SelectFromModel(Lasso(alpha=0.00001, max_iter=50000))),  # Lasso用于特征选择
    ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))  # 使用XGBoost分类器
])

# 定义超参数网格
param_grid = {
    'xgb__n_estimators': [100, 200, 300, 400, 500],  # XGBoost树的数量
    'xgb__max_depth': [10, 20, 30, 40, 50],  # 树的最大深度
    'xgb__learning_rate': [0.05, 0.1, 0.15, 0.2,0.25],  # 学习率
}

# 使用GridSearchCV进行超参数调整，使用十折交叉验证
grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数组合
print("Best Parameters from GridSearchCV:", grid_search.best_params_)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 提取与最佳参数组合对应的十折交叉验证结果
cv_results = grid_search.cv_results_
best_index = grid_search.best_index_

# 提取与最佳参数组合对应的10折交叉验证AUC结果
best_auc_scores = [cv_results[f'split{i}_test_score'][best_index] for i in range(10)]

# 计算最佳参数组合的十折交叉验证AUC得分的平均值和标准差
best_auc_mean = np.mean(best_auc_scores)
best_auc_std = np.std(best_auc_scores)

print(f"\n最佳参数组合的十折交叉验证AUC得分的平均值±标准差：{best_auc_mean:.4f} ± {best_auc_std:.4f}")

# 使用最佳模型在测试集上进行预测
y_probs_test = best_model.predict_proba(X_test)[:, 1]

# 计算测试集的AUC值
auc_score_rf_la = roc_auc_score(y_test, y_probs_test)
print(f"Test AUC Score: {auc_score_rf_la:.4f}")

# 计算FPR和TPR
fpr_rf_la, tpr_rf_la, thresholds = roc_curve(y_test, y_probs_test)

# 绘制ROC曲线
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
from sklearn.neural_network import MLPClassifier  # 导入MLP分类器
import matplotlib.pyplot as plt

# 读取数据
auc = pd.read_csv('beneficial bacteria.csv')

# 分离特征和目标变量
X = auc.drop(columns=["Exposure"])
y = auc["Exposure"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用Pipeline将特征选择和模型训练结合起来
pipe = Pipeline([
    ('feature_selection', SelectFromModel(Lasso(alpha=0.00001, max_iter=50000))),  # Lasso用于特征选择
    ('mlp', MLPClassifier(random_state=42, max_iter=5000))  # 使用MLP分类器
])

# 定义超参数网格
param_grid = {
    'mlp__hidden_layer_sizes': [(50,), (100,), (100, 100), (50, 50)],  # 隐藏层大小
    'mlp__learning_rate_init': [0.02, 0.04, 0.06, 0.08,0.1],  # 初始学习率
    'mlp__alpha': [0.0001, 0.001, 0.01],  # L2正则化参数
}

# 使用GridSearchCV进行超参数调整，使用十折交叉验证
grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数组合
print("Best Parameters from GridSearchCV:", grid_search.best_params_)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 提取与最佳参数组合对应的十折交叉验证结果
cv_results = grid_search.cv_results_
best_index = grid_search.best_index_

# 提取与最佳参数组合对应的10折交叉验证AUC结果
best_auc_scores = [cv_results[f'split{i}_test_score'][best_index] for i in range(10)]

# 计算最佳参数组合的十折交叉验证AUC得分的平均值和标准差
best_auc_mean = np.mean(best_auc_scores)
best_auc_std = np.std(best_auc_scores)

print(f"\n最佳参数组合的十折交叉验证AUC得分的平均值±标准差：{best_auc_mean:.4f} ± {best_auc_std:.4f}")

# 使用最佳模型在测试集上进行预测
y_probs_test = best_model.predict_proba(X_test)[:, 1]

# 计算测试集的AUC值
auc_score_rf_la = roc_auc_score(y_test, y_probs_test)
print(f"Test AUC Score: {auc_score_rf_la:.4f}")

# 计算FPR和TPR
fpr_rf_la, tpr_rf_la, thresholds = roc_curve(y_test, y_probs_test)

# 绘制ROC曲线
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
from sklearn.svm import SVC  # 导入SVM分类器
import matplotlib.pyplot as plt

# 读取数据
auc = pd.read_csv('beneficial bacteria.csv')

# 分离特征和目标变量
X = auc.drop(columns=["Exposure"])
y = auc["Exposure"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用Pipeline将特征选择和模型训练结合起来
pipe = Pipeline([
    ('feature_selection', SelectFromModel(Lasso(alpha=0.00001, max_iter=50000))),  # Lasso用于特征选择
    ('svm', SVC(probability=True, random_state=42))  # 使用SVM分类器
])

# 定义超参数网格
param_grid = {
    'svm__C': [0.01, 0.1, 1, 10],  # 正则化参数
    'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # 核函数类型
    'svm__gamma': [0.0001, 0.001,0.01],  # 核函数系数
}

# 使用GridSearchCV进行超参数调整，使用十折交叉验证
grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=10, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数组合
print("Best Parameters from GridSearchCV:", grid_search.best_params_)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 提取与最佳参数组合对应的十折交叉验证结果
cv_results = grid_search.cv_results_
best_index = grid_search.best_index_

# 提取与最佳参数组合对应的10折交叉验证AUC结果
best_auc_scores = [cv_results[f'split{i}_test_score'][best_index] for i in range(10)]

# 计算最佳参数组合的十折交叉验证AUC得分的平均值和标准差
best_auc_mean = np.mean(best_auc_scores)
best_auc_std = np.std(best_auc_scores)

print(f"\n最佳参数组合的十折交叉验证AUC得分的平均值±标准差：{best_auc_mean:.4f} ± {best_auc_std:.4f}")

# 使用最佳模型在测试集上进行预测
y_probs_test = best_model.predict_proba(X_test)[:, 1]

# 计算测试集的AUC值
auc_score_rf_la = roc_auc_score(y_test, y_probs_test)
print(f"Test AUC Score: {auc_score_rf_la:.4f}")

# 计算FPR和TPR
fpr_rf_la, tpr_rf_la, thresholds = roc_curve(y_test, y_probs_test)

# 绘制ROC曲线
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
from sklearn.naive_bayes import GaussianNB  # 导入Naive Bayes分类器
import matplotlib.pyplot as plt

# 读取数据
auc = pd.read_csv('beneficial bacteria.csv')

# 分离特征和目标变量
X = auc.drop(columns=["Exposure"])
y = auc["Exposure"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用Pipeline将特征选择和模型训练结合起来
pipe = Pipeline([
    ('feature_selection', SelectFromModel(Lasso(alpha=0.00001, max_iter=50000))),  # Lasso用于特征选择
    ('nb', GaussianNB())  # 使用Naive Bayes分类器
])

# 使用十折交叉验证来评估模型性能
cv_scores = cross_val_score(pipe, X_train, y_train, cv=10, scoring='roc_auc')

# 输出十折交叉验证的AUC得分的平均值和标准差
cv_mean_auc = np.mean(cv_scores)
cv_std_auc = np.std(cv_scores)
print(f"十折交叉验证AUC得分的平均值±标准差：{cv_mean_auc:.4f} ± {cv_std_auc:.4f}")

# 重新在整个训练集上拟合模型
pipe.fit(X_train, y_train)

# 在测试集上进行预测
y_probs_test = pipe.predict_proba(X_test)[:, 1]

# 计算测试集的AUC值
auc_score_nb = roc_auc_score(y_test, y_probs_test)
print(f"Test AUC Score: {auc_score_nb:.4f}")

# 计算FPR和TPR
fpr_nb, tpr_nb, thresholds = roc_curve(y_test, y_probs_test)

# 绘制ROC曲线
plt.figure(figsize=(6, 6))
plt.plot(fpr_nb, tpr_nb, label='ROC curve (AUC = %0.2f)' % auc_score_nb)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.axis('square')
plt.show()

