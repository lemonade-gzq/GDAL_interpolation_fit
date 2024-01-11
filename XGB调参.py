## 从sklearn库中导入网格调参函数
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np

data = pd.read_csv(r"E:\城市与区域生态\大熊猫和竹\道路对大熊猫栖息地的影响\道路距离分析\适宜性分析数据准备\xgb和maxent数据准备\01.csv",
                   engine='python', header=0)
data = data.iloc[:, :]
data.info()
X, Y = data[[x for x in data.columns if x != 'label' and x != 'FID']], data['label']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=5)
params = {'objective': 'reg:logistic', 'booster': 'gbtree', 'silent': 1}

param_test1 = {
    'max_depth': range(3, 25, 1),}
param_test2 = {
    "min_child_weight": range(1, 10, 1)
}
param_test3 = {'gamma': [0, 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4, 0.5,1]}
param_test4 = {
    'subsample': [i / 100.0 for i in range(75,90)]
}
param_test6 = {
    'colsample_bytree': [i / 100.0 for i in range(1, 101, 1)]
}
param_test7 = {
    'reg_alpha':[1e-5, 1e-2, 0, 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4, 0.5,1]
}
param_test8 = {
    'lambda':[1e-5, 1e-2, 0, 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4, 0.5,1]
}
param_test9 = {'n_estimators':range(1,100)}
gsearch1 = GridSearchCV(estimator=XGBClassifier(objective="binary:logistic",
                                                seed=1024,
                                                learning_rate=0.1,
                                                max_depth=13,
                                                min_child_weight=1,
                                                gamma=0,
                                                subsample=0.78,
                                                colsample_bytree=0.57,
                                                reg_alpha=1e-5,
                                                reg_lambda=0.09,
                                                n_estimators=44


),
                        param_grid=param_test9, scoring='roc_auc', cv=5, n_jobs=-1)
# max_depth=2,min_child_weight=5,gamma=0.55,subsample=0.8,colsample_bytree=0.3,alpha=0.31,reg_lambda=0.96,n_estimators=75
#  0, 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4, 0.5,1
gsearch1.fit(X_train, y_train)
print(gsearch1.cv_results_)
print(gsearch1.best_params_)
print(gsearch1.best_score_)
