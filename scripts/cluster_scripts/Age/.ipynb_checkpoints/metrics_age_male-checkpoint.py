import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(1, '../../')
from ml_functions import *

def prepare_df(path):
    df = pd.read_csv(path)
    labels = df.Age
    df.drop(list(df)[:2], axis = 1, inplace = True)
    return df, labels
    
train_df_male, train_labels_male = prepare_df('../../../steps/train_df_male_age.csv')
test_df_male, test_labels_male = prepare_df('../../../steps/test_df_male_age.csv')

from sklearn.preprocessing import StandardScaler

scaler_male = StandardScaler().fit(train_df_male)

train_df_male_stand = scaler_male.transform(train_df_male)
test_df_male_stand = scaler_male.transform(test_df_male)

from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, RidgeCV, LassoCV, Ridge, Lasso
from sklearn.svm import LinearSVR, SVR
from xgboost import XGBRegressor

#%%
# PLS
pls_mod = PLSRegression(n_components = 6, scale = False) 

# Ridge
ridge_mod = Ridge(alpha = 60) 

# Lasso
lasso_mod = Lasso(alpha = 0.385) 

# SVM 
svm_mod = SVR(kernel='linear', C = 0.1, epsilon = 2.1)

# RandomForest
#rfmod = RandomForestRegressor(bootstrap = True, max_depth = 6, max_features = None, 
#                              min_samples_leaf = 5, min_samples_split = 5, n_estimators = 1000)
rfmod = RandomForestRegressor(bootstrap = True, max_depth = 4, max_features = 0.8, 
                              min_samples_leaf = 5, min_samples_split = 5, n_estimators = 1000)

# Gradient Boosting
grboost = GradientBoostingRegressor(learning_rate=0.01, max_depth = 3, max_features = 1, min_samples_leaf = 25, 
                                    min_samples_split = 30, n_estimators = 1000, subsample = 0.55)

# Adaptive Boosting
adaboost = AdaBoostRegressor(learning_rate = 0.1, n_estimators = 1000)

# XGBoost
xgbmod = XGBRegressor(colsample_bynode = 0.55, colsample_bytree = 0.55, gamma = 0, learning_rate = 1, max_depth = 2, 
                      n_estimators = 1000, objective = 'reg:squarederror', reg_lambda = 1000, subsample = 1)

#%%

metrics = ['r2', 'mse', 'mae']

metric_df = pd.DataFrame()

for metric in metrics:
    metric_temp_df = cv_test_wrapper(train_df_male_stand, train_labels_male, test_df_male_stand, test_labels_male,             
                                     stable_models = [pls_mod, ridge_mod, lasso_mod, svm_mod],
                                     unstable_models = [rfmod, grboost, adaboost, xgbmod], 
                                     scoring=metric, n_max = 20)
    
    metric_temp_df['Metric'] = metric
    metric_df = pd.concat([metric_df, metric_temp_df])
    
metric_df.to_csv('../../../results/Age_prediction/age_male_metrics.csv')
