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
    
train_df_female, train_labels_female = prepare_df('../../../steps/train_df_female_age.csv')
test_df_female, test_labels_female = prepare_df('../../../steps/test_df_female_age.csv')

from sklearn.preprocessing import StandardScaler

scaler_female = StandardScaler().fit(train_df_female)

train_df_female_stand = scaler_female.transform(train_df_female)
test_df_female_stand = scaler_female.transform(test_df_female)

from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, RidgeCV, LassoCV, Ridge, Lasso
from sklearn.svm import LinearSVR, SVR
from xgboost import XGBRegressor

#%%
# PLS
pls_modf = PLSRegression(n_components = 4, scale = False) # this is a bit better with unskewed variables

# Ridge
ridge_modf = Ridge(alpha = 282)

# Lasso
lasso_modf = Lasso(alpha = 1.24)

# SVM
svm_modf = SVR(kernel='linear', C = 0.1, epsilon = 6)

# RandomForest
rfmodf = RandomForestRegressor(bootstrap = True, max_depth = 8, max_features = 0.5, 
                               min_samples_leaf = 5, min_samples_split = 10, n_estimators = 1000)

# Gradient Boosting
grboostf = GradientBoostingRegressor(learning_rate=0.1, max_depth = 2, max_features = 0.325, min_samples_leaf = 30, 
                                     min_samples_split = 10, n_estimators = 1000, subsample = 1)

# Adaptive Boosting
adaboostf = AdaBoostRegressor(learning_rate = 1, n_estimators = 2000)

# XGBoost
xgbmodf = XGBRegressor(colsample_bynode = 1, colsample_bytree = 1, gamma = 10, learning_rate = 0.1, max_depth = 4, 
                       n_estimators = 1000, objective = 'reg:squarederror', reg_lambda = 0.01, subsample = 0.325)

#%%

metrics = ['r2', 'mse', 'mae']

metric_df = pd.DataFrame()

for metric in metrics:
    metric_temp_df = cv_test_wrapper(train_df_female_stand, train_labels_female, test_df_female_stand, test_labels_female, 
                                     stable_models = [pls_modf, ridge_modf, lasso_modf, svm_modf],
                                     unstable_models = [rfmodf, grboostf, adaboostf, xgbmodf], 
                                     scoring=metric, n_max = 20)
    
    metric_temp_df['Metric'] = metric
    metric_df = pd.concat([metric_df, metric_temp_df])
    
metric_df.to_csv('../../../results/Age_prediction/age_female_metrics.csv')
