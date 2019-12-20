#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:31:03 2019

@author: tomasla
"""

#%%
import sys 
sys.path.insert(1, '../../')
from classification_gridsearch import *

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
#%%

def preparedf(path):
    dfraw = pd.read_csv(path)
    labels = dfraw['Menopause']
    df = dfraw.iloc[:, 2:]
    return df, labels

train_df, train_labels = preparedf('../../../steps/train_df_meno.csv')
#%%
from sklearn.preprocessing import MinMaxScaler
train_df_minmax = MinMaxScaler().fit_transform(train_df)

#%%
# XGBoost
lambdas = [10**(i) for i in range(-3, 4)]
xgparams = boost(train_df_minmax, train_labels, clf = 'xgb', params = {
    #'n_estimators':[10]
    'n_estimators':[1000],
    'learning_rate':[0.001, 0.01, 0.1, 1, 2],
    'max_depth':[2, 3, 4], 
    'colsample_bytree':np.linspace(0.1, 1, 5),
    'colsample_bynode':np.linspace(0.1, 1, 5),
    'reg_lambda':lambdas,
    'subsample':np.linspace(0.1, 1, 5)
}, cvres = True, n_jobs = -1)
    
#%%
with open('xgbparams', 'w') as f:
    f.write(str(xgparams))
    
#%%    
xgbest = pick_best(train_df_minmax, train_labels, XGBClassifier(), np.array(cv_results(xgparams, 20))[:, 0])
xgbest.to_csv('../../../results/MS_prediction/xgbest.csv')
