#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import sys 
sys.path.insert(1, '../../')
from classification_gridsearch import *

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Reading the files
train_df_raw = pd.read_csv('../../../steps/train_df_sex.csv')
train_df_raw.drop(list(train_df_raw)[0], axis = 1, inplace = True)
train_labels = train_df_raw.Sex

# Scaling
from sklearn.preprocessing import MinMaxScaler
train_df_raw.drop('Sex', axis = 1, inplace = True)

train_df_minmax = MinMaxScaler().fit_transform(train_df_raw)
#%%
# GrBoost
grparams = boost(train_df_minmax, train_labels, params = {
    'n_estimators':[1000],
    'learning_rate':[0.001, 0.01, 0.1, 1, 2],
    'min_samples_split':np.arange(5, 40, 5),
    'min_samples_leaf':np.arange(5, 40, 5),
    'max_depth':[2, 3, 4], 
    'max_features':np.linspace(0.1, 1, 5),
    'subsample':np.linspace(0.1, 1, 5)
}, cvres = True, n_jobs = -1)
#%%
with open('grparams', 'w') as f:
    f.write(str(grparams))

#%%
grbest = pick_best(train_df_minmax, train_labels, GradientBoostingClassifier(), np.array(cv_results(grparams, 20))[:, 0])
grbest.to_csv('../../../results/Sex_prediction/grbest.csv')

