#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import sys 
sys.path.insert(1, '../../')
from classification_gridsearch import *

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Reading the files
train_df_raw = pd.read_csv('../../../steps/train_df_sex.csv')
train_df_raw.drop(list(train_df_raw)[0], axis = 1, inplace = True)
train_labels = train_df_raw.Sex

# Scaling
from sklearn.preprocessing import MinMaxScaler
train_df_raw.drop('Sex', axis = 1, inplace = True)

train_df_minmax = MinMaxScaler().fit_transform(train_df_raw)
#%%
# randomForest
rfparams = rf(train_df_minmax, train_labels, params = {
    'n_estimators':[1000],
    'min_samples_split':np.arange(5, 40, 5),
    'min_samples_leaf':np.arange(5, 40, 5),
    'max_depth':[2, 4, 6, 8], 
    'max_features':['sqrt', 'log2', None, 0.5, 0.8], #21, 9, 441, 220, 353 
    'bootstrap':[True]
}, cvres = True, n_jobs = -1)
#%%
with open('rfparams_sex', 'w') as f:
    f.write(str(rfparams))

#%%
rfbest = pick_best(train_df_minmax, train_labels, RandomForestClassifier(), np.array(cv_results(rfparams, 20))[:, 0])
rfbest.to_csv('../../../results/Sex_prediction/rfbest.csv')
