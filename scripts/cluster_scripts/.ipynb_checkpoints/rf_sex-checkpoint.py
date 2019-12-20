#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys 
sys.path.insert(1, '../')
from classification_gridsearch import *

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Reading the files
train_df_raw = pd.read_csv('../../steps/train_df_sex.csv')
train_df_raw.drop(list(train_df_raw)[0], axis = 1, inplace = True)
train_labels = train_df_raw.Sex

# Scaling
from sklearn.preprocessing import MinMaxScaler
train_df_raw.drop('Sex', axis = 1, inplace = True)

train_df_minmax = MinMaxScaler().fit_transform(train_df_raw)

# randomForest
rfparams = rf(train_df_minmax, train_labels, params = {
    'n_estimators':[10],
    #'min_samples_split':np.arange(5, 50, 3),#
    #'min_samples_leaf':np.arange(5, 50, 3),#
    'max_depth':np.arange(2, 11, 2), #
    #'max_features':np.linspace(0.1, 1, 10),#
    #'bootstrap':[True, False]
}, cvres = True, n_jobs = -1)

rfbest = pick_best(train_df_minmax, train_labels, RandomForestClassifier(), np.array(cv_results(rfparams, 20))[:, 0])
rfbest.to_csv('../../results/Sex_prediction/rfbest.csv')
