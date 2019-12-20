#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:59:19 2019

@author: tomasla
"""

#%%
import sys 
sys.path.insert(1, '../../')
from classification_gridsearch import *

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
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
# GrBoost
grparams = boost(train_df_minmax, train_labels, params = {
    #'n_estimators':[10]
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
grbest.to_csv('../../../results/MS_prediction/grbest.csv')


