#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:58:58 2019

@author: tomasla
"""

#%%
import sys 
sys.path.insert(1, '../../')
from classification_gridsearch import *

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
# randomForest
rfparams = rf(train_df_minmax, train_labels, params = {
    #'n_estimators':[10]
    'n_estimators':[1000],
    'min_samples_split':np.arange(5, 40, 5),
    'min_samples_leaf':np.arange(5, 40, 5),
    'max_depth':[2, 4, 6, 8], 
    'max_features':['sqrt', 'log2', None, 0.5, 0.8], #21, 9, 441, 220, 353 
    'bootstrap':[True]
}, cvres = True, n_jobs = -1)

#%%    
with open('rfparams', 'w') as f:
    f.write(str(rfparams))
#%%
rfbest = pick_best(train_df_minmax, train_labels, RandomForestClassifier(), np.array(cv_results(rfparams, 20))[:, 0])
rfbest.to_csv('../../../results/MS_prediction/rfbest.csv')
