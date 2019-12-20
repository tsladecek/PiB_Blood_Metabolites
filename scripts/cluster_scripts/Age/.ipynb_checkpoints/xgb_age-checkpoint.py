#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:04:47 2019

@author: tomasla
"""

#%%
import sys 
sys.path.insert(1, '../../')
from regression_gridsearch import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from xgboost import XGBRegressor

#%%
def prepare_df(path):
    df = pd.read_csv(path)
    labels = df.Age
    df.drop(list(df)[:2], axis = 1, inplace = True)
    return df, labels

train_df_male, train_labels_male = prepare_df('../../../steps/train_df_male_age.csv')
train_df_female, train_labels_female = prepare_df('../../../steps/train_df_female_age.csv')

# Scaling
from sklearn.preprocessing import StandardScaler

train_df_male_stand = StandardScaler().fit_transform(train_df_male)
train_df_female_stand = StandardScaler().fit_transform(train_df_female)

#%% Parameters
lambdas = [10**(i) for i in range(-3, 4)]
xgparams = {
      #  'n_estimators':[10],
    'objective':['reg:squarederror'],
    'max_depth':[2, 3, 4],
    'learning_rate':[0.001, 0.01, 0.1, 1],
    'n_estimators':[1000],
    'gamma':np.linspace(0, 20, 3),
    'n_jobs':[-1],
    'subsample':np.linspace(0.1, 1, 5),
    'colsample_bytree':np.linspace(0.1, 1, 3),
    'colsample_bynode':np.linspace(0.1, 1, 3),
    'reg_lambda':lambdas
}

#%%
# Male
male_xgparams = boost(train_df_male_stand, train_labels_male, xgparams, mod = 'xgb', cvres = True, n_jobs = -1)

with open('xgparams_male', 'w') as f:
    f.write(str(male_xgparams))
    
#Female
female_xgparams = boost(train_df_female_stand, train_labels_female, xgparams, mod = 'xgb', cvres = True, n_jobs = -1)
with open('xgparams_female', 'w') as f:
    f.write(str(female_xgparams))

#%% 
# Best Params male
male_xgbest = pick_best(train_df_male_stand, train_labels_male, XGBRegressor(), 
                        np.array(cv_results(male_xgparams, 20))[:, 0])
male_xgbest.to_csv('../../../results/Age_prediction/xgbest_male.csv')

#%%
# Female
female_xgbest = pick_best(train_df_female_stand, train_labels_female, XGBRegressor(), 
                        np.array(cv_results(female_xgparams, 20))[:, 0])
female_xgbest.to_csv('../../../results/Age_prediction/xgbest_female.csv')


