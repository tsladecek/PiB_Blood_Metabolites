#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import numpy as np
import pandas as pd

import sys
sys.path.insert(1, '../../')
from regression_gridsearch import *

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
#%%
rfparams = params = {
       # 'n_estimators':[10]
    'n_estimators':[1000],
    'min_samples_split':np.arange(5, 40, 5),
    'min_samples_leaf':np.arange(5, 40, 5),
    'max_depth':[2, 4, 6, 8], 
    'max_features':['sqrt', 'log2', None, 0.5, 0.8], #21, 9, 441, 220, 353 
    'bootstrap':[True]
}
#%%
# Male
male_rfparams = rf(train_df_male_stand, train_labels_male, rfparams, cvres = True, n_jobs = -1)

with open('rfmale', 'w') as f:
    f.write(str(male_rfparams))

# Female
female_rfparams = rf(train_df_female_stand, train_labels_female, rfparams, cvres = True, n_jobs = -1)

with open('rffemale', 'w') as f:
    f.write(str(female_rfparams))

#%%
# Best Params Male
male_best = pick_best(train_df_male_stand, train_labels_male, RandomForestRegressor(), np.array(cv_results(male_rfparams, 20))[:, 0])
male_best.to_csv('../../../results/Age_prediction/rfbest_male.csv')

female_best = pick_best(train_df_female_stand, train_labels_female, RandomForestRegressor(), np.array(cv_results(female_rfparams, 20))[:, 0])
female_best.to_csv('../../../results/Age_prediction/rfbest_female.csv')

