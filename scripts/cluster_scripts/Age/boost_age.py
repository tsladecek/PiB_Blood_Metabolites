#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import sys 
sys.path.insert(1, '../../')
from regression_gridsearch import *

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

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

#%%
grparams = {
        #'n_estimators':[10]
    'n_estimators':[1000],
    'learning_rate':[0.001, 0.01, 0.1, 1, 2],
    'subsample':np.linspace(0.1, 1, 5),
    'min_samples_split':np.arange(5, 40, 5),
    'min_samples_leaf':np.arange(5, 40, 5),
    'max_depth':[2, 3, 4],
    'max_features':np.linspace(0.1, 1, 5)
}

#%%
# Male
male_grparams = boost(train_df_male_stand, train_labels_male, grparams, cvres = True, n_jobs = -1)

with open('grparams_male', 'w') as f:
    f.write(str(male_grparams))

female_grparams = boost(train_df_female_stand, train_labels_female, grparams, cvres = True, n_jobs = -1)

with open('grparams_female', 'w') as f:
    f.write(str(female_grparams))

#%%
# Best Params Male
male_grbest = pick_best(train_df_male_stand, train_labels_male, GradientBoostingRegressor(), 
                        np.array(cv_results(male_grparams, 20))[:, 0])
male_grbest.to_csv('../../../results/Age_prediction/grbest_male.csv')
#%%
# Female
female_grbest = pick_best(train_df_female_stand, train_labels_female, GradientBoostingRegressor(), 
                        np.array(cv_results(female_grparams, 20))[:, 0])
female_grbest.to_csv('../../../results/Age_prediction/grbest_female.csv')

