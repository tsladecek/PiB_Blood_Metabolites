#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

rfparams = params = {
    'n_estimators':[2000],
    'min_samples_split':np.arange(2, 40, 2),#
    'min_samples_leaf':np.arange(2, 40, 2),#
    'max_depth':np.arange(2, 11, 2), #
    'max_features':np.linspace(0.1, 1, 10),#
    'bootstrap':[True, False]
}
# Male
male_params = rf(train_df_male_stand, train_labels_male, rfparams, cvres = True)
male_best = pick_best(train_df_male_stand, train_labels_male, RandomForestRegressor(), np.array(cv_results(male_params, 20))[:, 0])
male_best.to_csv('../../../results/Age_prediction/rfbest_male.csv')

# Female
female_params = rf(train_df_female_stand, train_labels_female, rfparams, cvres = True)
female_best = pick_best(train_df_female_stand, train_labels_female, RandomForestRegressor(), np.array(cv_results(female_params, 20))[:, 0])
female_best.to_csv('../../../results/Age_prediction/rfbest_female.csv')

