#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import sys 
sys.path.insert(1, '../../')
from regression_gridsearch import *

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

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
    'n_estimators':[2000],
    'learning_rate':[0.001, 0.01, 0.1, 1, 2],
    'subsample':np.linspace(0.1, 1, 10),
    'min_samples_split':np.arange(5, 50, 3),
    'min_samples_leaf':np.arange(5, 50, 3),
    'max_depth':[2, 3, 4],
    'max_features':np.linspace(0.1, 1, 10)
}

adaparams = {
        'base_estimator':[DecisionTreeRegressor(max_depth = 2), DecisionTreeRegressor(max_depth = 3), 
                          DecisionTreeRegressor(max_depth = 4)],
        'n_estimators':[100, 500, 1000, 2000],
        'learning_rate':[0.001, 0.01, 0.1, 1, 2],
        }
#%%
# Male
# Gradient Boosting
male_grparams = boost(train_df_male_stand, train_labels_male, grparams, cvres = True)
male_grbest = pick_best(train_df_male_stand, train_labels_male, GradientBoostingRegressor(), 
                        np.array(cv_results(male_grparams, 20))[:, 0])
male_grbest.to_csv('../../../results/Age_prediction/grbest_male.csv')

# AdaBoost
male_adaparams = boost(train_df_male_stand, train_labels_male, adaparams, clf = 'ada', cvres = True)
male_adabest = pick_best(train_df_male_stand, train_labels_male, AdaBoostRegressor(), 
                         np.array(cv_results(male_adaparams, 20))[:, 0])
male_adabest.to_csv('../../../results/Age_prediction/adabest_male.csv')

#%%
# Female
# Gradient Boosting
female_grparams = boost(train_df_female_stand, train_labels_female, grparams, cvres = True)
female_grbest = pick_best(train_df_female_stand, train_labels_female, GradientBoostingRegressor(), 
                        np.array(cv_results(female_grparams, 20))[:, 0])
female_grbest.to_csv('../../../results/Age_prediction/grbest_female.csv')

# AdaBoost
female_adaparams = boost(train_df_female_stand, train_labels_female, adaparams, clf = 'ada', cvres = True)
female_adabest = pick_best(train_df_female_stand, train_labels_female, AdaBoostRegressor(), 
                         np.array(cv_results(female_adaparams, 20))[:, 0])
female_adabest.to_csv('../../../results/Age_prediction/adabest_female.csv')


