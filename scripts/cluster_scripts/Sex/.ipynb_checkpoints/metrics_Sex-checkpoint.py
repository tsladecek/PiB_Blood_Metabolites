#%%
import numpy as np
import pandas as pd

import sys
sys.path.insert(1, '../../')
from ml_functions import *

#### PREPARATION OF DATASETS ####

train_df_raw = pd.read_csv('../../../steps/train_df_sex.csv')
train_df_raw.drop(list(train_df_raw)[0], axis = 1, inplace = True)
train_labels = train_df_raw.Sex
train_df_raw.head()

test_df_raw = pd.read_csv('../../../steps/test_df_sex.csv')
test_labels = test_df_raw.Sex
test_df_raw.drop(list(test_df_raw)[:2], axis = 1, inplace = True)

################### SCALING #######################
from sklearn.preprocessing import StandardScaler, MinMaxScaler
train_df_raw.drop('Sex', axis = 1, inplace = True)

standscaler = StandardScaler()
standscaler.fit(train_df_raw)
train_df_stand = standscaler.transform(train_df_raw)
test_df_stand = standscaler.transform(test_df_raw)

minmaxscaler = MinMaxScaler()
minmaxscaler.fit(train_df_raw)
train_df_minmax = minmaxscaler.transform(train_df_raw)
test_df_minmax = minmaxscaler.transform(test_df_raw)

#### IMPORTING FUNCTIONS USED FOR MODELLING ####

from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

#%%
################
#### MODELS ####
################

# SVM
svc_mod = SVC(C = 1, gamma = 'auto', kernel = 'linear', probability = True)

# Logistic regression
log_mod = LogisticRegression(penalty = 'l2', solver = 'liblinear', C = 2)

# LDA 
lda_mod = LinearDiscriminantAnalysis(shrinkage = 0.5, solver = 'lsqr')

# QDA 
qda_mod = QuadraticDiscriminantAnalysis(reg_param = 0.75)

# Gradient Boosting
gr_mod = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 2, max_features = 0.1, 
                                    min_samples_leaf = 20, min_samples_split = 15, n_estimators = 1000, subsample = 1)

# Adaptive Boosting 
ada_mod = AdaBoostClassifier(n_estimators = 500, learning_rate = 1)

# XGBoost
xgb_mod = XGBClassifier(colsample_bynode = 0.775, colsample_bytree = 1, learning_rate = 1, 
                        max_depth = 2, n_estimators = 1000, reg_lambda = 1000, subsample = 0.325)

# RandomForest
#rf_mod = RandomForestClassifier(bootstrap = True, max_depth = 4, max_features = None, 
#                                min_samples_leaf = 5, min_samples_split = 20, n_estimators = 1000)
rf_mod = RandomForestClassifier(bootstrap = True, max_depth = 6, max_features = 'sqrt', 
                                min_samples_leaf = 5, min_samples_split = 10, n_estimators = 1000)


### PCA ### 
pca = PCA(n_components=112, svd_solver = 'full')
pca.fit(train_df_stand)
train_pcs = pca.transform(train_df_stand)
test_pcs = pca.transform(test_df_stand)

# pca-lda
pca_lda = LinearDiscriminantAnalysis(shrinkage = 0.1, solver = 'lsqr')
pca_lda.fit(train_pcs[:, :112], train_labels) 

# pca_qda 
pca_qda = QuadraticDiscriminantAnalysis(reg_param = 0.75)
pca_qda.fit(train_pcs[:, :30], train_labels)

# KNN 
knn_mod = KNeighborsClassifier(n_neighbors = 6, algorithm = 'auto', weights = 'uniform')

#%%
metrics = ['accuracy', 'precision', 'recall', 'f1', 'matthews_corrcoef', 'roc_auc']

metric_df = pd.DataFrame()

for metric in metrics:
    metric_temp_df = cv_test_wrapper(train_df_minmax, train_labels, test_df_minmax, test_labels, train_pcs = train_pcs, test_pcs = test_pcs, 
                                     stable_models = [lda_mod, qda_mod, knn_mod], 
                                     unstable_models = [svc_mod, log_mod, gr_mod, ada_mod, xgb_mod, rf_mod], 
                                     pc_models = [pca_lda, pca_qda], max_pcs = [112, 30], 
                                     scoring = metric, n_max = 20)

    metric_temp_df['Metric'] = metric
    metric_df = pd.concat([metric_df, metric_temp_df])
    
metric_df.to_csv('../../../results/Sex_prediction/sex_metrics.csv')