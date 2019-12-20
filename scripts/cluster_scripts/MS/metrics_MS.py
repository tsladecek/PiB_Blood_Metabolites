import numpy as np
import pandas as pd

import sys
sys.path.insert(1, '../../')
from ml_functions import *

def preparedf(path):
    dfraw = pd.read_csv(path)
    labels = dfraw['Menopause']
    df = dfraw.iloc[:, 2:]
    return df, labels

train_df, train_labels = preparedf('../../../steps/train_df_meno.csv')
test_df, test_labels = preparedf('../../../steps/test_df_meno.csv')

### SCALING ###
from sklearn.preprocessing import StandardScaler, MinMaxScaler
stand_scaler, mm_scaler = StandardScaler(), MinMaxScaler()

stand_scaler.fit(train_df)
mm_scaler.fit(train_df)

train_df_stand, test_df_stand = stand_scaler.transform(train_df), stand_scaler.transform(test_df)
train_df_minmax, test_df_minmax = mm_scaler.transform(train_df), mm_scaler.transform(test_df)

from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
#%%
# SVM
svc_mod = SVC(C = 110, gamma = 'auto', kernel = 'poly', degree = 2)

# Logistic regression
log_mod = LogisticRegression(penalty = 'l2', solver = 'lbfgs', C = 1)

# LDA DONE
lda_mod = LinearDiscriminantAnalysis(shrinkage = 'auto', solver = 'lsqr')

# QDA DONE
qda_mod = QuadraticDiscriminantAnalysis(reg_param = 0.75)

# Gradient Boosting
gr_mod = GradientBoostingClassifier(learning_rate = 1, max_depth = 4, max_features = 0.775, 
                                    min_samples_leaf = 35, min_samples_split = 10, n_estimators = 1000, subsample = 1)

# Adaptive Boosting
ada_mod = AdaBoostClassifier(n_estimators = 800, learning_rate = 0.01)

# XGBoost
xgb_mod = XGBClassifier(colsample_bynode = 0.55, colsample_bytree= 0.1, learning_rate = 2,
                        max_depth = 2, n_estimators = 1000, reg_lambda = 10, subsample = 0.325)

# RandomForest
rf_mod = RandomForestClassifier(bootstrap = True, max_depth = 6, max_features = 0.8, 
                                min_samples_leaf = 5, min_samples_split = 15, n_estimators = 1000)

### PCA ### 
pca = PCA()
pca.fit(train_df_stand)
train_pcs = pca.transform(train_df_stand)
#phi = pca.components_
#test_pcs = test_df_stand.dot(phi.T)
test_pcs = pca.transform(test_df_stand)

# pca-lda 
pca_lda = LinearDiscriminantAnalysis(shrinkage = 0.6, solver = 'lsqr')

# pca_qda 
pca_qda = QuadraticDiscriminantAnalysis(reg_param = 0.875)

# KNN 
knn_mod = KNeighborsClassifier(n_neighbors = 5, algorithm = 'auto', weights = 'uniform')

#%%

metrics = ['accuracy', 'precision', 'recall', 'f1', 'matthews_corrcoef', 'roc_auc']

metric_df = pd.DataFrame()

for metric in metrics:
    metric_temp_df = cv_test_wrapper(train_df_minmax, train_labels, test_df_minmax, test_labels, train_pcs = train_pcs, test_pcs = test_pcs, 
                                     stable_models = [lda_mod, qda_mod, knn_mod], 
                                     unstable_models = [svc_mod, log_mod, gr_mod, ada_mod, xgb_mod, rf_mod], 
                                     pc_models = [pca_lda, pca_qda], max_pcs = [96, 20], 
                                     scoring = metric, n_max = 20)

    metric_temp_df['Metric'] = metric
    metric_df = pd.concat([metric_df, metric_temp_df])
    
metric_df.to_csv('../../../results/MS_prediction/MS_metrics.csv')