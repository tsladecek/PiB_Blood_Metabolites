import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict


# All Functions below Look very similar to each other. Their basic structure looks as follows:

# def function_name(training_df, training_labels, params = None, cv = 10, scoring = 'accuracy', cvres = False, other_params):
#     '''Specify which model is going to be used'''
#     model = MyModel()
#
#     ''' if statement that asks whether any parameters were supplied '''
#     if params is none:
#         # set default params
#
#     ''' Perform Gridsearch '''
#     grid = GridSearchCV(model, params, cv = cv, scoring)
#          #cv controls how many folds are created during CV. Scoring allows the user to specify a metric
#
#     '''If the raw gridsearch results are demanded'''
#     if cvres:
#         return grid.cv_results_
#
#     '''Otherwise return them in a nice format together with training error'''
#     else:
#         return cv(grid.cv_results_), training_accuracy(training_labels, grid.best_estimator_.predict(training_df))



def training_accuracy(real, preds):
    """Computes training accuracy"""
    return np.sum(real == preds) / len(real)

def cv_results(cv_arr, num = 5):
    """Extracts Results from GridSearch. By default shows only top 5 models"""
    cvs = []
    for i in range(len(cv_arr['params'])):
        cvs.append((cv_arr['params'][i], cv_arr['mean_test_score'][i]))
    cvs.sort(key = lambda x: x[1])
    
    if num == 0:
        return cvs[::-1]
    else:
        return cvs[::-1][:num]

############################
########### SVM ############
############################

def svm(df, labels, params = None, cv = 10, scoring = 'accuracy', n_jobs=-3, linearsvc = False, cvres = False):
    
    if linearsvc:
        svc_mod = LinearSVC()
        if params == None:
            params = {
                'C':[1, 10, 100, 1000, 10000],
                'loss':['hinge', 'squared_hinge'], 
                'penalty':['l2']
            }
    else:
        svc_mod = SVC()
        if params == None:
            params = [
                {
                    'C':[1, 10, 100, 1000, 10000, 100000],
                    'kernel':['linear'], 
                },
                {
                    'C':[1, 10, 100, 1000, 10000, 100000],
                    'kernel':['rbf', 'sigmoid'], 
                    'gamma':['auto', 'scale']  
                },
                {
                    'C':[1, 10, 100, 1000, 10000, 100000],
                    'kernel':['poly'],
                    'degree':[2, 3, 4],
                    'gamma':['auto', 'scale']
                }
            ]

    svc_grid = GridSearchCV(svc_mod, params, cv = cv, scoring = scoring, n_jobs = n_jobs, iid = False)
    svc_grid.fit(df, labels)
    if cvres:
        #print(cv_results(svc_grid.cv_results_, num = 5), training_accuracy(labels, svc_grid.best_estimator_.predict(df)))
        return svc_grid.cv_results_
    else:		
        return cv_results(svc_grid.cv_results_, num = 5), training_accuracy(labels, svc_grid.best_estimator_.predict(df))

############################
######## LDA, QDA ##########
############################

def da(df, labels, params = None, which = 'lda', scoring = 'accuracy', cv = 10, cvres = False, n_jobs = -3):
    if which == 'lda':
        if params is None:
            params = [
                {'solver':['lsqr', 'eigen'], 'shrinkage':['auto']},
                {'solver':['lsqr', 'eigen'], 'shrinkage':np.linspace(0.01, 0.99, 20)}
            ]
        da = LinearDiscriminantAnalysis()
    else:
        if params is None:
             params = {'reg_param':np.linspace(0, 1, 9)}
        da = QuadraticDiscriminantAnalysis()
        
    da_grid = GridSearchCV(da, params, scoring = scoring, cv = cv, iid = False)
    da_grid.fit(df, labels)
    
    if cvres:
        #print('Traning accuracy of the best model: ', training_accuracy(labels, da_grid.best_estimator_.predict(df)))
        return da_grid.cv_results_
    else:
        return cv_results(da_grid.cv_results_), training_accuracy(labels, da_grid.best_estimator_.predict(df))

############################
######### LogReg ###########
############################

def log_reg(df, labels, params = None, scoring = 'accuracy', cv = 10, n_jobs = -3, cvres = False):
    
    log = LogisticRegression()
        
    if params == None:
        params = [
            {
                'penalty':['l2', 'l1'],
                'solver':['liblinear'],
                'C':[0.001, 0.01, 0.1, 1, 10],
            }, 
            {
                'penalty':['l2'],
                'solver':['lbfgs'],
                'C':[0.001, 0.01, 0.1, 1, 10],
            }
        ]
    log_grid = GridSearchCV(log, params, scoring = scoring, cv = cv, iid = False, n_jobs = n_jobs)
    log_grid.fit(df, labels)
    
    if cvres:
        #print('Traning accuracy of the best model: ', training_accuracy(labels, log_grid.best_estimator_.predict(df)))
        return log_grid.cv_results_
    else:
        return cv_results(log_grid.cv_results_), training_accuracy(labels, log_grid.best_estimator_.predict(df))

############################
########### RF #############
############################

def rf(df, labels, params = None, scoring = 'accuracy', cv = 10, n_jobs = -3, cvres = False):
    
    rf = RandomForestClassifier()
        
    if params == None:
        params = {
            'n_estimators':[500],
            'max_depth':np.arange(2, 10, 2),
            'min_samples_leaf':np.arange(3, 6),
            'bootstrap':[True, False]
            
        }
    rf_grid = GridSearchCV(rf, params, scoring = scoring, cv = cv, iid = False, n_jobs = n_jobs)
    rf_grid.fit(df, labels)
    
    if cvres:
        #print('Traning accuracy of the best model: ', training_accuracy(labels, rf_grid.best_estimator_.predict(df)))
        return rf_grid.cv_results_
    else:
        return cv_results(rf_grid.cv_results_), training_accuracy(labels, rf_grid.best_estimator_.predict(df))

############################
######## Boosting ##########
############################

def boost(df, labels, clf = 'gradient', params = None, cv = 10, scoring = 'accuracy', n_jobs = -3, cvres = False):
    
    # Gradient Boosting
    if clf == 'gradient' or clf == 'grboost':
        boost_mod = GradientBoostingClassifier()
        if params is None:
            params = {'n_estimators':[100, 500, 1000, 5000], 
                      'learning_rate':[0.001, 0.01, 0.1, 1], 
                      'subsample':[0.3, 0.7, 1], 
                      'min_samples_split':[2,3,4]}
            
    # AdaBoost        
    elif clf == 'ada' or clf == 'adaboost':
        boost_mod = AdaBoostClassifier()
        if params is None:
            params = {'n_estimators':[100, 500, 1000, 5000], 'learning_rate':[0.001, 0.01, 0.1, 1]}
    
    # XGBoost        
    elif clf == 'xgb' or clf == 'xgboost':
        boost_mod = XGBClassifier()
        if params is None:
            params = {'max_depth':[2, 3], 
                      'learning_rate':[0.01, 0.1, 1], 
                      'subsample':[0.2, 0.6, 1],
                      'reg_lambda':[0.1, 1, 10, 100],
                      'n_estimators':[200]}

    boost_grid = GridSearchCV(boost_mod, params, cv = cv, scoring = scoring, n_jobs = n_jobs, iid = False)
    boost_grid.fit(df, labels)
    
    if cvres:
        #print('Traning accuracy of the best model: ', training_accuracy(labels, boost_grid.best_estimator_.predict(df)))
        return boost_grid.cv_results_
    else:
        return cv_results(boost_grid.cv_results_), training_accuracy(labels, boost_grid.best_estimator_.predict(df)) 


def knn(df, labels, params = None, scoring = 'accuracy', cv = 10, n_jobs = -3, cvres = False):
    
    knnc = KNeighborsClassifier()
        
    if params == None:
        params = {
            'n_neighbors':[1, 5, 10, 15, 20],
            'weights':['uniform', 'distance'],
            'algorithm':['auto'],
            'n_jobs':[n_jobs]
        }
    knn_grid = GridSearchCV(knnc, params, scoring = scoring, cv = cv, iid = False, n_jobs = n_jobs)
    knn_grid.fit(df, labels)
    
    if cvres:
        #print('Traning accuracy of the best model: ', training_accuracy(labels, log_grid.best_estimator_.predict(df)))
        return knn_grid.cv_results_
    else:
        return cv_results(knn_grid.cv_results_), training_accuracy(labels, knn_grid.best_estimator_.predict(df))
    
def pick_best(df, labels, model, candidate_params, n = 10, cv = 10):
    """This function can be used to pick among several equivalently good models by inputting their parameter values. Then it predicts the traning data set using cross validation and each time computes the accuracy. The function returns the list of means, variances and corresponding models (their parameters)."""
    all_cv_errors = []
    
    for params in candidate_params:
        cv_errors = []
        for i in range(n):
            model.set_params(**params)
            
            preds = cross_val_predict(model, df, labels, cv = cv)
            cv_errors.append(training_accuracy(labels, preds))
        all_cv_errors.append([np.round(np.mean(cv_errors), 5), 
                              np.round(np.var(cv_errors), 5), 
                              params])
    all_cv_errors.sort(key=lambda x: x[0])
    return pd.DataFrame(all_cv_errors[::-1], columns = ['MeanCV', 'VarCV', 'Model_params'])