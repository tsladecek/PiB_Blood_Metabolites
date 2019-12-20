import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, RidgeCV, LassoCV, Ridge, Lasso
from sklearn.svm import LinearSVR, SVR

from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error


# All Functions below (except Ridge&Lasso) Look very similar to each other. Their basic structure looks as follows:

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
#         return cv(grid.cv_results_), model_error(training_labels, grid.best_estimator_.predict(training_df))


def model_error(real, preds):
    """real, preds
    Computes training error
    """
    rmse = round(np.sqrt(mean_squared_error(real, preds)) ,5)
    r_squared = r2_score(real, preds)
    #return np.array(['RMSE: ', rmse], ['R^2', r_squared])
    return 'train R2: ', r_squared, 'train RMSE:', rmse

def cv_results(cv_arr, num = 5, r2 = True):
    """Extracts info about CV_test scores from grid search for every particular model configuration"""
    cvs = []
    for i in range(len(cv_arr['params'])):
        if r2:
            cvs.append((cv_arr['params'][i], cv_arr['mean_test_score'][i]))
        else:
            cvs.append((cv_arr['params'][i], np.sqrt(-cv_arr['mean_test_score'][i])))
    cvs.sort(key = lambda x: x[1])
    return cvs[::-1][0:num]

#############
#### PLS ####
#############

def pls(df, labels, params = None, which = 'classic', scoring = 'r2', cv = 10, n_jobs = -3, cvres = False):
    
    if which == 'classic':
        pls_mod = PLSRegression()
        if params is None:
            params = {'n_components':np.arange(2, 70), 'scale':[False]}
    else:
        pls_mod = PLSCanonical()
        if params is None:
            params = {'n_components':np.arange(2, 70), 'algorithm':['nipals', 'svd'], 'scale':[False]}
    
    pls_grid = GridSearchCV(pls_mod, param_grid=params, cv = cv, scoring = scoring, n_jobs = -3, iid = False)
    pls_grid.fit(df, labels)
    
    if cvres:
        return pls_grid.cv_results_
    else:
        return cv_results(pls_grid.cv_results_), model_error(labels, pls_grid.best_estimator_.predict(df))
    
#####################    
#### ELASTIC NET ####
#####################

def elnet(df, labels, params = None, scoring = 'r2', cv = 10, n_jobs = -3, cvres = False):
    
    elnet_mod = ElasticNet()
    
    if params is None:
        params = {
            'alpha':10.0**np.arange(-3, 4), 
            'l1_ratio':np.linspace(0, 1, 10), 
            'tol':[0.001, 0.01, 0.1, 0.5, 1], 
            'max_iter':[5000]
        }
    elnet_grid = GridSearchCV(elnet_mod, params, cv = cv, scoring = scoring, n_jobs = n_jobs, iid = False)
    elnet_grid.fit(df, labels)
    
    if cvres:
        return elnet_grid.cv_results_
    else:
        return cv_results(elnet_grid.cv_results_), model_error(labels, elnet_grid.best_estimator_.predict(df))
    
#######################
#### RIDGE & LASSO ####
#######################

def risso(df, labels, model = 'ridge', alphas = None, tol = 0.0001, cv = 10, scoring = 'r2'):
    
    if alphas is None:
        alphas = np.linspace(1, 100, 100) # default alphas the gridsearch is going to be performed on
    
    res = []
    for a in alphas:
        # pick a model (Ridge/Lasso)
        if model == 'ridge':
            mod = Ridge(alpha = a)
        else:
            mod = Lasso(alpha = a, tol = tol)
        mod.fit(df, labels)
        
        # perform Cross-Validation
        res.append(np.mean(cross_val_score(mod, df, labels, scoring = scoring, cv = cv)))
    return np.array([alphas, res])

#############
#### SVM ####
#############

def svm(df, labels, params = None, cv = 10, scoring = 'r2', n_jobs=-3, linearsvr = False, cvres = False):
    
    if linearsvr:
        svm_mod = LinearSVR()
        if params == None:
            params = {
                'C':[1, 10, 100, 1000, 10000],
                'loss':['squared_epsilon_insensitive', 'epsilon_insensitive'], 
                'random_state':[26]
            }
    else:
        svm_mod = SVR()
        if params == None:
            params = {
                'C':[1, 10, 100, 1000, 10000, 50000, 100000],
                'kernel':['linear', 'rbf', 'sigmoid', 'poly'], 
                'gamma':['auto'], 
                'epsilon':[0.1, 0.2, 0.5, 1, 2, 5] # controls the width of the 'street' - the wider the street less variance and more bias 
            }

    svm_grid = GridSearchCV(svm_mod, params, cv = cv, scoring = scoring, n_jobs = n_jobs, iid = False)
    svm_grid.fit(df, labels)
    
    if cvres:
        return svm_grid.cv_results_
    else:
        return cv_results(svm_grid.cv_results_, num = 5), model_error(labels, svm_grid.best_estimator_.predict(df))

######################
#### RandomForest ####
######################

def rf(df, labels, params = None, scoring = 'r2', cv = 10, n_jobs = -3, cvres = False):
    
    rf = RandomForestRegressor()
        
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
        return rf_grid.cv_results_
    else:
        return cv_results(rf_grid.cv_results_), model_error(labels, rf_grid.best_estimator_.predict(df))
    
################################
#### Boosting (gr, ada, xg) ####
################################
    
def boost(df, labels, params = None, mod = 'gradient', cv = 10, scoring = 'r2', n_jobs = -3, cvres = False):
    
    # Gradient Boosting
    if mod == 'gradient':
        boost_mod = GradientBoostingRegressor()
        if params is None:
            params = {'n_estimators':[100, 500, 1000, 5000], 
                      'learning_rate':[0.001, 0.01, 0.1, 1], 
                      'subsample':[0.3, 0.7, 1], 
                      'min_samples_split':[2,3,4], 
                      'n_iter_no_change':[5]
                      }
            
    # AdaBoost
    elif mod == 'ada':
        boost_mod = AdaBoostRegressor()
        if params is None:
            params = {'n_estimators':[100, 500, 1000, 5000], 'learning_rate':[0.001, 0.01, 0.1, 1]}
    
    # XGBoost
    elif mod == 'xgb':
        boost_mod = XGBRegressor()
        if params is None:
            params = {'max_depth':[2],
                  'learning_rate':[0.01, 0.1, 1],
                  'n_estimators':[1000], 
                  'booster':['gbtree'],
                  'n_jobs':[-1],
                  'subsample':np.linspace(0.2, 1, 5),
                  'reg_lambda':[1, 5, 10, 50, 100],
                  'colsample_bytree':np.linspace(0.2, 1, 5)
                 }
    boost_grid = GridSearchCV(boost_mod, params, cv = cv, scoring = scoring, n_jobs = n_jobs, iid = False)
    boost_grid.fit(df, labels)
    
    if cvres:
        return boost_grid.cv_results_
    else:
        return cv_results(boost_grid.cv_results_), model_error(labels, boost_grid.best_estimator_.predict(df)) 

    
def pick_best(df, labels, model, candidate_params, n = 10, cv = 10):
    """This function can be used to pick among several equivalently good models by inputting their parameter values. Then it predicts the traning data set using cross validation and each time computes the accuracy. The function returns the list of means, variances and corresponding models (their parameters)."""
    all_cv_errors = []
    
    for params in candidate_params:
        cv_errors = []
        for i in range(n):
            model.set_params(**params)
            
            preds = cross_val_predict(model, df, labels, cv = cv)
            cv_errors.append(model_error(labels, preds)[0])
        all_cv_errors.append([np.round(np.mean(cv_errors), 5), 
                              np.round(np.var(cv_errors), 5), 
                              params])
    all_cv_errors.sort(key=lambda x: x[0])
    return pd.DataFrame(all_cv_errors[::-1], columns = ['MeanCV', 'VarCV', 'Model_params'])