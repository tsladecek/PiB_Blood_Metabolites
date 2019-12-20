import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, r2_score, mean_squared_error, mean_absolute_error
from copy import deepcopy

#### IDEA ####
# 1. set random seed
# 2. perform cross validation: train a model on k-1 folds and save predictions for the kth fold -> in the end there will be one prediction for every observation in the train dataset
# 3. Calculate the scoring metric (accuracy, recall, mathews, ...) for the entire dataset
# 4. repeat the process with different seed

def cvcalc(model, df, labels, scoring, k = 10, n_max = 20, random_seed = None):
    """Cross validation performed n_max times, but the error is calculated on the end of each run. This results in more precise CV estimates"""
        
    # raw indices of rows -> [0, 1, 2, ..., n]
    df_indices = np.arange(len(df))
    labels = np.array(labels)
    
    # Folds that are going to be used in CV
    # this generates k lists, where each sublist is composed of two lists - one with indices to train on and
    # another with indices to predict on
    fold_indices = []
    kf = KFold(n_splits = k)
    for train_ind, validation_ind in kf.split(df):
        fold_indices.append([train_ind, validation_ind])
    
    scorings = []
    for i in range(n_max):
        
        # this condition is neccessary if n_max = 1. See cv_test_wrapper function with unstable models. It makes sure that if we use this function
        # many times in a loop, than every stratification of the dataset will be different
        if random_seed is None:
            np.random.seed(i)
        else: 
            np.random.seed(random_seed)
            
        # shuffling dataset and labels
        shuffled_indices = np.random.choice(df_indices, replace = False, size = len(df_indices))
        shuffled_df = df[shuffled_indices, :]
        shuffled_labels = labels[shuffled_indices]
        
        # Performing cv
        fold_predictions = []
        for j in range(k):
            train_df, train_labels = shuffled_df[fold_indices[j][0], :], shuffled_labels[fold_indices[j][0]]
            validation_df = shuffled_df[fold_indices[j][1], :]
            
            model.fit(train_df, train_labels)
            fold_predictions.extend(model.predict(validation_df))
        
        if scoring == 'accuracy':
            scorings.append(accuracy_score(shuffled_labels, fold_predictions))
        elif scoring == 'precision':
            scorings.append(precision_score(shuffled_labels, fold_predictions))
        elif scoring == 'recall':
            scorings.append(recall_score(shuffled_labels, fold_predictions))
        elif scoring == 'f1' or scoring == 'f1_score':
            scorings.append(f1_score(shuffled_labels, fold_predictions))
        elif scoring == 'roc_auc':
            scorings.append(roc_auc_score(shuffled_labels, fold_predictions))
        elif scoring == 'matthews_corrcoef' or scoring == 'mcc':
            scorings.append(matthews_corrcoef(shuffled_labels, fold_predictions))
        elif scoring == 'r2':
            scorings.append(r2_score(shuffled_labels, fold_predictions))
        elif scoring == 'mean_squared_error' or scoring == 'mse':
            scorings.append(mean_squared_error(shuffled_labels, fold_predictions))
        elif scoring == 'mean_absolute_error' or scoring == 'mae':
            scorings.append(mean_absolute_error(shuffled_labels, fold_predictions))
        else:
            return 'Unknown scoring function'
    return scorings

def cv_test_errors(train_df, train_labels, test_df, test_labels, models, scoring, model_names = None, k = 10, n_max = 20, random_seed = None):
    """Function to generate CV errors and Test errors for different models"""
    
    if model_names is None:
        # this just extracts the name of the model and appends an Abbreviation based on the capital letters
        # eg. RandomForestClassifier -> RFC 
        model_names = [''.join([i for i in str(m.get_params).split('(')[0].split(' ')[-1] if i.isupper()]) for m in models]
    
    # CV error
    cv_scores = []
    
    for i, model in enumerate(models):
        modelcv = []
        model_copy = deepcopy(model)
        
        # Cross-Validation
        cv = cvcalc(model_copy, train_df, train_labels, scoring, k, n_max, random_seed = random_seed)
        
        model.fit(train_df, train_labels)
        # Test error
        if scoring == 'accuracy':
            test_error = accuracy_score(test_labels, model.predict(test_df))
        elif scoring == 'precision':
            test_error = precision_score(test_labels, model.predict(test_df))
        elif scoring == 'recall':
            test_error = recall_score(test_labels, model.predict(test_df))
        elif scoring == 'f1' or scoring == 'f1_score':
            test_error = f1_score(model.predict(test_df), test_labels)
        elif scoring == 'roc_auc':
            test_error = roc_auc_score(test_labels, model.predict(test_df))
        elif scoring == 'matthews_corrcoef' or scoring == 'mcc':
            test_error = matthews_corrcoef(model.predict(test_df), test_labels)
        elif scoring == 'r2':
            test_error = r2_score(test_labels, model.predict(test_df))
        elif scoring == 'mean_squared_error' or scoring == 'mse':
            test_error = mean_squared_error(test_labels, model.predict(test_df))
        elif scoring == 'mean_absolute_error' or scoring == 'mae':
            test_error = mean_absolute_error(test_labels, model.predict(test_df))
        else:
            return 'Unknown scoring function'
        
        if n_max > 1:
            modelcv.extend([model_names[i], test_error, np.mean(cv), np.var(cv)])
        else:
            modelcv.extend([model_names[i], test_error])
        modelcv.extend(cv)
        cv_scores.append(modelcv)
    
    if n_max > 1:
        names = ['Model', 'Test_error', 'MeanCV', 'VarCV']

        names1 = [f'run{i}' for i in range(n_max)]
        names.extend(names1)
    else:
        names = ['Model', 'Test_error', 'CV']
    cv_scores_df = pd.DataFrame(cv_scores, columns = names)
    return cv_scores_df


##################################################################
def gather(df):
    df.drop(columns = ['MeanCV', 'VarCV'], inplace = True)
    return pd.melt(df, id_vars = ['Model', 'Test_error'], value_name = 'CV').drop(columns = ['variable'])

def cv_test_wrapper(train_df = None, train_labels = None, test_df = None, test_labels = None, train_pcs = None, test_pcs = None, stable_models = None, unstable_models = None, pc_models = None, max_pcs = None, scoring = 'accuracy', n_max = 20):
    """Wrapper Function. Creates a table with estimates of Error from Cross validation and also from Test dataset for a certain metric. The models are divided in three categories: PCA models, non-PCA models that are stable (eg. LDA, QDA) and the rest that create a slightly different model each time
    the `fit` command is called"""
    
    all_models = pd.DataFrame()
    
    # PCA MODELS
    if pc_models != None:
        for i, pcm in enumerate(pc_models):
            model_name = 'PCA_' + ''.join([i for i in str(pcm.get_params).split('(')[0].split(' ')[-1] if i.isupper()]) # extract name
            pcadf = cv_test_errors(train_pcs[:, :max_pcs[i]], train_labels, test_pcs[:, :max_pcs[i]], test_labels,  
                                   models = [pcm], scoring = scoring, model_names = [model_name], n_max = n_max) # calculate cv & test errors
            
            pcadf = gather(pcadf)
            all_models = pd.concat([all_models, pcadf]) # append to a dataframe


    # Stable models
    if stable_models != None:
        stable_df = cv_test_errors(train_df, train_labels, test_df, test_labels, stable_models, scoring = scoring, n_max = n_max)
        stable_df = gather(stable_df)
        
        all_models = pd.concat([all_models, stable_df])
    
    # Unstable Models
    if unstable_models != None:
        unstable_df = pd.DataFrame()

        for m, model in enumerate(unstable_models):
            for i in range(n_max):
                model = model.set_params(**{'random_state':i}) # this ensures that the model parameters will be different with every seed
                unstable_df = pd.concat([unstable_df, cv_test_errors(train_df, train_labels, test_df, test_labels, 
                                                                     [model], scoring, n_max = 1, random_seed = i)])
                                                # the random seed here ensures that the splitting of the dataframe during Cross-Validation
                                                # results in different splits for each iteration. (It is neccessary since we are doing the
                                                # CV one model at a time. It is a problem only if n_max = 1. Otherwise the function takes
                                                # care of this itself.)
        all_models = pd.concat([all_models, unstable_df])
        
    return all_models.sort_values('Model')


