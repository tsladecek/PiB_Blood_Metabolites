# SCRIPTS

Scripts used for modelling. They are either Python Notebooks which contain the analyses or python 
scripts that contain some wrapper functions that were created on the run to make the whole process 
easier.

### Data_Preparation_karmen.ipynb
Contains Data preparation, removing missing values and creation of Train and Test datasets for the prediction of Sex, Age and Menopausal Status

### Sex_prediction.ipynb, Age_prediction.ipynb, MS_prediction.ipynb
Model optimizations through hyperparameter Grid-Search

### classification_gridsearch.py, regression_gridsearch.py
Files containing wrapper functions used when searching through the hyper-parameter space

### ml_functions.py
Functions created for Cross-Validation

### cluster_scripts/
Scripts for performing gridsearch on tree-based methods and the repeated cross-validation on different metrics

### Final_models/
Files containing the final hyperparameter choices for all the models
