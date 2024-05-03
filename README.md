# 02620_ml_group5
Collection of code files used for 02-620 Machine Learning for Scientists project by Anagha Anil, Leila Michal, Brian Zhang, and Jonathan Zhu.

## Instructions for use
All relevant code is in Jupyter Notebooks whose cells can simply be executed in order, after which results are visible. In this project, the relevant code includes logistic regression, recursive feature elimination, decision trees, random forests, and gradient boosting. 

Recursive feature elimination and logistic regression functions are contained in the file 'FeatureEliminationLogisticRegression.ipynb'. Parameters for recreating accuracy metrics are provided in the file. To test functions, simply download either data.csv or toxicity_data.csv from the toxicity-2 data archive folder in this repository, then read in  data.csv prior to running the models.

Decision trees and random forests have their underying functions coded in 'decisiontrees.py' and 'randomforest.py', while the actual prediction process is found in 'predict_trees_forest.ipynb'. Additionally, the selected parameters used to build the models can be found in the folder 'dtrf_params' should you want to remake the models.

Gradient boosting is found in 'GradientBoosting.ipynb' and 'GradientBoostSklearnFeatureElimination.ipynb'.
