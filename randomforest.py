# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#we are going to import our functions from decisiontrees.py
import decisiontrees as dt

# %%
#function: build a random forest
def BuildRandomForest(train_data, trees, subset_frac = 0.7, min = 5, frac = 0.9, maxDepth = 20):
    """ 
    BuildRandomForest() uses BuildTree() to build the a random forest.
    It takes the following inputs:
    - train_data: the data used to build the tree, as a pandas dataframe
    - trees: an integer corresponding to the number of trees to make
    - min: a parameter given to BuildTree() that sets the minimum number of datapoints needed to make a leaf, default is 5
    - frac: a parameter given to BuildTree() that sets the minimum fraction of prediction needed to make a leaf, default is 0.9
    - maxDepth: a parameter given to BuildTree() that sets the maximum depth of the tree, default is 20
    The function returns a list of trees
    """
    forest = []
    for i in range(trees):
        current_data = GetRandomSubset(train_data, subset_frac = subset_frac)
        current_root = dt.MakeNode(current_data, 1)
        current_tree = dt.BuildTree(current_root, current_data, 0, min = min, frac = frac, maxDepth = maxDepth)
        forest.append(current_tree)

    return forest

def GetRandomSubset(data, subset_frac):
    """ 
    GetRandomSubset() is a function that takes in a pandas dataframe and returns 
    a new pandas dataframe with a subset of the original variables.

    It operates heuritstically and gets the nearest integer for the number of 
    variables to select. 
    """
    cols = len(data.columns) - 1 #gives us the upper bound, and we exclude the last which is class
    n = int(subset_frac * (cols + 1))
    topick = []
    
    current_rand = np.random.randint(cols)
    topick.append(current_rand)
    while len(topick) < n:
        current_num = np.random.randint(cols)
    
        #check if exist
        e = False
        for j in range(len(topick)):
            if current_num == topick[j]:
                e = True

        if e != True: 
            topick.append(current_num)
    
    #we also want to append the last column
    topick.append(cols)
    newsubset = data[data.columns[topick]]
    return newsubset
    
def ForestPredict(forest, test_data):
    """
    ForestPredict() is a function that takes a prebuilt forest from BuildForest()
    and the data for which to predict. It returns a list of predictions by the
    forest for each point in the testdata.
    """
    n = len(forest)
    preds = []
    for i in range(len(test_data)): #run for each data point
        tox = 0
        non = 0
        #for each tree, run a prediction
        for j in range(n):
            current_res = dt.PredictOne(forest[j], test_data.iloc[i])
            if current_res == "Toxic":
                tox += 1
            else:
                non += 1
        
        #take the majority rule
        if tox >= non:
            preds.append("Toxic")
        else:
            preds.append("NonToxic")

    return preds


