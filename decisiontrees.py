# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
class Node:
    #defining the class of nodes for this case.
    def __init__(self, name, var, thres, res):
        self.id = name #the node number
        self.var = var #the feature on which we split
        self.thres = thres #the split threshold
        self.c1 = None
        self.c2 = None
        self.res = res

    def __str__(self):
        return f"Node {self.id}: Split on {self.var} with threshold {self.thres} or return result {self.res} \nLeft Child of {self.id}: {self.c1} \nRight Child of {self.id}: {self.c2}"
    
    #need to establish children methodologies
    def insert_left_child(self, data, id):
        #self.c1 = Node(new_var, new_thres, new_res)
        self.c1 = MakeNode(data, id)
    
    def insert_right_child(self, data, id):
        self.c2 = MakeNode(data, id)

    def depth(self):
        #get the depth of the tree
        #base case: leaf
        if self.c1 == None and self.c2 == None:
            return 1
        #recursive case: get depth of both subtrees and pick larger
        if self.c1 != None:
            d1 = self.c1.depth()
        if self.c2 != None:
            d2 = self.c2.depth()
        if d2 > d1:
            return d2
        else:
            return d1

#a few functions to define information gain, a key aspect of splitting.
def InformationGain(varname, data):
    """
    InformationGain() takes in a variable name string on which to split, and the data on which
    to run the decision tree as a pandas dataframe. It returns the
    information gain for splitting on that variable, and for numerical variables,
    it returns the threshold on which it splits. 
    """
    
    #first, decide the split
    thres = GetThreshold(varname, data)
    
    #now what we are going to do is split based on this threshold and
    #calculate the entropy for each one.
    s1 = data[data[varname] >= thres]
    s2 = data[data[varname] < thres]
    v1 = list(s1["Class"])
    v2 = list(s2["Class"])
    v = list(data["Class"])
    
    n1 = len(v1)
    n2 = len(v2)
    n = len(v)
    #get entropies
    e1 = Entropy(v1)
    e2 = Entropy(v2)
    e = Entropy(v)

    #now we do a weighted average
    tot = e1 * (float(n1) / float(n)) + e2 * (float(n2) / float(n))
    gain = e - tot
    
    return gain, thres

def GetThreshold(varname, data):
    """
    GetThreshold() takes in a variable name string on which to split, and the data on which to split on.
    It returns the threshold of splitting for that variable. Note that this is a heuristic
    method that determines the split based on the mean between the minimum of one class and the
    maximum of the other outcome class; it may not give optimal split results.
    """
    #this is a heuristic method that will determine the split based on the mean
    #between the minimum value of one class and the maximum of another class.
    #it may not be good, but since we are testing on many variables, we are likely
    #to get a decent result.
    cat1 = data[data["Class"] == "Toxic"]
    cat2 = data[data["Class"] == "NonToxic"]

    vals1 = np.array(list(cat1[varname]))
    vals2 = np.array(list(cat2[varname]))
    if len(vals1) == 0 and len(vals2) == 0:
        return -100000
    if len(vals1) == 0:
        return np.min(vals2)
    if len(vals2) == 0:
        return np.min(vals1)

    min1 = np.min(vals1)
    max1 = np.max(vals1)
    min2 = np.min(vals2)
    max2 = np.max(vals2)

    #form split
    thres = (min1 + max2) / 2.0
    return thres

def Entropy(outcomes):
    """
    Entropy() takes a list of outcomes and returns the calculated conditional entropy.
    In the case of this project, the outcomes are all 'Toxic' or 'NonToxic'.
    """

    n = len(outcomes)

    #determine the number of toxic
    n_toxic = 0
    for i in range(n):
        if outcomes[i] == "Toxic":
            n_toxic += 1
    
    #nontoxic is easy from there
    n_nontoxic = n - n_toxic

    #and to do the rest we simply base off the entropy formula
    #but we need to check for zeroes too
    if n_toxic == 0 or n_nontoxic == 0:
        #we have zero entropy if we only have one class
        return 0
    
    f1 = float(n_toxic) / float(n)
    f2 = float(n_nontoxic) / float(n)

    p1 = -1 * f1 * np.log2(f1)
    p2 = -1 * f2 * np.log2(f2)
    e = p1 + p2

    return e

# %%
#a function to get the maximum information gain out of all variables. 
def GetBestSplit(data):
    vars = list(data.columns)
    n = len(vars)
    features = vars[:n-1] #removes the last column, the class.

    gain_list = []
    thres_list = []
    #loop through the features
    for f in features:
        if f != "Class": #a double check just to be sure
            currentGain, currentThres = InformationGain(f, data)
            gain_list.append(currentGain)
            thres_list.append(currentThres)

    best_index = np.argmax(np.array(gain_list))
    best_var = features[best_index]
    best_gain = gain_list[best_index]
    best_thres = thres_list[best_index]

    return best_var, best_gain, best_thres    

# %%
#initializing a node
def MakeNode(data, n):
    feature, gain, threshold = GetBestSplit(data)
    root = Node(n, feature, threshold, None)
    return root

#initializing a leaf
def MakeLeaf(data, n):
    numToxic = len(data[data["Class"] == "Toxic"])
    numNonToxic = len(data[data["Class"] == "NonToxic"])
    if numToxic > numNonToxic:
        leaf = Node(n, None, None, "Toxic")
    else:
        leaf = Node(n, None, None, "NonToxic")

    return leaf

# %%
#function to build the tree
def BuildTree(root, data, currentIter, min = 5, frac = 0.9, maxDepth = 20):
    """BuildTree() takes in the following inputs:
    - the root of the tree
    - the data for the tree, which is the full set at the beginning
    - the minimum number of items required to make a split
    - the minimum proportion of class to make a leaf
    - current iteration: set to 0 when first called.
    It returns the root of the tree, hopefully with all its branches extended.
    """
    #base case: we have less than the minimum number
    #or we have a fraction less than the threshold
    #in this case we make a leaf
    numToxic = len(data[data["Class"] == "Toxic"])
    numNonToxic = len(data[data["Class"] == "NonToxic"])
    prop_toxic = float(numToxic) / float(len(data))
    prop_nontox = float(numNonToxic) / float(len(data))
    if len(data) < min:
        root = MakeLeaf(data, root.id)
    elif prop_toxic > frac or prop_nontox > frac:
        root = MakeLeaf(data, root.id)
    elif currentIter >= maxDepth:
        root = MakeLeaf(data, root.id)
    else:
        #recursive case: we do a split
        #root = MakeNode(data, root.id)
        left_split = data[data[root.var] < root.thres]
        right_split = data[data[root.var] >= root.thres]

        root.insert_left_child(left_split.drop(columns = root.var), 2*root.id)
        root.c1 = BuildTree(root.c1, left_split.drop(columns = root.var), currentIter+1, min, frac, maxDepth)
        
        root.insert_right_child(right_split.drop(columns = root.var), 2*root.id+1)
        root.c2 = BuildTree(root.c2, right_split.drop(columns = root.var), currentIter+1 , min, frac, maxDepth) 

    return root 

# %%
def Predict(root, test_data):
    """
    Predict() is a function that uses the root of a pre-built tree and a pandas dataframe
    of testing data; these form the inputs. For each row in the dataframe, it
    traverses the tree and returns the result, storing it in a list of predicted
    outcomes.
    """
    outcomes = []
    for i in range(len(test_data)):
        current_point = test_data.iloc[i]
        current_outcome = PredictOne(root, current_point)
        outcomes.append(current_outcome)

    return outcomes

def PredictOne(root, dp):
    """ 
    PredictOne() is a subroutine of Predict() that takes in a single point and
    traverses the tree to predict it, returning the prediction.
    """
    #this is done through recursion.
    #base case: root is a leaf, in which case result != None
    if root.res != None:
        return root.res
    else:
        #recursive case; decide by threshold.
        current_val = dp[root.var]
        if current_val < root.thres:
            return PredictOne(root.c1, dp)
        else:
            return PredictOne(root.c2, dp)

# %%
def GetConfusionMatrix(data, pred):
    """ 
    GetMetrics() takes in a dataframe containing the true values of the predicted points
    in the column 'Class', and the list of predicted values given by the Predict() function.
    It returns the values for the confusion matrix in a list in the order:
    true pos, false pos, true neg, false neg.
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    truth = list(data["Class"])

    #critically, we assume we get the same number of predictions.
    #if we don't, we may get an error, or not get the right values.
    if len(truth) != len(pred):
        print("Warning: Truth values and predicted values inequal length")
    for i in range(len(truth)):
        if truth[i] == pred[i]: #true prediction
            if truth[i] == "Toxic":
                tp += 1
            else:
                tn += 1
        else: #false prediction
            if truth[i] == "Toxic":
                fp += 1
            else:
                fn += 1
    
    return tp, fp, tn, fn

def MakeTuningGrid(n = 50, min_min = 0, min_max = 15, frac_min = 0.0,
                   frac_max = 1.0, depth_min = 10, depth_max = 50):
    """
    MakeTuningGrid() makes a 2-d list, which you can convert into a
    numpy array, of random numbers to use for tuning a decision tree.
    It requires no parameters but you can set the range for the following:
    - n: integer, the number of rows to define. Default is 50.
    - min: integer, the minimum needed to make a split in the tree
    - frac: float between 0 and 1, the minimum fraction to define a leaf
    - maxDepth: integer, the maximum depth for the tree
    """
    grid = []
    for i in range(n):
        #random int sampling in range: easy
        current_min = np.random.randint(low = min_min, high = min_max)
        current_depth = np.random.randint(low = depth_min, high = depth_max)

        #random float sampling in range: harder
        current_frac = 0.0
        inRange = False
        while inRange == False:
            current_frac = np.random.rand()
            if current_frac > frac_min and current_frac < frac_max:
                inRange = True

        params = [current_min, current_frac, current_depth]
        grid.append(params)

    return grid
