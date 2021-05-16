# Starter code for CS 165B HW3
import numpy

def run_train_test(training_data, training_labels, testing_data):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: List[List[float]]
        training_label: List[int]
        testing_data: List[List[float]]

    Output:
        testing_prediction: List[int]
    Example:
    return [1]*len(testing_data)
    """
    return [1]*len(testing_data)

    #TODO implement the decision tree and return the prediction
    # Create decision tree model
    
    # Run model predictor on testing data set


class DecisionTreeNode:

    def __init__(self, label, featureSplit):
        self.label = label
        self.featureSplit = featureSplit    # feature that splits this node's data set
        self.children = []


class DecisionTree:

    def __init__(self):
        # Hard-coded
        self.numberOfFeatures = 8
        
    def train(training_data, training_labels):
        pass

    def test(testing_data):
        pass

    def _classify(data_point):
        "Classifies a single data point/example"
        pass

    def _growTree(D):
        if homogeneous(D) then return (Node with Label(D) )
        S = bestSplit(D)
        split D into subsets D_i according to the literals in S
        for each i:
            if D_i is not empty then T_i = self(D_i)
            else T_i = (Node with label=Label(D) )
        
        return (Node with label=None, featureSplit=S, children T_i)

    def homogeneous(D):
        "Returns True iff D is at least 99% homogeneous"
        for each feature f, count the number of examples with feature f and store them in a List
        return True iff a feature f is at least 99% |D|

    def label(D):
        "Returns the label of D by majority vote"
        return the feature f whose count in D is greatest

    def bestSplit(D):
        "Returns the feature that best splits D"
        for each feature f:
            for each feature value v:
                D_f_v = subset of D with f=v
                valueImpurity[v] = Impurity(D_f_v)

            featureTotalImpurity[f] = 1/|D| * sum (i=1, L) [ |D_i| * valueImpurity[i] ]

        return feature f with least featureTotalImpurity

    def Impurity(D):
        featureProbability = dictionary from feature f to count(f, D) / |D|
        return sum (i=1, K) [ Gini(p_i) ]

    def Gini(p):
        return p * (1 - p)