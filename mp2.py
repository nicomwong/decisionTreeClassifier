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
        self._numFeatures = 8
        self._numClasses = 3
        self._featureValues = []    # TODO

        # Hyper-parameters
        self._homogeneousPercentage = 0.99  # percentage of data for a node to be considered "homogeneous"

        self._root = None
        
    def train(self, training_data, training_labels):
        self._root = self._growTree(training_data, training_labels)

    def test(self, testing_data):
        pass

    def _classify(self, data_point):
        "Classifies a single data point/example"
        pass

    # def _growTree(self, data, labels):
    #     if homogeneous(D) then return (Node with Label(D) )
    #     S = bestSplit(D)
    #     split D into subsets D_i according to the literals in S
    #     for each i:
    #         if D_i is not empty then T_i = self(D_i)
    #         else T_i = (Node with label=Label(D) )
        
    #     return (Node with label=None, featureSplit=S, children T_i)

    def _homogeneous(self, labels):
        "Returns True iff data is at least 99% homogeneous"
        classProbability = self._getClassProbabilities(labels)
        
        for c, p in enumerate(classProbability):
            if p >= self._homogeneousPercentage:
                return True

        return False

    def _label(self, labels):
        "Returns the label of D by majority vote"
        classProbability = self._getClassProbabilities(labels)
        return classProbability.index( max(classProbability) )

    # def _bestSplit(self, data, labels):
    #     "Returns the feature that best splits D"
    #     totalFeatureImpurity = []
        
    #     for f in self._numFeatures:
    #         featureValueImpurity = {}

    #         # labels_f : feature value v -> subset of labels whose data has f=v
    #         labels_f = {v: {} for v in self._featureValues[f] }

    #         for v in self._featureValues[f]:
    #             for ex, c in zip(data, labels):
    #                 if ex[f] == v:
    #                     labels_f[v].add(c)

    #             featureValueImpurity[v] = self._impurity(labels_f_v)

    #         totalFeatureImpurity.append( 1/len(data) * sum( len(labels_f[v]) * featureValueImpurity[v] for v in self._featureValues[f] ) )

    #     return totalFeatureImpurity.index( min(totalFeatureImpurity) )

    def _impurity(self, labels):
        classProbability = self._getClassProbabilities(labels)
        return sum(self._giniIndex(p) for p in classProbability)

    def _giniIndex(self, p):
        return p * (1 - p)

    def _getClassProbabilities(self, labels):
        classProbability = [0 for i in range(self._numClasses) ]
        for c in labels:
            classProbability[c] += 1 / len(labels)
        return classProbability

dt = DecisionTree()

# # _homogeneous and _label tests : PASSED

# labels = [0, 1, 2, 0, 1, 2, 0]

# print( dt._homogeneous(labels) )    # false
# print( dt._label(labels) )          # 0

# labels = [0, 0, 0, 0, 0, 0, 0]
# print( dt._homogeneous(labels) )    # true
# labels = [0, 0, 0, 0, 0, 0, 1]
# print( dt._homogeneous(labels) )    # false
# labels = [0] * 99 + [1]
# print( dt._homogeneous(labels) )    # true


# # _impurity and _giniIndex tests : PASSED

# labels = [0, 1, 2, 0, 1, 2, 0]

# print( dt._getClassProbabilities(labels) )  # [0.43, 0.29, 0.29]
# print( dt._impurity(labels) )   # 0.65
