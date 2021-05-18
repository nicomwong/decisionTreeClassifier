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

    def __init__(self, label=None, featureSplit=None, children={}):
        self.label = label
        self.featureSplit = featureSplit    # feature that splits this node's data set
        self.children = children

    def __repr__(self):
        cls = self.__class__
        return f"{cls.__name__}(label={self.label}, featureSplit={self.featureSplit}, children=<FILL>)"


class DecisionTree:

    def __init__(self):
        # Hard-coded
        self._numFeatures = 8
        self._numClasses = 3
        self._featureValues = [
            {0, 1, 2},
            {0, 1},
            {0, 1},
            {0, 1},
            {0, 1},
            {0, 1},
            {0, 1},
            {0, 1}
        ]
        self._featureMean = [None, 0.524, 0.408, 0.140, 0.829, 0.359, 0.181, 0.239]

        # TODO Probably very low accuracy using continuous values; need to handle continuous values by making an _interpretFeatureValue(f, v) function
        # TODO find [s] and [f] and add the interpretation
        # TODO hard-code feature values

        # Hyper-parameters
        self._homogeneousPercentage = 0.99  # percentage of data for a node to be considered "homogeneous"

        self._root = None
        
    def train(self, training_data, training_labels):
        self._preProcess(training_data)
        features = {f for f in range(self._numFeatures) }
        self._root = self._growTree(training_data, training_labels, features)

    def getTestingPredictions(self, testing_data):
        self._preProcess(testing_data)
        predictions = []
        for dataPoint in testing_data:
            predictions.append(self._classify(dataPoint) )
        return predictions

    def printPreOrder(self):
        self._printPreOrderHelper(self._root)

    def _printPreOrderHelper(self, node):
        if not node:
            return
        print(node)
        for child in node.children.values():
            self._printPreOrderHelper(child)

    def _preProcess(self, data):
        "Modifies data such that the feature values are replaced with the interpreted values"
        for ex in data:
            for f, v in enumerate(ex):
                ex[f] = self._interpretFeatureValue(f, v)

    def _interpretFeatureValue(self, feature, value):
        if feature == 0:
            return value
        else:   # split continuous feature value by its mean
            return int(value >= self._featureMean[feature])  # 0: under mean, 1: above mean

    def _classify(self, dataPoint):
        "Classifies a single data point/example"
        return self._decide(self._root, dataPoint)

    def _decide(self, node, dataPoint):
        if node.label is not None:
            return node.label
        
        v = dataPoint[node.featureSplit]
        return self._decide(node.children[v], dataPoint)

    def _growTree(self, data, labels, features):
        "Returns a DecisionTreeNode with proper children given the relevant data, labels, and (remaining) features"
        # print(f"features = {features}")
        # base cases
        if not features:    # No features left to split by
            # print("not features")
            return DecisionTreeNode(label=self._label(labels) )

        if self._homogeneous(labels):   # Data is homogeneous enough to stop
            # print("homogeneous")
            return DecisionTreeNode(label=self._label(labels) )

        s = self._bestSplit(data, labels, features)
        # print(f"s = {s}")

        # split data into subsets according to the feature values of s
        featureValueData = {v: [] for v in self._featureValues[s]}
        featureValueLabels = {v: [] for v in self._featureValues[s]}
        for ex, c in zip(data, labels):
            v = ex[s]
            featureValueData[v].append(ex)
            featureValueLabels[v].append(c)
        
        # recursive step: get children
        children = {}
        for v, dataSubset in featureValueData.items():
            labelsSubset = featureValueLabels[v]

            if dataSubset:
                children[v] = self._growTree(dataSubset, labelsSubset, features - {s})
            else:   # Empty subset, so use parent's label
                children[v] = DecisionTreeNode(label=self._label(labels) )

        return DecisionTreeNode(featureSplit=s, children=children)

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

    def _bestSplit(self, data, labels, features):
        "Returns the feature from the given set of features which best splits the data"
        totalFeatureImpurity = {}
        
        for f in features:

            # labels_f[v]: sublist of labels whose data has f=v
            labels_f = {v: [] for v in self._featureValues[f] }
            for ex, c in zip(data, labels):
                v = ex[f]
                labels_f[v].append(c)

            # Calculate impurity for each feature value path
            featureValueImpurity = {}
            for v in self._featureValues[f]:
                featureValueImpurity[v] = self._impurity(labels_f[v])

            featureValueCount = {v: len(labels_f[v]) for v in self._featureValues[f]}

            totalFeatureImpurity[f] = self._getTotalImpurity(f, featureValueImpurity, featureValueCount)

        # print(f"totalFeatureImpurity: {totalFeatureImpurity}")
        return min(totalFeatureImpurity, key=totalFeatureImpurity.get)  # key with min value

    def _getTotalImpurity(self, feature, impurity, count):
        return 1/sum(count.values() ) * sum( count[v] * impurity[v] for v in self._featureValues[feature] )

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


# # _getTotalImpurity : PASSED

# dt._featureValues = [ {0, 1, 2} ]
# impurities = {0: 0.65, 1: 0.43, 2: 0.25}
# counts = {0: 3, 1: 6, 2: 2}
# print( dt._getTotalImpurity(0, impurities, counts) )   # 0.46
# dt._featureValues = [ {0, 1} ]
# impurities = {0: 0.23, 1: 0.99}
# counts = {0: 5, 1: 6}
# print( dt._getTotalImpurity(0, impurities, counts) )   # 0.64


# # _bestSplit : PASSED

# dt._numFeatures = 3
# dt._numClasses = 2
# dt._featureValues = [ {0, 1}, {0, 1}, {0, 1} ]

# data =  [   [1, 0, 0],
#             [0, 0, 0],
#             [1, 1, 1],
#             [0, 0, 1],
#             [0, 1, 1],
#             [0, 1, 0],
#             [1, 0, 1]
# ]
# labels = [0, 0, 0, 0, 1, 1, 1]
# features = {0, 1, 2}

# print( dt._bestSplit(data, labels, features) )    # impurities = [0.476, 0.405, 0.476] => feature 1


# # _growTree : PASSED

# dt._numFeatures = 3
# dt._numClasses = 2
# dt._featureValues = [ {0, 1}, {0, 1}, {0, 1} ]

# data =  [   [1, 0, 0],
#             [0, 0, 0],
#             [1, 1, 1],
#             [0, 0, 1],
#             [0, 1, 1],
#             [0, 1, 0],
#             [1, 0, 1]
# ]
# labels = [0, 0, 0, 0, 1, 1, 1]
# features = {0, 1, 2}
# import sys
# sys.setrecursionlimit(100)
# dt._root = dt._growTree(data, labels, features)
# dt.printPreOrder()
# # featureSplit preorder: [1, 0, 2, 0]
# # label preorder: [0, 0, 1, 1, 0]


# # _classify -> _decide : PASSED

# dt._numFeatures = 3
# dt._numClasses = 2
# dt._featureValues = [ {0, 1}, {0, 1}, {0, 1} ]

# data =  [   [1, 0, 0],
#             [0, 0, 0],
#             [1, 1, 1],
#             [0, 0, 1],
#             [0, 1, 1],
#             [0, 1, 0],
#             [1, 0, 1]
# ]
# labels = [0, 0, 0, 0, 1, 1, 1]

# dt.train(data, labels)
# print( dt._classify([0, 0, 0]) )    # 0
# print( dt._classify([0, 0, 1]) )    # 0
# print( dt._classify([1, 0, 0]) )    # 0
# print( dt._classify([1, 0, 1]) )    # 1
# print( dt._classify([0, 1, 0]) )    # 1
# print( dt._classify([0, 1, 1]) )    # 1
# print( dt._classify([1, 1, 0]) )    # 0
# print( dt._classify([1, 1, 1]) )    # 0


# # _preProcess -> _interpretFeatureValue : PASSED

# data =  [
#     [
#         1,
#         0.64,
#         0.5,
#         0.18,
#         1.4995,
#         0.593,
#         0.314,
#         0.431
#     ],
#     [
#         2,
#         0.44,
#         0.345,
#         0.115,
#         0.545,
#         0.269,
#         0.111,
#         0.1305
#     ],
#     [
#         1,
#         0.62,
#         0.48,
#         0.17,
#         1.1045,
#         0.535,
#         0.25,
#         0.287
#     ]
# ]

# processedData = [
#     [
#         1,
#         1,
#         1,
#         1,
#         1,
#         1,
#         1,
#         1
#     ],
#     [
#         2,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0
#     ],
#     [
#         1,
#         1,
#         1,
#         1,
#         1,
#         1,
#         1,
#         1
#     ]
# ]
# dt._preProcess(data)
# print(data)
# print(processedData)
# print(data == processedData)    # true
