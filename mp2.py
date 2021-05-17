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

    def __init__(self, label=None, featureSplit=None, children=[]):
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
        self._featureValues = []    # TODO

        # Hyper-parameters
        self._homogeneousPercentage = 0.99  # percentage of data for a node to be considered "homogeneous"

        self._root = None
        
    def train(self, training_data, training_labels):
        self._root = self._growTree(training_data, training_labels)

    def test(self, testing_data):
        pass

    def printPreOrder(self):
        self._printPreOrderHelper(self._root)

    def _printPreOrderHelper(self, node):
        if not node:
            return
        print(node)
        for child in node.children:
            self._printPreOrderHelper(child)

    def _classify(self, data_point):
        "Classifies a single data point/example"
        pass

    def _growTree(self, data, labels):
        if self._homogeneous(labels):
            return DecisionTreeNode(label=self._label(labels) )

        s = self._bestSplit(data, labels)

        # split data into subsets according to the feature values of s
        featureValueData = {v: [] for v in self._featureValues[s]}
        featureValueLabels = {v: [] for v in self._featureValues[s]}
        for ex, c in zip(data, labels):
            v = ex[s]
            featureValueData[v].append(ex)
            featureValueLabels[v].append(c)
        
        # recursive step: get children
        children = []
        for v, dataSubset in featureValueData.items():
            labelsSubset = featureValueLabels[v]

            if dataSubset:
                children.append(self._growTree(dataSubset, labelsSubset) )
            else:   # Empty subset, so use parent's label
                children.append(DecisionTreeNode(label=self._label(labels)) )

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

    def _bestSplit(self, data, labels):
        "Returns the feature that best splits D"
        totalFeatureImpurity = []
        
        for f in range(self._numFeatures):
            featureValueImpurities = []

            # labels_f[v]: sublist of labels whose data has f=v
            labels_f = {v: [] for v in self._featureValues[f] }

            for v in self._featureValues[f]:
                for ex, c in zip(data, labels):
                    if ex[f] == v:
                        labels_f[v].append(c)

                featureValueImpurities.append(self._impurity(labels_f[v]) )

            # print(f"labels_f: {labels_f}")
            # print(f"featureValueImpurities: {featureValueImpurities}")

            totalFeatureImpurity.append(self._getTotalImpurity(featureValueImpurities, [len(labels_f[v]) for v in self._featureValues[f] ]) )

        # print(f"totalFeatureImpurity: {totalFeatureImpurity}")
        return totalFeatureImpurity.index( min(totalFeatureImpurity) )

    def _getTotalImpurity(self, impurities, counts):
        return 1/sum(counts) * sum( count * impurity for count, impurity in zip(counts, impurities) )

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

# impurities = [0.65, 0.43, 0.25]
# counts = [3, 6, 2]
# print( dt._getTotalImpurity(impurities, counts) )   # 0.46
# impurities = [0.23, 0.99]
# counts = [5, 6]
# print( dt._getTotalImpurity(impurities, counts) )   # 0.64


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

# print( dt._bestSplit(data, labels) )    # impurities = [0.476, 0.405, 0.476] => feature 1


# # train -> _growTree : PASSED

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
# dt.printPreOrder()
# # featureSplit preorder: [1, 0, 2, 0]
# # label preorder: [0, 0, 1, 1, 0]