#!/usr/bin/env python
# -*- coding: utf-8 -*-

from files import files
from random import randint
from main import preprocessing, DecisionTree, DataSet
from bootstrapping import bootstrap, fold

def flatten(l):
    return [item for sublist in l for item in sublist]

class Ensemble(object):
    def __init__(self, file, ntree):
        self.file = file
        self.ntree = ntree
    def generateTree(self, x,y, attrList, possibleValuesList):
        dt = DecisionTree(x,y, attrList, possibleValuesList, int(len(x[0])**0.5)) # **1 for test dataset, **0.5 for the other ones
        dt.training(possibleValuesList)
        # dt.printTree()
        return dt
    # - divide 80/20
    # - 
    # - cria ntree bootstraps (dentro dos 80)
    # - for bootstrap in bootstraps
    #     - cria uma árvore
    #     - treina com o bootstrap[train]
    #     - calcula a performance usando bootstrap[teste]
    # - for test in tests:
    #     - classifica entre todas as árvores e faz votação
    # - 

    # k folds
    def crossValidation(self, f, ds, folds):
        k = len(folds)        
        accuracies = []
        for i in range(k):
            train = flatten(folds[:i] + folds[i+1:])
            test = folds[i]
            trees = self.createRandomForest(f,ds,train)
            correct, total = self.evaluatePrediction(trees, test)
            # print "\nPrediction: {} out of {}".format(correct, total)
            accuracies.append(correct/float(total)) 
        return self.getForestPerformance(accuracies)

    def createRandomForest(self, f, ds, train):
        bootstraps = bootstrap(matrix=train, n = self.ntree)
        trees = []
        accuracies = []
        for i, boots in enumerate(bootstraps):
            ds.dataMatrix = boots[0]
            x, y, attrList, possibleValuesList = preprocessing(f,ds)
            tree = self.generateTree(x, y, attrList, possibleValuesList)        
            trees.append(tree)      
        return trees

    def getForestPerformance(self, accuracies):
        # print accuracies
        mean = reduce(lambda x,y: x+y, accuracies) / float(len(accuracies))
        stdDeviation = (reduce(lambda x,y: x+y, 
                            map(lambda x: (x-mean) ** 2, accuracies)) / (len(accuracies)-1) ) ** 0.5
        return mean, stdDeviation
    
    def evaluatePrediction(self, trees, tests):
        correct = 0
        for test in tests:            
            prediction = self.getForestPrediction(trees, test[:-1])
            # print "\nPrediction: {}, real: {}".format(prediction, test[-1])
            if prediction == test[-1]:
                correct += 1
        # print "\nPrediction: {} out of {}".format(correct, len(tests))
        return correct, len(tests)

    def getForestPrediction(self, trees, data):
        votes = {}
        for tree in trees:
            prediction = tree.predict(data)
            votes[prediction] = votes.get(prediction, 0) + 1
        return max(votes)
            
