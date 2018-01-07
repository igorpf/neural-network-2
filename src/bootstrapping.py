#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import randint
# bootstraping

# matrix
# classIndex -> which position is the class predicted? default: last
# n -> number of bootstraps
# r -> bootstraps size
# useIndex -> create bootstraps of the indices instead of data. Use for testing purposes
def bootstrap(matrix, classIndex=-1, n=5,r=None, useIndex=False):    
    # list of tuples in the form of (training set, test set)   
    matrixSize = len(matrix)
    if r==None:
        r = matrixSize
    bootstraps = []
    for b in range(n):
        #keep track of the rows not selected (the ones that will be used for testing)
        rowCount = {}
        training = []       
        for iteration in range(r):
            index = randint(0, matrixSize-1)
            rowCount[index] = rowCount.get(index, 1)
            training.append(index if useIndex else matrix[index])
            
        test = []
        notUsedRows = filter(lambda k: rowCount.get(k,0) == 0, [i for i in range(0,matrixSize)] )
        for row in notUsedRows:
            test.append(row if useIndex else matrix[row])
        bootstraps += [(training, test)]
    return bootstraps