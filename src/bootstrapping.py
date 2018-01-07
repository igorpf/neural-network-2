#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import randint, shuffle
# bootstrapping

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

# k-folding

# proportions -> list of proportion of each class ([0,1])
# classIndex -> which position is the class predicted? default: last
def fold(matrix, proportions, classIndex=-1,k=5):    
    groups = {}
    for i, row in enumerate(matrix):
        key = row[classIndex]
        groups[key] = groups.get(key, []) + [row]
    #randomness is introduced here to simplify the algorithm
    for key in groups:
        shuffle(groups[key])

    size = len(matrix)/k
    folds = []
    
    for iteration in range(k):    
        fold = []       
        for i,prop in enumerate(proportions):
            s = int(prop * size)
            key = float(i+1)
            fold += groups[key][:s]
            groups[key] = groups[key][s:]
        folds+= [fold]
    #Some elements may be left after this procedure, so we want to distribute them equally to the folds
    anyElementsLeft = True
    nextFold = 0
    while anyElementsLeft:
        anyElementsLeft = False
        for key in groups:
            if len(groups[key]) == 0:
                continue
            anyElementsLeft = True
            folds[nextFold].append(groups[key].pop()) 
            nextFold = (nextFold+1) % k
    return folds