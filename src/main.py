#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from files import files
from random import randint, seed
from operator import itemgetter
import math
import sys

# Seed to keep results deterministic
#seed(9001)

class DataSet():
    def __init__(self, file):
        self.dataMatrix = []
        self.generateDataMatrix(file.fileName)
        #self.normalizeFeatures(file.normRanges)

    def generateDataMatrix(self, file):
        with open(file) as file:
            for i, line in enumerate(file):
                line = line.rstrip("\n")
                words = map(lambda x: float(x), line.split(","))
                self.dataMatrix += [words]

    def normalizeFeatures(self, ranges=None):
        if ranges==None:
            self.ranges = self.defineRanges()
            print self.ranges
        else:
            self.ranges = ranges

        for i, row in enumerate(self.dataMatrix):
            for j, element in enumerate(self.dataMatrix[i]):
                bounds = self.ranges[j]
                if bounds == None:
                    continue
                self.dataMatrix[i][j] = (element-bounds[0])/(bounds[1]-bounds[0])

    def defineRanges(self):
        ranges = []
        for i in range(0, len(self.dataMatrix[0])):
            column = map(lambda x:x[i],self.dataMatrix)
            ranges += [[min(column),max(column)]]
        return ranges

class DecisionTree():
    def __init__(self, x, y, attrList, possibleValuesList, m):
        self.children = []
        self.x = x
        self.y = y
        self.attrList = attrList
        self.possibleValuesList = possibleValuesList
        self.m = m
        self.isPure = False
        self.predictedClass = -1
        self.gain = 0
        self.selectedAttribute = -1

    def training(self):
        originalAttrList = map(lambda x:x, self.attrList)
        while len(self.attrList) > self.m:
            del self.attrList[randint(0,len(self.attrList) - 1)]
        gains = []
        for attribute in self.attrList:
            groups = []
            if len(self.possibleValuesList[attribute]) == 1: # numerical attributes
                #print attribute
                #print "antes ",
                #print self.possibleValuesList[attribute][0]
                #self.possibleValuesList[attribute][0] = calculateBestThreshold(self.x, self.y, attribute) # Comment this line to use the average
                #print "depois ",
                #print self.possibleValuesList[attribute][0]
                groups += [[]]
                groups += [[]]
                for i in range(len(self.x)):
                    if(self.x[i][attribute] <= self.possibleValuesList[attribute][0]):
                        groups[0].append(self.y[i])
                    else:
                        groups[1].append(self.y[i])
                #print groups
            else:
                for value in self.possibleValuesList[attribute]:
                    group = []
                    for i in range(len(self.x)):
                        if(self.x[i][attribute] == value):
                            group.append(self.y[i])
                    groups += [group]
            gains += [(attribute, gain(self.y, groups))]

        #print gains
        bestGain = -1000
        bestAttribute = 0
        for g in gains:        
            if g[1] > bestGain:
                bestGain = g[1]
                bestAttribute = g[0]
        self.gain = bestGain
        self.selectedAttribute = bestAttribute

        attrList_children = []
        for i in originalAttrList:
            if i != bestAttribute:
                attrList_children.append(i)

        if len(self.possibleValuesList[bestAttribute]) > 1: # categorical attributes
            for value in self.possibleValuesList[bestAttribute]:
                x_child = []
                y_child = []
                for i in range(len(self.x)):
                    if(self.x[i][bestAttribute] == value):
                        x_child.append(self.x[i])
                        y_child.append(self.y[i])
                self.children += [DecisionTree(x_child, y_child, attrList_children, self.possibleValuesList, self.m)]
        else: # numerical attributes
            x_child = []
            y_child = []
            for i in range(len(self.x)):
                if(self.x[i][bestAttribute] <= self.possibleValuesList[bestAttribute][0]):
                    x_child.append(self.x[i])
                    y_child.append(self.y[i])
            self.children += [DecisionTree(x_child, y_child, attrList_children, self.possibleValuesList, self.m)]
            x_child = []
            y_child = []
            for i in range(len(self.x)):
                if(self.x[i][bestAttribute] > self.possibleValuesList[bestAttribute][0]):
                    x_child.append(self.x[i])
                    y_child.append(self.y[i])
            self.children += [DecisionTree(x_child, y_child, attrList_children, self.possibleValuesList, self.m)]
       
        for child in self.children:
            if len(child.attrList) > 0 and len(set(child.y)) > 1:
                child.training()
            else:
                child.isPure = True
                if not child.y: # child is empty
                    child.predictedClass = max(set(self.y), key=self.y.count)
                elif len(set(child.y)) == 1: # child is pure
                    child.predictedClass = child.y[0]
                else: #attrList is empty
                    child.predictedClass = max(set(child.y), key=child.y.count)

    def predict(self, example):
        if self.isPure:
            return self.predictedClass
        else:
            if len(self.possibleValuesList[self.selectedAttribute]) == 1:
                if example[self.selectedAttribute] <= self.possibleValuesList[self.selectedAttribute][0]:
                    return self.children[0].predict(example)
                else:
                    return self.children[1].predict(example)
            else:
                for child in range(len(self.children)):
                    if example[self.selectedAttribute] == self.possibleValuesList[self.selectedAttribute][child]:
                        return self.children[child].predict(example)

    def printTree(self, level = 0):
        if self.isPure:
            print "class =", int(self.predictedClass),
        else:
            print "Attribute for division: " + str(self.selectedAttribute) + " (gain: " + str(self.gain) + ")",
            for child in range(len(self.children)):
                print "\n",
                for i in range(level+1):
                    print "    ",
                self.children[child].printTree(level+1)  

# ds can be passed as an argument if the dataset
# needs to be previously processed (to separate 80/20, for instance)
def preprocessing(f, ds=None):
    if ds == None:
        ds = DataSet(f)
    x = [[ds.dataMatrix[i % len(ds.dataMatrix)][j] for j in range(len(ds.dataMatrix[0]) - 1)] for i in range(len(ds.dataMatrix))]
    y = [ds.dataMatrix[i % len(ds.dataMatrix)][-1] for i in range(len(ds.dataMatrix))]
    attrList = []
    possibleValuesList = []
    for i in range(len(x[0])):
        attrList.append(i)
        if f.numericalAttributes[i] == 0:
            possibleValues = []
            for instance in x:
                if not (instance[i] in possibleValues):
                    possibleValues.append(instance[i])
            possibleValuesList += [possibleValues]
        else:
            total = 0
            for instance in x:
                total += instance[i]
            possibleValuesList += [[total / len(x)]]
    return x, y, attrList, possibleValuesList

def entropy(group):
    possibleClasses = []
    for i in group:
        if not (i in possibleClasses):
            possibleClasses.append(i)
    info = 0
    for i in possibleClasses:
        info -= (float(group.count(i))/len(group)) * math.log(float(group.count(i))/len(group), 2)
    return info

def gain(y, groups):
    infoA = 0
    for group in groups:
        infoA += (float(len(group)) / len(y)) * entropy(group)
    info = entropy(y)
    return info - infoA

def calculateBestThreshold(x, y, attributeIndex):
    for i in range(len(y)):
        x[i] = x[i] + [y[i]]

    x = sorted(x, key=itemgetter(attributeIndex))
    values = []
    gains = []
    for i in range(len(x) - 1):
        if x[i][-1] != x[i+1][-1]:
            newValue = (x[i][attributeIndex] + x[i+1][attributeIndex]) / 2
            values.append(newValue)
            groups = [[],[]]
            for j in x:
                if j[attributeIndex] <= newValue:
                    groups[0].append(j[-1])
                else:
                    groups[1].append(j[-1])
            gains.append(gain(y,groups))
    #print values
    #print gains
    #print values[np.argmax(gains)]
    return values[np.argmax(gains)]
    

"""if __name__ == '__main__':

    f = files["haberman"]
    
    x, y, attrList, possibleValuesList = preprocessing(f)

    #print possibleValuesList

    dt = DecisionTree(x,y, attrList, possibleValuesList, round(len(x[0])**0.5)) # **1 for test dataset, **0.5 for the other ones
    dt.training()

    print "\nDecision Tree:\n"
    dt.printTree()

    example = [0,1,2,1,1,1,2,2,1]
    print "\n", dt.predict(example)"""

