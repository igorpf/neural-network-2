#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from files import files
from random import randint
import math
import sys

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
        self.selectedAttribute = -1

    def training(self):
        originalAttrList = []
        for i in self.attrList:
            originalAttrList.append(i)
        #while len(self.attrList) > self.m:
        #    del self.attrList[randint(0,len(self.attrList) - 1)]
        gains = []
        for attribute in self.attrList:
            groups = []
            for value in possibleValuesList[attribute]:
                group = []
                for i in range(len(self.x)):
                    if(self.x[i][attribute] == value):
                        group.append(self.y[i])
                groups += [group]
            gains += [(attribute, gain(self.y, groups))]
        print gains
        bestGain = -1000
        bestAttribute = 0
        for g in gains:        
            if g[1] > bestGain:
                bestGain = g[1]
                bestAttribute = g[0]

        self.selectedAttribute = bestAttribute

        attrList_children = []
        for i in originalAttrList:
            if i != bestAttribute:
                attrList_children.append(i)

        for value in possibleValuesList[bestAttribute]:
            x_child = []
            y_child = []
            for i in range(len(self.x)):
                if(self.x[i][bestAttribute] == value):
                    x_child.append(self.x[i])
                    y_child.append(self.y[i])
            self.children += [DecisionTree(x_child, y_child, attrList_children, self.possibleValuesList, self.m)]

       
        for child in self.children:
            if len(child.attrList) > 0 and len(set(child.y)) > 1:
                child.training()
            else:
                child.isPure = True
                if not child.y:
                    child.predictedClass = max(set(self.y), key=self.y.count)
                elif len(set(child.y)) == 1:
                    child.predictedClass = child.y[0]
                else: #attrList is empty
                    child.predictedClass = max(set(child.y), key=child.y.count)

    def predict(self, example):
        if self.isPure:
            return self.predictedClass
        else:
            for child in range(len(self.children)):
                if example[self.selectedAttribute] == self.possibleValuesList[self.selectedAttribute][child]:
                    return self.children[child].predict(example)

    def printTree(self, level = 0):
        if self.isPure:
            print "class =", int(self.predictedClass),
        else:
            print "Attribute for division: " + str(self.selectedAttribute),
            for child in range(len(self.children)):
                print "\n",
                for i in range(level+1):
                    print "    ",
                self.children[child].printTree(level+1)  

def preprocessing(ds):
    x = [[ds.dataMatrix[i % len(ds.dataMatrix)][j] for j in range(len(ds.dataMatrix[0]) - 1)] for i in range(len(ds.dataMatrix))]
    y = [ds.dataMatrix[i % len(ds.dataMatrix)][-1] for i in range(len(ds.dataMatrix))]
    attrList = []
    possibleValuesList = []
    for i in range(len(x[0])):
        attrList.append(i)
        possibleValues = []
        for instance in x:
            if not (instance[i] in possibleValues):
                possibleValues.append(instance[i])
        possibleValuesList += [possibleValues]
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

if __name__ == '__main__':

    f = files["test"]
    ds = DataSet(f)
    
    x, y, attrList, possibleValuesList = preprocessing(ds)
    
    dt = DecisionTree(x,y, attrList, possibleValuesList, int(len(x[0])**0.5))
    dt.training()

    print "\nDecision Tree:\n"
    dt.printTree()
    
    #example = [2,1,0,1]
    #print dt.predict(example)
