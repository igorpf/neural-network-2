#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class DataSet():
    def __init__(self, file):
        self.dataMatrix = []
        self.generateDataMatrix(file.fileName)
        self.normalizeFeatures(file.normRanges)

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

if __name__ == '__main__':
    pass    
