#!/usr/bin/env python
# -*- coding: utf-8 -*-

# These two lines are necessary to find source files!!!
import sys
sys.path.append('../src')

from files import files
from main import DataSet, preprocessing, DecisionTree

if __name__ == '__main__':
    f = files["test"]
    
    x, y, attrList, possibleValuesList = preprocessing(f)

    dt = DecisionTree(x,y, attrList, possibleValuesList, int(len(x[0])**1)) # **1 for test dataset, **0.5 for the other ones
    dt.training()

    print "\nDecision Tree:\n"
    dt.printTree()

