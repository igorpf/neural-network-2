#!/usr/bin/env python
# -*- coding: utf-8 -*-

# These two lines are necessary to find source files!!!
import sys
sys.path.append('../src')

from files import files
from main import DataSet, preprocessing, DecisionTree

def test(file, examples):
    print "Testing bootstrap for: ", file
    f = files[file]
    x, y, attrList, possibleValuesList = preprocessing(f)
    print possibleValuesList

    dt = DecisionTree(x,y, attrList, possibleValuesList, int(len(x[0])**0.5)) # **1 for test dataset, **0.5 for the other ones
    dt.training()

    print "\nDecision Tree:\n"
    dt.printTree()

    for example in examples:
    	print "\n", dt.predict(example)
    
    print "\n---------------------------------------------------\n"
if __name__ == '__main__':
    # test('haberman')
    # test('wine')
    test('cmc', [[0,1,2,1,1,1,2,2,1]])
