#!/usr/bin/env python
# -*- coding: utf-8 -*-

# These two lines are necessary to find source files!!!
import sys
sys.path.append('../src')

from files import files
# from main import DataSet, preprocessing, DecisionTree
from ensemble import Ensemble

def test(file, ntree):
    print "\n---------------------------------------------------\n"
    print "Testing bootstrap for: ", file
    ensemble = Ensemble(file, ntree)
    ensemble.createEnsemble()
    print "\n---------------------------------------------------\n"
if __name__ == '__main__':
    # test('haberman')
    # test('wine', 7)
    test('cmc', 5)
    # l1 = [1,2,3,4]
    # l2 = map(lambda x: x, l1)
    # l1.append(1)
    # print l1, l2
    # ensemble = Ensemble('cmc', 123)
    # print ensemble.splitDataset([1,2,3,4,5,6,7,8,9,10])
    pass
