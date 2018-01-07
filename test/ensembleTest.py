#!/usr/bin/env python
# -*- coding: utf-8 -*-

# These two lines are necessary to find source files!!!
import sys
sys.path.append('../src')

from files import files
# from main import DataSet, preprocessing, DecisionTree
from ensemble import Ensemble

def test(file):
    print "\n---------------------------------------------------\n"
    print "Testing bootstrap for: ", file    
    ntrees = [2,3,5,10, 15]
    performance = {}
    for ntree in ntrees:        
        ensemble = Ensemble(file, ntree)
        trees, accuracies = ensemble.createRandomForest()
        performance[ntree] = ensemble.getForestPerformance(accuracies)
    print sorted(performance, key=performance.get)
    print [(i, performance[i]) for i in sorted(performance, key=performance.get)]
    print "\n---------------------------------------------------\n"
if __name__ == '__main__':
    datasets = ['haberman', 'wine', 'cmc']
    for dataset in datasets:    
        test(dataset)
    # test('haberman', 10)
    # test('wine', 7)
    # test('cmc', 5)    
    # ensemble = Ensemble('cmc', 123)
    # print ensemble.splitDataset([1,2,3,4,5,6,7,8,9,10])
    pass
