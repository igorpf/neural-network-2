#!/usr/bin/env python
# -*- coding: utf-8 -*-

# These two lines are necessary to find source files!!!
import sys
sys.path.append('../src')

from files import files
from main import DataSet
from ensemble import Ensemble
from bootstrapping import fold

def test(file):
    print "\n---------------------------------------------------\n"
    print "Testing bootstrap for: ", file        
    f = files[file]
    ds = DataSet(f)
    folds = fold(ds.dataMatrix, f.classProportion, f.classIndex, 5)
    train = folds[:-1]
    test = folds[-1]
    # ntrees = [2,3,5,10, 15, 20, 30]
    ntrees = [2,3,5]

    performance = {}
    for ntree in ntrees:        
        ensemble = Ensemble(file, ntree)
        mean, stdDev = ensemble.crossValidation(f, ds, train)
        performance[ntree] = mean
        print "ntree {}, mean {}, stdDev {}".format(ntree, mean, stdDev)
        
    bestNTree = sorted(performance, key=performance.get, reverse=True)
    #TODO: evaluate best ntree. Which trained random forest should we get? 
    # Do we train a new one?

    print "\n---------------------------------------------------\n"

if __name__ == '__main__':
    datasets = ['haberman', 'wine', 'cmc']
    for dataset in datasets:    
        test(dataset)
