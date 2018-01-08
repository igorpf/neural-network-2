#!/usr/bin/env python
# -*- coding: utf-8 -*-

# These two lines are necessary to find source files!!!
import sys
sys.path.append('../src')

from files import files
from main import DataSet
from ensemble import Ensemble, flatten
from bootstrapping import fold
import matplotlib.pyplot as plt
#to install matplotlib
#$ pip install matplotlib
#$ sudo apt-get install python-tk

def test(file):
    # print "\n---------------------------------------------------\n"
    print "Testing bootstrap for: ", file        
    f = files[file]
    ds = DataSet(f)
    folds = fold(ds.dataMatrix, f.classProportion, f.classIndex, 5)
    train = folds[:-1]
    test = folds[-1]
    ntrees = [2,3,5,10, 15, 20, 30, 40, 50]
    # ntrees = [2,3,5, 10]

    performance = {}
    means = []
    for ntree in ntrees:        
        ensemble = Ensemble(file, ntree)
        mean, stdDev = ensemble.crossValidation(f, ds, train)
        performance[ntree] = mean
        means.append(mean)
        # print "ntree {}, mean {}, stdDev {}".format(ntree, mean, stdDev)
    
    #"plt.show()" interrupts execution, so remember to close it for the tests to continue
    plt.plot(ntrees,means)
    plt.show()
        
    bestNTree = sorted(performance, key=performance.get, reverse=True)[0]
    #TODO: evaluate best ntree. Which trained random forest should we get? 
    # Do we train a new one?
    # print test
    ensemble.ntree = bestNTree
    trees = ensemble.createRandomForest(f,ds,flatten(train))
    correct, total = ensemble.evaluatePrediction(trees, test)
    print "\nBest ntree:{} Prediction: {} out of {}, accuracy {}".format(bestNTree,correct, len(test), correct/float(len(test)))
    print "\n---------------------------------------------------\n"

if __name__ == '__main__':
    datasets = ['haberman', 'wine', 'cmc']
    for dataset in datasets:    
        test(dataset)
