#!/usr/bin/env python
# -*- coding: utf-8 -*-

# These two lines are necessary to find source files!!!
import sys
sys.path.append('../src')

from bootstrapping import bootstrap
from files import files
from main import DataSet

def test(file):
    print "Testing bootstrap for: ", file
    f = files[file]
    ds = DataSet(f)
    bootstraps = bootstrap(ds.dataMatrix, useIndex=True)
    for i, bs in enumerate(bootstraps):
        training = set(bs[0])
        test = set(bs[1])
        print "Training set size: {}, test set size: {}, ratio: {}, whole dataset size: {}".format(len(training), len(test), len(test)/float(len(training)), len(ds.dataMatrix))
        print "Bootstrap intersection between training and set should be empty and is {}".format(training.intersection(test)) 
        # print "Bootstrap union between training and set should be the whole matrix is {}".format(training.union(test)) 
    print "\n---------------------------------------------------\n"
if __name__ == '__main__':
    test('haberman')
    test('wine')
    test('cmc')
