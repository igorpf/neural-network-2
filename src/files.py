#!/usr/bin/env python
# -*- coding: utf-8 -*-

class File(object):
    def __init__(self, fileName, normRanges, classes, classProportion):
        self.fileName = fileName
        self.normRanges = normRanges
        self.classes = classes
        self.classProportion = classProportion #class distribution for data stratification
        self.classIndex=normRanges.index(None)

files = {
    "haberman":File("../datasets/habermandata.txt",
        [[30.0, 83.0], [58.0, 69.0], [0.0, 52.0], None],
        [1.0, 2.0],
        [0.7353,0.2647]),
    "cmc": File("../datasets/cmcdata.txt",
        [[16.0, 49.0], [1.0, 4.0], [1.0, 4.0], [0.0, 16.0], [0.0, 1.0], [0.0, 1.0], [1.0, 4.0], [1.0, 4.0], [0.0, 1.0], None],
        [1.0, 2.0, 3.0],
        [0.427,0.2261,0.3469]),
    "wine":File("../datasets/winedata.txt",
        [[11.03, 14.83], [0.74, 5.8], [1.36, 3.23], [10.6, 30.0], [70.0, 162.0], [0.98, 3.88], [0.34, 5.08], [0.13, 0.66], [0.41, 3.58], [1.28, 13.0], [0.48, 1.71], [1.27, 4.0], [278.0, 1680.0], None],
        [1.0, 2.0, 3.0],
        [0.331,0.3988,0.2696])
}
    
        