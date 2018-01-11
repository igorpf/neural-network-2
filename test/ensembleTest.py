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
    stdDevs = []
    for ntree in ntrees:        
        ensemble = Ensemble(file, ntree)
        mean, stdDev = ensemble.crossValidation(f, ds, train)
        performance[ntree] = mean
        means.append(mean)
        stdDevs.append(stdDev)
        # print "ntree {}, mean {}, stdDev {}".format(ntree, mean, stdDev)
    #print means
    #print stdDevs
    
    #"plt.show()" interrupts execution, so remember to close it for the tests to continue
    plt.plot(ntrees,means)
    plt.axis([0,50,0,1])
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

    # graphics
    """ntrees = [2,3,5,10, 15, 20, 30, 40, 50]
    m1 = [[0.5066763617133792, 0.5054204124801692, 0.6200423056583818, 0.31035166578529877, 0.3428080380750925, 0.34260973030142783, 0.2815970386039133, 0.2734003172924379, 0.2653358011634056],[0.7223691168693813, 0.6902432575356954, 0.7265335801163406, 0.726467477525119, 0.7347964040190376, 0.7023400317292438, 0.6862109994711793, 0.7265996827075621, 0.7022078265468007],[0.7385642517186674, 0.7102062400846114, 0.7306319407720783, 0.702009518773136, 0.7182046536224219, 0.7140401903754627, 0.6855499735589635, 0.7101401374933897, 0.6855499735589635],[0.4043495505023797, 0.6242067689053411, 0.35536753040719193, 0.32654680063458486, 0.33890798519301957, 0.2856292966684294, 0.28576150185087257, 0.273532522474881, 0.26943416181914326]]
    m2 = [[0.7625, 0.8384920634920635, 0.6926587301587301, 0.5033730158730159, 0.3984126984126984, 0.40575396825396826, 0.42718253968253966, 0.36468253968253966, 0.30079365079365084],[0.8392857142857143, 0.797420634920635, 0.727579365079365, 0.6575396825396825, 0.5529761904761905, 0.6297619047619047, 0.47559523809523807, 0.48908730158730157, 0.5033730158730159],[0.8388888888888888, 0.8527777777777779, 0.8382936507936508, 0.7827380952380953, 0.7410714285714286, 0.7057539682539683, 0.6990079365079365, 0.7202380952380952, 0.6224206349206349],[0.7625, 0.8384920634920635, 0.6926587301587301, 0.5033730158730159, 0.3984126984126984, 0.40575396825396826, 0.42718253968253966, 0.36468253968253966, 0.30079365079365084]]
    m3 = [[0.4206964141588839, 0.4139109881240632, 0.4020235212729159, 0.3723596218148276, 0.35622910181021566, 0.35539029170990427, 0.348599100657212, 0.3494523233022022, 0.34775164303009337],[0.4257926899573389, 0.3969099504208463, 0.3943646950305546, 0.38589300126830395, 0.37657096736999884, 0.3740285944886429, 0.3587743572005073, 0.3655511357085207, 0.3655482531995849],[0.41473826818863135, 0.38082555055920675, 0.3935431799838579, 0.3799780929320881, 0.372342326761213, 0.3731955494062032, 0.3757235097428802, 0.36215554018217455, 0.3731897843883316],[0.4224230370114147, 0.400311310965064, 0.4224230370114147, 0.37235673930589186, 0.37320131442407467, 0.35963334486336906, 0.35369249394673125, 0.3536838464199239, 0.348599100657212]]
    plt.plot(ntrees,m1[0],color = "k",label='m = d**0.5; metodo dos slides para atributos numericos'.format(1))
    plt.plot(ntrees,m1[1],color = "b",label='m = d**0.5; uso da media para atributos numericos'.format(1))
    plt.plot(ntrees,m1[3],color = "g",label='m = d; metodo dos slides para atributos numericos'.format(1))
    plt.plot(ntrees,m1[2],color = "r",label='m = d; uso da media para atributos numericos'.format(1))
    plt.legend(loc = 'best', fontsize = 'small')
    plt.axis([0,50,0,1])
    plt.show()

    plt.plot(ntrees,m2[0],color = "k",label='m = d**0.5; metodo dos slides para atrib. numericos'.format(1))
    plt.plot(ntrees,m2[1],color = "b",label='m = d**0.5; uso da media para atributos numericos'.format(1))
    plt.plot(ntrees,m2[3],color = "g",label='m = d; metodo dos slides para atributos numericos'.format(1))
    plt.plot(ntrees,m2[2],color = "r",label='m = d; uso da media para atributos numericos'.format(1))
    plt.legend(loc = 'best', fontsize = 'small')
    plt.axis([0,50,0,1])
    plt.show()

    plt.plot(ntrees,m3[0],color = "k",label='m = d**0.5; metodo dos slides para atributos numericos'.format(1))
    plt.plot(ntrees,m3[1],color = "b",label='m = d**0.5; uso da media para atributos numericos'.format(1))
    plt.plot(ntrees,m3[3],color = "g",label='m = d; metodo dos slides para atributos numericos'.format(1))
    plt.plot(ntrees,m3[2],color = "r",label='m = d; uso da media para atributos numericos'.format(1))
    plt.legend(loc = 'best', fontsize = 'small')
    plt.axis([0,50,0,1])
    plt.show()"""
