import numpy as np
import random
import logging
from math import sqrt
import quant
from data import Data
from multiprocessing import Pool
from functools import partial

def doOneRun(data, sketchName, k):
    sketch = getattr(quant, sketchName)(k)
    for item in data:
        sketch.update(item)
    estRanks = sketch.ranks()
    nums, estRanks = zip(*estRanks)
    realRanks = Data.getQuantiles(data, nums)
    return np.max(np.abs(np.array(realRanks) - np.array(estRanks)))

def doManyRuns(data, sketchName, k, runsNum):
    res = np.zeros(runsNum)
    for i in     range(runsNum):
        res[i] = doOneRun(data, sketchName, k)
        print res[i]
    return res

def doManyRunsWithPool(data, sketchName, k, runsNum):
    # rep = lambda _: Experiment.doOneRun(data, sketchName, k)
    rep = partial(doOneRun, data, sketchName)
    pool = Pool(processes=2)
    print (pool.map(rep, [k]*runsNum))
    pool.close()
    pool.join()




if __name__ == '__main__':
    data = Data.load("./datasets/s_random.npy")
    # for k_i in range(5, 20):
    #     Experiment.doManyRuns(data, "Quant1", 2 ** k_i, 100)
    # print  doManyRuns(data, "Quant1", 128, 2)
    doManyRunsWithPool(data, "Quant1", 128, 2)

    # a = [np.zeros(2), "a", 3 , 34]
    # print a *3

    # rep = lambda _: Experiment.doOneRun(data, "Quant1", 120)
    # print rep(1)


        # # Data.gen2file("./datasets/", 10000000, "sqrt")
    # Data.gen2file("./datasets/", 10000000, "zoomin")
    # Data.gen2file("./datasets/", 10000000, "zoomout")
    # Data.gen2file("./datasets/", 10000000, "sorted")
    # Data.gen2file("./datasets/", 10000000, "random")
    #
    # # data = Data.load("./s_test.npy")
    # # for i in range(10):
    # #     print data[i]
    # # quants = Data.getQuantiles("./s_test.npy", [123,321])
    # # for i in range(2):
    # #     print quants[i]



