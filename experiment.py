import numpy as np
from random import random

import logging
from math import sqrt, log
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

def doManyRunsWithPool(data, sketchName, k, runsNum, poolSize):
    # rep = lambda _: Experiment.doOneRun(data, sketchName, k)
    rep = partial(doOneRun, data, sketchName)
    pool = Pool(processes= poolSize)
    res =  pool.map(rep, [k]*runsNum)
    pool.close()
    pool.join()
    return res

def doManyRunsWithPoolForRangeOfKs(data, algoName, resfilePath, k_start, k_end, runsNum, poolSize ):
    print(algoName)
    resFile = open(resfilePath, "w")
    for k_i in range(k_start, k_end):
        print(k_i)
        res = doManyRunsWithPool(data, algoName, 2 ** k_i, runsNum, poolSize)
        resFile.write(" ".join(map(str, res)) + "\n")
    resFile.close()


if __name__ == '__main__':

    runsNum = 20
    poolSize = 20
    k_start = 5
    k_end = 12
    for i in range(6,9):
        data = Data.load("./datasets/s_random_" + str(10**i) + ".npy")
        doManyRunsWithPoolForRangeOfKs(data, "Quant1", "./results/rand_q1_" + str(10**i) + ".csv", k_start, k_end, runsNum, poolSize)
        doManyRunsWithPoolForRangeOfKs(data, "Quant1S", "./results/rand_q1s_" + str(10**i) + ".csv", k_start, k_end, runsNum, poolSize)
        doManyRunsWithPoolForRangeOfKs(data, "Quant2", "./results/rand_q2_" + str(10**i) + ".csv", k_start, k_end, runsNum, poolSize)
        doManyRunsWithPoolForRangeOfKs(data, "Quant2S", "./results/rand_q2s_" + str(10**i) + ".csv", k_start, k_end, runsNum, poolSize)
        doManyRunsWithPoolForRangeOfKs(data, "Quant5", "./results/rand_q5_" + str(10**i) + ".csv", k_start, k_end, runsNum, poolSize)
        doManyRunsWithPoolForRangeOfKs(data, "Quant5S", "./results/rand_q5s_" + str(10**i) + ".csv", k_start, k_end, runsNum, poolSize)
        doManyRunsWithPoolForRangeOfKs(data, "CormodeRandom", "./results/rand_cr_" + str(10**i) + ".csv", k_start, k_end, runsNum, poolSize)


    # resFile = open("./results/rand_q1.csv", "w")
    # for k_i in range(k_start, k_end):
    #     print(k_i)
    #     res = doManyRunsWithPool(data, "Quant1", 2**k_i, 100)
    #     resFile.write(" ".join(map(str, res))  +"\n")
    # resFile.close()
    #
    # resFile = open("./results/rand_q2.csv", "w")
    # print("Quant2")
    # for k_i in range(k_start, k_end):
    #     print (k_i)
    #     res = doManyRunsWithPool(data, "Quant2", 2 ** k_i, 100)
    #     resFile.write(" ".join(map(str, res)) + "\n")
    # resFile.close()
    # #
    # resFile = open("./results/rand_q3.csv", "w")
    # for k_i in range(k_start, k_end):
    #     print(k_i)
    #     res = doManyRunsWithPool(data, "Quant3", 2 ** k_i, 100)
    #     resFile.write(" ".join(map(str, res)) + "\n")
    # resFile.close()
    #
    # print("Quant6")
    # resFile = open("./results/rand_q4.csv", "w")
    # for k_i in range(k_start, k_end):
    #     print(k_i)
    #     res = doManyRunsWithPool(data, "Quant4", 2 ** k_i, 100)
    #     resFile.write(" ".join(map(str, res)) + "\n")
    # resFile.close()
    #
    # print("Quant6")
    # resFile = open("./results/rand_q5.csv", "w")
    # for k_i in range(k_start, k_end):
    #     print(k_i)
    #     res = doManyRunsWithPool(data, "Quant5", 2 ** k_i, 100)
    #     resFile.write(" ".join(map(str, res)) + "\n")
    # resFile.close()
    #
    # print("Quant6")
    # resFile = open("./results/rand_q6.csv", "w")
    # for k_i in range(k_start, k_end):
    #     print(k_i)
    #     res = doManyRunsWithPool(data, "Quant6", 2 ** k_i, 100)
    #     resFile.write(" ".join(map(str, res)) + "\n")
    # resFile.close()
    #
    # resFile = open("./results/rand_cr.csv", "w")
    # for k_i in range(k_start, k_end):
    #     print(k_i)
    #     res = doManyRunsWithPool(data, "CormodeRandom", 2 ** k_i, 100)
    #     resFile.write(" ".join(map(str, res)) + "\n")
    # resFile.close()

    # # data = Data.load("./datasets/tiny/s_random.npy")
    # # resFile = open("./results/tiny/rand_q1.csv", "w")
    # # for k_i in range(5, 20):
    # #     res = doManyRunsWithPool(data, "Quant1", 2**k_i, 100)
    # #     resFile.write(" ".join(map(str, res))  +"\n")
    # # resFile.close()
    # #
    # # resFile = open("./results/tiny/rand_q2.csv", "w")
    # #
    # # print("Quant2")
    # # for k_i in range(5, 20):
    # #     print (k_i)
    # #     res = doManyRunsWithPool(data, "Quant2", 2 ** k_i, 100)
    # #     resFile.write(" ".join(map(str, res)) + "\n")
    # # resFile.close()
    # #
    # # print("Quant3")#
    # # resFile = open("./results/tiny/rand_q3.csv", "w")
    # # for k_i in range(5, 20):
    # #     res = doManyRunsWithPool(data, "Quant3", 2 ** k_i, 100)
    # #     resFile.write(" ".join(map(str, res)) + "\n")
    # # resFile.close()
    # #
    # # print("Quant4")
    # # resFile = open("./results/tiny/rand_q4.csv", "w")
    # # for k_i in range(5, 20):
    # #     res = doManyRunsWithPool(data, "Quant4", 2 ** k_i, 100)
    # #     resFile.write(" ".join(map(str, res)) + "\n")
    # # resFile.close()
    # #
    # print("Quant5")
    # resFile = open("./results/tiny/rand_q5.csv", "w")
    # for k_i in range(5, 12):
    #     print(k_i)
    #     res = doManyRunsWithPool(data, "Quant5", 2 ** k_i, 100)
    #     resFile.write(" ".join(map(str, res)) + "\n")
    # resFile.close()
    #
    # # print("Quant6")
    # # resFile = open("./results/tiny/rand_q6.csv", "w")
    # # for k_i in range(5, 20):
    # #     print(k_i)
    # #     res = doManyRunsWithPool(data, "Quant6", 2 ** k_i, 100)
    # #     resFile.write(" ".join(map(str, res)) + "\n")
    # # resFile.close()
    #
    # print("Quant7")
    # resFile = open("./results/tiny/rand_q7.csv", "w")
    # for k_i in range(5, 12):
    #     print(k_i)
    #     res = doManyRunsWithPool(data, "Quant7", 2 ** k_i, 100)
    #     resFile.write(" ".join(map(str, res)) + "\n")
    # resFile.close()
    #
    #
    # # resFile = open("./results/tiny/rand_cr.csv", "w")
    # # for k_i in range(5, 12):
    # #     res = doManyRunsWithPool(data, "CormodeRandom", 2 ** k_i, 100)
    # #     resFile.write(" ".join(map(str, res)) + "\n")
    # # resFile.close()
    #
    #
    #
    #
    # #     Experiment.doManyRuns(data, "Quant1", 2 ** k_i, 100)
    # # print  doManyRuns(data, "Quant1", 128, 2)
    #
    #
    # # a = [np.zeros(2), "a", 3 , 34]
    # # print a *3
    #
    # # rep = lambda _: Experiment.doOneRun(data, "Quant1", 120)
    # # print rep(1)
    #
    #
    # # # Data.gen2file("./datasets/", 10000000, "sqrt")
    # # Data.gen2file("./datasets/", 10000000, "zoomin")
    # # Data.gen2file("./datasets/", 10000000, "zoomout")
    # # Data.gen2file("./datasets/", 10000000, "sorted")
    # # Data.gen2file("./datasets/", 10000000, "random")
    # #
    # # # data = Data.load("./s_test.npy")
    # # # for i in range(10):
    # # #     print data[i]
    # # # quants = Data.getQuantiles("./s_test.npy", [123,321])
    # # # for i in range(2):
    # # #     print quants[i]
    #
    #
    #
