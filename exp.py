import numpy as np
from random import random

import logging
from math import sqrt, log
# import quant
from algos import *
from data import Data
from awsTools import *
from multiprocessing import Pool
from functools import partial

def runExp(start, end):
    queue = readSettingQueue("queue.csv")
    queue = queue[start:end]
    dataPath = ''
    data = None
    for setting_i, setting in enumerate(queue):
        if dataPath != setting[0]:
            dataPath = setting[0]
            data = Data.load(dataPath)
        sketchName = setting[1]
        space = int(setting[2])
        dataN = 10**int(setting[0][-5])
        algoMode = map(int, list(setting[3]))
        cParam = float(setting[4])
        sketch = getattr(algos, sketchName)(s=space,c=cParam, mode=algoMode, n=dataN)
        rep = int(setting[5])
        for item in data:
            sketch.update(item)
        estRanks = sketch.ranks()
        nums, estRanks = zip(*estRanks)
        realRanks = Data.getQuantiles(data, nums)
        settingNew = setting[:]
        settingNew.append(str(np.max(np.abs(np.array(realRanks) - np.array(estRanks)))))
        print ", ".join(settingNew)

def doOneRun(setting):
    dataPath = setting[0]
    data = Data.load(dataPath)
    sketchName = setting[1]
    space = int(setting[2])
    dataN = 10 ** int(setting[0][-5])
    algoMode = map(int, list(setting[3]))
    cParam = float(setting[4])
    sketch = KLL(s=space, c=cParam, mode=algoMode, n=dataN)
    rep = int(setting[5])
    for item_i, item in enumerate(data):
        sketch.update(item)
        # if item_i%10000==0:
        #     print item_i
    estRanks = sketch.ranks()
    nums, estRanks = zip(*estRanks)
    realRanks = Data.getQuantiles(data, nums)
    settingNew = setting[:]
    settingNew.append(str(np.max(np.abs(np.array(realRanks) - np.array(estRanks)))))
    return ", ".join(settingNew)


def runExpWithPool(start, end):
    queue = readSettingQueue("queue.csv")
    queue = queue[start:end]
    pool = Pool(processes=60)
    results = pool.map(doOneRun, queue)
    # print doOneRun(queue[0])
    pool.close()
    pool.join()
    for result in results:
        print result



def installPy2(sshList):
    outList = sshParRequest(sshList, "export DEBIAN_FRONTEND=noninteractive;  sudo apt-get -y install python-minimal; ")
    sshWaitToFinish(outList) # printOutList(outList)
    outList = sshParRequest(sshList, "export DEBIAN_FRONTEND=noninteractive; sudo apt-get -y install python-numpy python-paramiko;")
    sshWaitToFinish(outList) #printOutList(outList)


def cloneRepo(sshList):
    outList = sshParRequest(sshList, '''sudo rm -r  /home/ubuntu/kllexperiments;''')
    sshWaitToFinish(outList) # printOutList(outList)
    outList = sshParRequest(sshList, '''git clone https://github.com/nikitaivkin/kllexperiments''')
    sshWaitToFinish(outList) # printOutList(outList)

def poolGit(sshList):
    outList = sshParRequest(sshList, '''cd  /home/ubuntu/kllexperiments; git pull''')
    sshWaitToFinish(outList) # printOutList(outList)
    sshParRequest(sshList, "mkdir -p /home/ubuntu/kllexperiments/datasets")
    sshWaitToFinish(outList) # printOutList(outList)


def copyData(sshList, folder):
    sshParSendFolder(sshList, folder, "/home/ubuntu/kllexperiments/datasets")

def copyQueue(sshList):
    sshParSendFile(sshList, "queue.csv", "/home/ubuntu/kllexperiments/queue.csv")

def genData(sshList):
    outList = sshParRequest(sshList, "mkdir -p /home/ubuntu/kllexperiments/datasets")
    sshWaitToFinish(outList) # printOutList(outList)
    outList = sshParRequest(sshList, '''cd /home/ubuntu/kllexperiments; python2 data.py''')
    sshWaitToFinish(outList) # printOutList(outList)


def prepCodeAndData(sshList):
    print "installing python2 and packages"
    installPy2(sshList)

    print "cloning repo on remote nodes"
    cloneRepo(sshList)

    print "pulling fresh code from git"
    poolGit(sshList)

    print "generating datasets on remote nodes"
    genData(sshList)
    
    # print "sending datasets to remote nodes"
    # copyData(sshList, "/home/local/ANT/ivkin/projects/amazon/quantiles/kllexperiments/datasets")
    # print "Done"

    print "sending queue to remote nodes"
    copyQueue(sshList)
    print "Done"

def genQueue(datasets, algos, srange, modes, crange,repsNum, path):
    f = open(path, "w")
    c = 2./3.
    for dataset in datasets:
        for algo in algos:
            for space in srange:
                for mode in modes:
                    # for c in crange:
                    for rep in range(repsNum):
                        f.write(" ".join([dataset,algo,str(space),mode, str(c), str(rep)]) + "\n")
                        # if algo == 'CormodeRandom' or algo == 'MRL': break;
                    if algo == 'CormodeRandom' or algo == 'MRL': break;



def runAllExp():
    print "reading hosts file ..."
    hostfile = '''/home/local/ANT/ivkin/pssh/nodes'''
    print "Done"

    print "connecting to all the nodes"
    sshList = sshParConnect(hostfile)
    print "Done"

    prepCodeAndData(sshList)

    queue = readSettingQueue("queue.csv")
    batchSize = 128
    queueSize = len(queue)
    batchBegin = 0
    nodesN = len(sshList)
    resFile = open("results.csv", "w")
    while batchBegin < queueSize - batchSize:
        print str(batchBegin) + " out of " + str(queueSize)
        outList  = []
        for ssh in sshList:
            # outList.append(sshRequest(ssh, '''echo ''' +str(batchBegin) +  ''',''' + str(batchBegin + batchSize) + ''';'''))
            outList.append(sshRequest(ssh, '''cd kllexperiments; python2    -c "import exp; exp.runExpWithPool(''' +str(batchBegin) +  ''',''' + str(batchBegin + batchSize) + ''')"'''))
            batchBegin += batchSize
        printOutList2File(outList, resFile)

    resFile.close()


def readSettingQueue(path):
    params = []
    for line in tuple(open(path, 'r')):
        params.append(line.rstrip().split())
    return params


if __name__ == '__main__':

    
    datasets = ["./datasets/r6.npy", "./datasets/s6.npy", "./datasets/zi6.npy", "./datasets/zo6.npy"]
    # datasets = ["./datasets/r6.npy", "./datasets/r7.npy",
    #             "./datasets/s6.npy", "./datasets/s7.npy",
    #             "./datasets/zi6.npy", "./datasets/zi7.npy",
    #             "./datasets/zo6.npy", "./datasets/zo7.npy"]
    algos = ['KLL']
    srange = 2**np.array(range(6,11))
    # modes = ["00000","00001","00010","00011","00100","00101","00110","00111","01000","01001","01010","01011","01100","01101","01110","01111","10000","10001","10010","10011","10100","10101","10110","10111","11000","11001","11010","11011","11100","11101","11110","11111","20000","20001","20010","20011","20100","20101","20110","20111","21000","21001","21010","21011","21100","21101","21110","21111"]
    modes = ["00000","10000","20000","21000","21100","21110","21111"]
    # modes = ["00000"]#,"00001","00010","00100","01000","10000","11111","20000","21111"]
    # crange = np.arange(0.51,              0.99, 0.05)
    crange = np.arange(0.1, 0.5, 0.05)
    repsNum = 20
    path = "./queue.csv"
    genQueue(datasets, algos, srange, modes, crange, repsNum, path)
    # runExpWithPool(1,10)
    runAllExp()
    # runExpWithPool
    # runExp(0, 10)
