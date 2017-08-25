import numpy as np
from random import random

import logging
from math import sqrt, log
import quant
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
        cParam = float(setting[3])
        sketch = getattr(quant, sketchName)(space, cParam)
        rep = int(setting[4])
        for item in data:
            sketch.update(item)
        estRanks = sketch.ranks()
        nums, estRanks = zip(*estRanks)
        realRanks = Data.getQuantiles(data, nums)
        settingNew = setting[:]
        settingNew.append(str(np.max(np.abs(np.array(realRanks) - np.array(estRanks)))))
        print ", ".join(settingNew)

def installPy2(sshList):
    outList = sshParRequest(sshList, "export DEBIAN_FRONTEND=noninteractive;  sudo apt-get -y install python-minimal; ")
    sshWaitToFinish(outList) # printOutList(outList)
    outList = sshParRequest(sshList, "export DEBIAN_FRONTEND=noninteractive; sudo apt-get -y install python-numpy python-scipy python-matplotlib python-paramiko")
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
    print "installing python2"
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

def genQueue(datasets, algos, srange, crange,repsNum, path):
    f = open(path, "w")
    for dataset in datasets:
        for algo in algos:
            for space in srange:
                for c in crange:
                    for rep in range(repsNum):
                        f.write(" ".join([dataset,algo,str(space), str(c), str(rep)]) + "\n")
                if algo == 'CormodeRandom': break;


def runAllExp():
    print "reading hosts file ..."
    hostfile = '''/home/local/ANT/ivkin/pssh/nodes'''
    print "Done"

    print "connecting to all the nodes"
    sshList = sshParConnect(hostfile)
    print "Done"

    prepCodeAndData(sshList)

    queue = readSettingQueue("queue.csv")
    batchSize = 10
    queueSize = len(queue)
    batchBegin = 0
    nodesN = len(sshList)
    resFile = open("results.csv", "w")
    while batchBegin < queueSize - batchSize:
        print str(batchBegin) + " out of " + str(queueSize)
        outList  = []
        for ssh in sshList:
            # outList.append(sshRequest(ssh, '''echo ''' +str(batchBegin) +  ''',''' + str(batchBegin + batchSize) + ''';'''))
            outList.append(sshRequest(ssh, '''cd kllexperiments; python2    -c "import exp; exp.runExp(''' +str(batchBegin) +  ''',''' + str(batchBegin + batchSize) + ''')"'''))
            batchBegin += batchSize
        printOutList2File(outList, resFile)
        break

    resFile.close()


def readSettingQueue(path):
    params = []
    for line in tuple(open(path, 'r')):
        params.append(line.rstrip().split())
    return params


if __name__ == '__main__':

    datasets = ["./datasets/s6.npy", "./datasets/s7.npy",
                "./datasets/r6.npy", "./datasets/r7.npy",
                "./datasets/zi6.npy", "./datasets/zi7.npy",
                "./datasets/zo6.npy", "./datasets/zo7.npy"]
    algos = ["Quant5S" , "CormodeRandom"]
    srange = 2**np.array(range(5,9))
    crange = np.arange(0.51, 0.99, 0.05)
    repsNum = 20
    path = "./queue.csv"
    genQueue(datasets, algos, srange, crange, repsNum, path)

    runAllExp()
