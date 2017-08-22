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
            setting[0] = dataPath
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
        print ", ".join(setting.append(np.max(np.abs(np.array(realRanks) - np.array(estRanks)))))

def installPy2(sshList):
    outList = sshParRequest(sshList, "export DEBIAN_FRONTEND=noninteractive;  sudo apt-get -y install python-minimal; ")
    sshWaitToFinish(outList)

def cloneRepo(sshList):
    outList = sshParRequest(sshList, '''git clone https://github.com/nikitaivkin/kllexperiments''')
    sshWaitToFinish(outList)

def poolGit(sshList):
    outList = sshParRequest(sshList, '''cd kllexperiments''')
    sshWaitToFinish(outList)
    outList = sshParRequest(sshList, '''git pull''')
    sshWaitToFinish(outList)

def copyData(sshList, folder):
    sshParRequest(sshList, "mkdir /home/ubuntu/kllexperiments/datasets")
    sshParSendFolder(sshList, folder, "/home/ubuntu/kllexperiments/datasets")

def genData(sshList):
    outList = sshParRequest(sshList, "mkdir /home/ubuntu/kllexperiments/datasets; cd /home/ubuntu/kllexperiments")
    sshWaitToFinish(outList)
    outList = sshParRequest(sshList, '''python2 data.py''')
    sshWaitToFinish(outList)


def prepCodeAndData(sshList):
    print "installing python2"
    installPy2(sshList)
    print "Done"

    print "cloning repo on remote nodes"
    cloneRepo(sshList)
    print "Done"

    print "pulling fresh code from git"
    poolGit(sshList)
    print "Done"

    print "generating datasets on remote nodes"
    genData(sshList)
    print "Done"

    # print "sending datasets to remote nodes"
    # copyData(sshList, "/home/local/ANT/ivkin/projects/amazon/quantiles/kllexperiments/datasets")
    # print "Done"



def runAllExp():
    print "reading hosts file ..."
    hostfile = '''/home/local/ANT/ivkin/pssh/nodes'''
    print "Done"

    print "connecting to all the nodes"
    sshList = sshParConnect(hostfile)
    print "Done"

    prepCodeAndData(sshList)

    # queue = readSettingQueue("queue.csv")
    # batchSize = 10
    # queueSize = len(queue)
    # batchBegin = 0
    # nodesN = len(sshList)
    # resFile = open("results", "w")
    # while batchBegin < queueSize - batchSize:
    #     print str(batchBegin) + " out of " + str(queueSize)
    #     for ssh in sshList:
    #         outList.append(sshRequest(ssh, '''echo ''' +str(batchBegin) +  ''',''' + str(batchBegin + batchSize) + ''')"'''))
    #         # outList.append(sshRequest(ssh, '''python27 -c "import exp; exp.runExp(''' +str(batchBegin) +  ''',''' + str(batchBegin + batchSize) + ''')"'''))
    #         batchBegin += batchSize
    #     printOutList2File(outList, resFile)
    # resFile.close()


def readSettingQueue(path):
    params = []
    for line in tuple(open(path, 'r')):
        params.append(line.rstrip().split())
    return params


if __name__ == '__main__':
    runAllExp()
