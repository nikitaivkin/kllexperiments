import numpy as np
import random
import logging
from math import sqrt

class Data:
    @staticmethod
    def load(path):
        data = np.load(path)
        return data

    @staticmethod
    def onTheFly(n, order=''):
        # random.seed(42)
        orders = ['sorted', 'zoomin', 'zoomout', 'sqrt', 'random', 'test']
        assert (order in orders)
        if order == 'sorted':  # sorted order
            for item in range(n):
                yield item
        elif order == 'zoomin':  # zoom1
            for item in range(int(n / 2)):
                yield item
                yield n - item
        elif order == 'zoomout':  # zoom1
            for item in range(1, int(n / 2)):
                yield n / 2 + item
                yield n / 2 - item
        elif order == 'sqrt':  # zoom1
            t = int(sqrt(2 * n))
            item = 0
            initialItem = 0
            initialSkip = 1
            for i in range(t):
                item = initialItem
                skip = initialSkip
                for j in range(t - i):
                    yield item
                    item += skip
                    skip += 1
                initialSkip += 1
                initialItem += initialSkip
        elif order == 'test':  # zoom1
            for item in range(n):
                yield item*3
        else:  # order == 'random':
            items = list(range(n))
            random.shuffle(items)
            for item in items:
                yield item

    #     print quants[i]

    @staticmethod
    def getQuantiles(data, nums):
        return np.searchsorted(np.sort(data), nums)






    # Data.gen2file("./datasets/tiny/", 1000000, "random")

    # data = Data.load("./s_test.npy")
    # for i in range(10):
    #     print data[i]
    # quants = Data.getQuantiles("./s_test.npy", [123,321])
    # for i in range(2):
    @staticmethod
    def gen2file(path, n, order):
        data = np.zeros(n)
        for item_i, item in enumerate(Data.onTheFly(n, order)):
            data[item_i] = item
        np.save(path, data)

if __name__ == '__main__':
    for i in range(6,8):
        print i
        # Data.gen2file("./datasets/", 10**i, "sqrt")
        Data.gen2file("./datasets/zi" + str(i), 10**i, "zoomin")
        print i
        Data.gen2file("./datasets/zo" + str(i), 10**i, "zoomout")
        print i
        Data.gen2file("./datasets/s" + str(i), 10**i, "sorted")
        print i
        Data.gen2file("./datasets/r" + str(i), 10**i, "random")

