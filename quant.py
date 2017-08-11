from __future__ import print_function
import sys
from random import random
from random import randint
from math import ceil, log, sqrt
import numpy as np

import logging

class QuantProto(object):
    def __init__(self, k, c=2.0 / 3.0):
        if k <= 0: raise ValueError("k must be a positive integer.")
        if c <= 0.5 or c > 1.0: raise ValueError("c must larger than 0.5 and at most 1.0.")
        self.k = k
        self.c = c
        self.compactors = []
        self.H = 0
        self.size = 0
        self.maxSize = 0
        self.grow()

    def grow(self):
        self.compactors.append(CompactorProto())
        self.H = len(self.compactors)
        self.maxSize = sum(self.capacity(height) for height in range(self.H))

    def capacity(self, hight):
        depth = self.H - hight - 1
        return int(ceil(self.c ** depth * self.k)) + 1

    def update(self, item):
        self.compactors[0].append(item)
        self.size += 1
        if self.size >= self.maxSize:
            self.compress()
            assert (self.size < self.maxSize)

    def compress(self):
        for h in range(len(self.compactors)):
            if len(self.compactors[h]) >= self.capacity(h):
                if h + 1 >= self.H: self.grow()
                self.compactors[h + 1].extend(self.compactors[h].compact())
                self.size = sum(len(c) for c in self.compactors)

    def rank(self, value):
        r = 0
        for (h, c) in enumerate(self.compactors):
            for item in c:
                if item <= value:
                    r += 2 ** h
        return r

    def cdf(self):
        itemsAndWeights = []
        for (h, items) in enumerate(self.compactors):
            itemsAndWeights.extend((item, 2 ** h) for item in items)
        totWeight = sum(weight for (item, weight) in itemsAndWeights)
        itemsAndWeights.sort()
        cumWeight = 0
        cdf = []
        for (item, weight) in itemsAndWeights:
            cumWeight += weight
            cdf.append((item, float(cumWeight) / float(totWeight)))
        return cdf

    def ranks(self):
        ranksList = []
        itemsAndWeights = []
        for (h, items) in enumerate(self.compactors):
            itemsAndWeights.extend((item, 2 ** h) for item in items)
        itemsAndWeights.sort()
        cumWeight = 0
        for (item, weight) in itemsAndWeights:
            cumWeight += weight
            ranksList.append((item, cumWeight))
        return ranksList


class CompactorProto(list):
    def compact(self):
        self.sort()
        if random() < 0.5:
            while len(self) >= 2:
                _ = self.pop()
                yield self.pop()
        else:
            while len(self) >= 2:
                yield self.pop()
                _ = self.pop()





################################################################
class Quant1(QuantProto):
    def __init__(self, k, c=2.0 / 3.0):
        super(Quant1, self).__init__(k, c)


################################################################
class Quant2(QuantProto):
    def __init__(self, k, c=2.0 / 3.0):
        super(Quant2, self).__init__(k, c)


    def compress(self):
        for h in range(len(self.compactors)):
            if len(self.compactors[h]) >= self.capacity(h):
                if h + 1 >= self.H: self.grow()
                self.compactors[h + 1].extend(self.compactors[h].compact())
                self.size = sum(len(c) for c in self.compactors)
                break

################################################################
class Quant3(QuantProto):
    def __init__(self, k, c=2.0 / 3.0):
        super(Quant3, self).__init__(k, c)

    def compress(self):
        for h in range(len(self.compactors)):
            if len(self.compactors[h]) >= self.capacity(h):
                if h + 1 >= self.H: self.grow()
                self.compactors[h + 1].\
                    extend(self.compactors[h].compact())
                self.size = sum(len(c) for c in self.compactors)
                break

    def grow(self):
        self.compactors.append(compactor3())
        self.H = len(self.compactors)
        self.maxSize = sum(self.capacity(height) \
                           for height in range(self.H))


class compactor3(list):
    def compact(self):
        self.sort()
        if len(self) >= 4:
            pairId = randint(0, len(self) - 2)
        else:
            pairId = 0
        if random() < 0.5:
            _ = self.pop(pairId)
            yield self.pop(pairId)
        else:
            yield self.pop(pairId)
            _ = self.pop(pairId)


################################################################
class  Quant4(QuantProto):
    def __init__(self, k, c = 2.0/3.0):
        super(Quant4, self).__init__(k, c)

    def compress(self):
        minError = 1000000
        minErrorLayer = -1
        for (h, items) in enumerate(self.compactors):
            if len(self.compactors[h]) >= self.capacity(h) and \
                                    float(2**h)/len(items) < minError \
                                    and h + 1  <= self.H :
                minError = float(2**h)/len(items)
                minErrorLayer = h

        if minErrorLayer + 1 >= self.H: self.grow()
        if minErrorLayer >=0:
            self.compactors[minErrorLayer + 1].\
                extend(self.compactors[minErrorLayer].compact())
            self.size = sum(len(c) for c in self.compactors)

    def grow(self):
        self.compactors.append(compactor4())
        self.H = len(self.compactors)
        self.maxSize = sum(self.capacity(height)\
                           for height in range(self.H))


class compactor4(list):
    def compact(self):
        self.sort()
        if len(self) >= 4:
            pairId = randint(0, len(self) - 2 )
        else:
            pairId = 0
        if random() < 0.5:
            _ = self.pop(pairId)
            yield self.pop(pairId)
        else:
            yield self.pop(pairId)
            _ = self.pop(pairId)


################################################################
class Quant5(QuantProto):
    def __init__(self, k, c=2.0 / 3.0):
        self.randomness = []
        super(Quant5, self).__init__(k, c)


    def grow(self):
        self.compactors.append(compactor5())
        if random() < 0.5:
            self.randomness.append(1)
        else:
            self.randomness.append(0)
        self.H = len(self.compactors)
        self.maxSize = sum(self.capacity(height) \
                           for height in range(self.H))

    def compress(self):
        for h in range(len(self.compactors)):
            if len(self.compactors[h]) >= self.capacity(h):
                if h + 1 >= self.H: self.grow()
                self.randomness[h] = (self.randomness[h] + 1) % 2
                self.compactors[h + 1].\
                    extend(self.compactors[h].compact(self.randomness[h]))
                self.size = sum(len(c) for c in self.compactors)
                break

class compactor5(list):
    def compact(self, directionFlag):
        self.sort()
        if directionFlag:
            while len(self) >= 2:
                _ = self.pop()
                yield self.pop()
        else:
            while len(self) >= 2:
                yield self.pop()
                _ = self.pop()

################################################################

class Quant6(QuantProto):
    def __init__(self, k, c=2.0 / 3.0):
        super(Quant6, self).__init__(k, c)
        self.maxSize = 3 * self.k
        logging.basicConfig(filename='log', level=logging.DEBUG)

    def grow(self):
        self.compactors.append(compactor())
        self.H = len(self.compactors)
        self.maxSize = max(sum(self.capacity(height) for height in range(self.H)), 3*self.k)

    def capacity(self, hight):
        depth = self.H - hight - 1
        return min(int(ceil(self.c ** depth * self.k)) + 1,
                   int(1.0 / self.c ** hight) + 1)

    def compress(self):
        for h in range(len(self.compactors)):
            if len(self.compactors[h]) >= self.capacity(h):
                if h + 1 >= self.H: self.grow()
                self.compactors[h + 1].\
                    extend(self.compactors[h].compact())
                # self.logCompress(h)
                self.size = sum(len(c) for c in self.compactors)
                # Here we break because we reduced the size by at least 1
                break
                # Removing this "break" will result in more eager
                # compression which has the same theoretical guarantees
                # but performs worse in practice

    def logCompress(self, level):
        logging.info('compaction at level:' + str(level) + ":\n")
        # logging.info('compactors at levels '  + str(level) + " and "  + str(level+1) + ":")
        logging.info('level ' + str(level) + ":" + ', '.join(str(e) for e in sorted(self.compactors[level])))
        logging.info('level ' + str(level + 1) + ":" + ', '.join(str(e) for e in self.compactors[level + 1]))

    def logUpdate(self, item):
        logging.info('update:' + str(item) + ":")
        # logging.info('compactors:')
        for h in range(len(self.compactors)):
            logging.info('level ' + str(h) + ":" + ', '.join(str(e) for e in sorted(self.compactors[h])))

    def update(self, item):
        # self.logUpdate(item)
        self.compactors[0].append(item)
        self.size += 1
        if self.size >= self.maxSize:
            self.compress()
            assert (self.size < self.maxSize)

class compactor(list):
    def compact(self):
        self.sort()
        if random() < 0.5:
            while len(self) >= 2:
                _ = self.pop()
                yield self.pop()
        else:
            while len(self) >= 2:
                yield self.pop()
                _ = self.pop()

####################################################################
class Quant7(QuantProto):
    def __init__(self, k, c=2.0 / 3.0):
        self.randomness = []
        super(Quant7, self).__init__(k, c)


    def grow(self):
        self.compactors.append(compactor7())
        if random() < 0.5:
            self.randomness.append(1)
        else:
            self.randomness.append(0)
        self.H = len(self.compactors)
        self.maxSize = sum(self.capacity(height) \
                           for height in range(self.H))

    def compress(self):
        for h in range(len(self.compactors)):
            if len(self.compactors[h]) >= self.capacity(h):
                if h + 1 >= self.H: self.grow()
                self.randomness[h] = (self.randomness[h] + 1) % 2
                self.compactors[h + 1].\
                    extend(self.compactors[h].compact(self.randomness[h],self.capacity(h)))
                self.size = sum(len(c) for c in self.compactors)
                break

class compactor7(list):
    def compact(self, directionFlag, capacity):
        self.sort()
        tmp = np.array(range(0,len(self)/2))
        np.random.shuffle(tmp)
        tmp =  sorted(tmp[:capacity])[::-1]
        if directionFlag:
            for i in tmp:
                _ = self.pop(2*i)
                yield self.pop(2*i)
        else:
            for i in tmp:
                yield self.pop(2*i)
                _ = self.pop(2*i)



######################################################################

class CormodeRandom:
    def __init__(self, eps):
        self.b = int(ceil((log(1./eps,2) + 1)/2.)*2)
        self.s = 1./eps*sqrt(log(1./eps,2))
        self.alBuckets = [Bucket() for _ in range(self.b)]
        self.alBucket_i = 0 # index to nonFull bucket in Active Layer
        self.al = 0   #active layer value
        self.sampler = np.array([-1,-1])


    def update(self,item):
        if (self.sampler[1] == -1):
            self.sampler[0] = randint(0, 2 ** self.al - 1)
            self.sampler[1] = 2 ** self.al - 1

        if (self.sampler[0] == 0):
            self.alBuckets[self.alBucket_i].append(item)
            if (len(self.alBuckets[self.alBucket_i]) == int(self.s)):
                self.alBucket_i += 1
                if self.alBucket_i > len(self.alBuckets)-1:
                    for i in range(0, self.b/2):
                        self.alBuckets[i] = Bucket(self.alBuckets[i],
                                                   self.alBuckets[i+ self.b/2])
                    for b in self.alBuckets[self.b/2:]:
                        del b[:]
                    self.alBucket_i = self.b/2
                    self.al += 1
        self.sampler -= 1

    def ranks(self):
        allItems = []
        for b in self.alBuckets:
            allItems.extend(b)
        allItems.sort()
        ranks = np.array(range(len(allItems)))*(2**self.al)
        return zip(allItems, ranks)




class Bucket(list):
    def __init__(self, b1=None, b2=None):
        super(Bucket, self).__init__()
        if b1 is not None:
            self.extend(sorted(b1 + b2)[random() < 0.5::2])

################################################################



if __name__ == "__main__":
    q = CormodeRandom(0.05)
    q = Quant7(32)
    a = np.array(range(1000))
    np.random.shuffle(a)
    print("here")
    for i in a:
        q.update(i)
        # i+i


    maxError = 0
    for i,j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)


    # a = Bucket()
    # b = Bucket()
    # a.extend([1, 2, 3])
    # b.extend([4, 5, 6])
    # c = Bucket(a, b)
    # print(a.append(5))


    # c.extend([7, 8, 9])

    # print (a.merge(b))


