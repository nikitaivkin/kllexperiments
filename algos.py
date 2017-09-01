from __future__ import print_function
import sys
from random import random
from random import randint
from math import ceil, log, sqrt, factorial
import numpy as np
import time
import cProfile
import logging

class KLL(object):
    def __init__(self, s=128, c=2.0 / 3.0, mode=(0,0,0,0,0), n=None):
        self.greedyMode = mode[0]
        self.lazyMode = mode[1]
        self.samplingMode = mode[2]
        self.oneTossMode = mode[3]
        self.varOptMode = mode[4]
        # print(mode)

        self.s = s
        self.n = n
        if not self.samplingMode:
            self.s -= 2*ceil(log(self.n,2))
        self.k = int(self.s*(1-c))  # always
        self.c = c             # always
        self.compactors = []   # always
        self.H = 0             # always

        self.size = 0               # if greedy
        self.maxSize = 0            # if greedy
        self.ranBit = []
        self.ranState = []
        self.grow()  # always

        self.D = 0                  # if sampling
        self.sampler = Sampler()    # if sampling

    def grow(self):

        if self.oneTossMode:
            self.ranBit.append(random() < 0.5)
            self.ranState.append(False)

        self.compactors.append(Compactor())

        if self.samplingMode and self.H + 1 > ceil(log(self.k, 2)):
            self.D += 1
            self.compactors.pop(0)
            if self.oneTossMode:
                self.ranBit.pop(0)
                self.ranState.pop(0)
        else:
            self.H += 1

        self.maxSize = sum(self.capacity(height) for height in range(self.H))

    def capacity(self, hight):
        depth = self.H - hight - 1
        return int(ceil(self.c ** depth * self.k)) + 1

    def update(self, item):
        if (self.samplingMode):
            item = self.sampler.sample(item, self.D)
        if (not self.samplingMode) or (item is not None):
            self.compactors[0].append(item)
            self.size += 1
            if (not self.greedyMode and (len(self.compactors[0]) >= self.capacity(0) or (self.size >= self.s))) or \
                                    (self.greedyMode == 1 and (self.size >= self.maxSize)) or \
                                    (self.greedyMode == 2 and (self.size >= self.s)):
                self.compress()
                if (self.samplingMode != 1):
                    # assert (self.size < self.maxSize)
                    assert self.size < self.s + 2 * ceil(log(self.n, 2)), "over1"
                if (self.samplingMode == 1):
                    assert self.size < self.s, "over2"


    def compress(self):
        for h in range(len(self.compactors)):
            if len(self.compactors[h]) >= self.capacity(h):
                if h + 1 >= self.H:
                    self.grow()
                    if self.samplingMode and h + 1 >= self.H:
                        h-=1
                assert h >= 0, "under1"
                if not self.oneTossMode:
                    self.compactors[h + 1].extend(self.compactors[h].compact(random() < 0.5, (random() < 0.5)*self.varOptMode ))
                else:
                    self.compactors[h + 1].extend(self.compactors[h].compact(self.ranBit[h], (random() < 0.5)*self.varOptMode ))
                    self.ranState[h] = not self.ranState[h]
                    self.ranBit[h] = self.ranState[h]* (not self.ranBit[h]) + (not self.ranState[h]) * (random() < 0.5)
                self.size = sum(len(c) for c in self.compactors)
                if self.lazyMode:
                    break


    def rank(self, value):
        r = 0
        for (h, c) in enumerate(self.compactors):
            for item in c:
                if item <= value:
                    r += 2 ** h
        return r


    def ranks(self):
        ranksList = []
        itemsAndWeights = []
        for (h, items) in enumerate(self.compactors):
            itemsAndWeights.extend((item, 2 ** (h+ self.samplingMode*self.D)) for item in items)
        itemsAndWeights.sort()
        cumWeight = 0
        for (item, weight) in itemsAndWeights:
            cumWeight += weight
            ranksList.append((item, cumWeight))
        return ranksList


class Compactor(list):
    def compact(self, bit1, bit2): # bit1 -for even or odd decision, bit2 for starting point (0 or 1)
        self.sort()
        if bit2 == 1 and len(self) > 2:   #varOptMode with the shift
            _in = [self.pop(0)]
            _in.append(self.pop()) if len(self) % 2 else []
        else:
            _in = [self.pop()] if len(self) % 2 else []
        _out = self[bit1::2]
        self[:] = _in
        return _out



class CormodeRandom:
    def __init__(self, s = None,c = None, mode= None,n= None):
        eps = self.space2eps(s)
        # print(str(eps) +  " " + str(s))
        self.b = int(ceil((log(1./eps,2) + 1)/2.)*2)
        self.s = 1./eps*sqrt(log(1./eps,2))
        self.alBuckets = [BucketC() for _ in range(self.b)]
        self.alBucket_i = 0 # index to nonFull bucket in Active Layer
        self.al = 0   #active layer value
        self.sampler = Sampler()


    def update(self,item):
        item = self.sampler.sample(item, self.al)
        if item is not None:
            self.alBuckets[self.alBucket_i].append(item)
            if len(self.alBuckets[self.alBucket_i]) == int(self.s):
                self.alBucket_i += 1
                if self.alBucket_i > len(self.alBuckets)-1:
                    for i in range(0, self.b/2):
                        self.alBuckets[i] = BucketC(self.alBuckets[i],
                                                    self.alBuckets[i+ self.b/2])
                    for b in self.alBuckets[self.b/2:]:
                        del b[:]
                    self.alBucket_i = self.b/2
                    self.al += 1

    def ranks(self):
        allItems = []
        for b in self.alBuckets:
            allItems.extend(b)
        allItems.sort()
        ranks = np.array(range(len(allItems)))*(2**self.al)
        return zip(allItems, ranks)

    def eps2space(self,eps):
        return sqrt(log(1.0 / eps, 2)) * (log(1.0 / eps, 2) + 1) / eps

    def space2eps(self,space):
        left = 0.000001
        right = 0.999999
        while self.eps2space(left) >= self.eps2space(right) + 1:
            midpoint = (left + right) / 2
            if space > self.eps2space(midpoint):
                right = midpoint
            else:
                left = midpoint
        return left


class BucketC(list):
    def __init__(self, b1=None, b2=None):
        super(BucketC, self).__init__()
        if b1 is not None:
            self.extend(sorted(b1 + b2)[random() < 0.5::2])


class Sampler():
    def __init__(self):
        self.s1 = self.s2 = -1

    def sample(self, item, l):
        if l == 0: return item
        if (self.s2 == -1):
            self.s1 = randint(0, 2 ** l - 1)
            self.s2 = 2 ** l - 1
        self.s1 -= 1
        self.s2 -= 1
        return item if (self.s1 == -1) else None

######################################################################

class MRL:
    def __init__(self, s = None,c = None, mode= None,n= None):
        self.k, self.b = 10,10 # to do the function calculating k and b
        self.findKB(s, n)
        self.alBuckets = []   #active layer buckets
        self.tlBuckets = []    # top layer buckets
        self.alBuckets.append(BucketM(self.k, 1))
        self.decisionBit = 0

    def update(self,item):
        if len(self.alBuckets[-1]) < self.k:
            self.alBuckets[-1].append(item)
        else:
            if len(self.alBuckets) + len(self.tlBuckets) == self.b:
                tmp = BucketM(self.k, None, self.alBuckets, decisionBit = self.decisionBit)
                self.decisionBit = (self.decisionBit+1)%2
                self.alBuckets = []
                self.tlBuckets.append(tmp)
                if len(self.tlBuckets) == self.b - 1:
                    self.alBuckets[:] = self.tlBuckets[:]
                    self.tlBuckets = []
            self.alBuckets.append(BucketM(self.k, 1))
            self.alBuckets[-1].append(item)


    def ranks(self):
        allItems = []
        for bucket in self.alBuckets:
            allItems.extend(zip(bucket[:], np.ones(self.k)*bucket.w))
        for bucket in self.tlBuckets:
            allItems.extend(zip(bucket[:], np.ones(self.k) * bucket.w))
        allItems.sort(key=lambda x: x[0])
        allItems = np.array(allItems)
        allItems[:,1] = np.cumsum(allItems[:,1])
        return allItems


    def findKB(self, s, n):
        minEps = 1
        minB = None
        for b in range(5,30):
            k = s/b
            cond = True
            maxH = 2
            h = 2
            while cond:
                h*=1.1
                cond = self.NchK(int(b+h-2),int(h-1)) < n/k
            h = int(h)
            eps = float((h-2)*self.NchK(b+h-3,h-1) - self.NchK(b+h-3,h-3) - self.NchK(b+h-3,h-2)) /n
            if eps > 0 and eps < minEps:
                minEps = eps
                minB = b
        self.b = minB
        self.k = s/minB
        # print(minEps)

    def NchK(self,n,k):
        return factorial(n)/ (factorial(k) * factorial(n-k))



class BucketM(list):
    def __init__(self, k, bWeight, buckets=None, decisionBit=None):
        super(BucketM, self).__init__()
        self.w = bWeight
        if buckets is not None:
            self.w = sum([b.w for b in buckets])

            allItems = []
            for bucket in buckets:
                allItems.extend(zip(bucket[:], np.ones(k) * bucket.w))
            allItems.sort(key=lambda x: x[0])
            allItems = np.array(allItems)
            allItems[:, 1] = np.cumsum(allItems[:, 1])
            if self.w % 2 == 1:
                idxs = np.arange(k)*self.w +(self.w+1)/2
            else:
                idxs = np.arange(k) * self.w + (self.w + 2*decisionBit) / 2
            self[:] = allItems[np.searchsorted(allItems[:,1], idxs),0]

def test():
    # # q = MRL(96, 10**7)
    # q = KLL(96,mode=(0,0,0,1,0), n =10**6)
    # # q = CormodeRandom(96, Noneu)
    # # # q = Quant5(96, 2./3)
    a = np.array(range(1000000))
    np.random.shuffle(a)
    # # print("here")
    # for item_i,item in enumerate(a):
    #     q.update(item)
    #     if item_i%10000 == 0:
    #         #print(item_i)
    # maxError = 0
    # for i,j in q.ranks():
    #     maxError = max(maxError, abs(i - j))
    # print(maxError)

    # q = CormodeRandom(s = 96, n = 10**6)
    # for item_i, item in enumerate(a):
    #     q.update(item)
    #     #if item_i % 10000 == 0:
    #         #print(item_i)
    # maxError = 0
    # for i, j in q.ranks():
    #     maxError = max(maxError, abs(i - j))
    # print(maxError)
    #
    # q = MRL(s = 96, n = 10**6)
    # for item_i, item in enumerate(a):
    #     q.update(item)
    #     #if item_i % 10000 == 0:
    #         #print(item_i)
    # maxError = 0
    # for i, j in q.ranks():
    #     maxError = max(maxError, abs(i - j))
    # print(maxError)

    q = KLL(s=64, mode=(2, 1, 1, 1, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        if item_i % 10000 == 0:
            print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(0, 0, 0, 0, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(0, 0, 0, 1, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(0, 0, 0, 1, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(0, 0, 1, 0, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(0, 0, 1, 0, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(0, 0, 1, 1, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(0, 0, 1, 1, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(0, 1, 0, 0, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(0, 1, 0, 0, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(0, 1, 0, 1, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(0, 1, 0, 1, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(0, 1, 1, 0, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(0, 1, 1, 0, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(0, 1, 1, 1, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(0, 1, 1, 1, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(1, 0, 0, 0, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(1, 0, 0, 0, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(1, 0, 0, 1, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(1, 0, 0, 1, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(1, 0, 1, 0, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(1, 0, 1, 0, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(1, 0, 1, 1, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(1, 0, 1, 1, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(1, 1, 0, 0, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(1, 1, 0, 0, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(1, 1, 0, 1, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(1, 1, 0, 1, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(1, 1, 1, 0, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(1, 1, 1, 0, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(1, 1, 1, 1, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(1, 1, 1, 1, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)

    q = KLL(s=96, mode=(2, 0, 0, 0, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(2, 0, 0, 0, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(2, 0, 0, 1, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(2, 0, 0, 1, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(2, 0, 1, 0, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(2, 0, 1, 0, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(2, 0, 1, 1, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(2, 0, 1, 1, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(2, 1, 0, 0, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(2, 1, 0, 0, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(2, 1, 0, 1, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(2, 1, 0, 1, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(2, 1, 1, 0, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(2, 1, 1, 0, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(2, 1, 1, 1, 0), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)
    q = KLL(s=96, mode=(2, 1, 1, 1, 1), n=10 ** 6)
    for item_i, item in enumerate(a):
        q.update(item)
        #if item_i % 10000 == 0:
            #print(item_i)
    maxError = 0
    for i, j in q.ranks():
        maxError = max(maxError, abs(i - j))
    print(maxError)


if __name__ == "__main__":
    test()
#
#     # cProfile.run("test()")
#     test()
