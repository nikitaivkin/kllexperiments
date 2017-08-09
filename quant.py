from __future__ import print_function
import sys
from random import random
from random import randint
from math import ceil

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
        super(Quant5, self).__init__(k, c)
        self.randomness = []

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
        self.maxSize = 3 * k

    def grow(self):
        self.compactors.append(compactor())
        self.H = len(self.compactors)
        # self.maxSize = sum(self.capacity(height) for height in range(self.H))

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
                self.size = sum(len(c) for c in self.compactors)
                # Here we break because we reduced the size by at least 1
                break
                # Removing this "break" will result in more eager
                # compression which has the same theoretical guarantees
                # but performs worse in practice


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

