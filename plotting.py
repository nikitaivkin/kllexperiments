import numpy as np
import matplotlib.pyplot as plt
import random
import logging
from math import sqrt

class Plotting:
    @staticmethod
    def plotResults(resfiles):
        for resfile_i, resfile in enumerate(resfiles):
            res = np.genfromtxt(resfile)
            resCombined = []
            for i in range(res.shape[0]):
                for j in range(res.shape[1]):
                    resCombined.append([(5 + i) + resfile_i*0.1, res[i,j]])
            resCombined = np.array(resCombined)
            plt.scatter(resCombined[:, 0], resCombined[:, 1], alpha=0.5)
        plt.show()

        # data = np.load(path)
        # return data

if __name__ == '__main__':
    # Plotting.plotResults(["results/rand_q1.csv", "results/rand_q2.csv", "results/rand_q5.csv", "results/rand_q6.csv"])
    Plotting.plotResults(["results/tiny/rand_q5.csv", "results/tiny/rand_q7.csv"])


