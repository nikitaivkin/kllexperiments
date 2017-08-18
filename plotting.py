import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import logging
from math import sqrt

class Plotting:
    @staticmethod
    def plotResults(resfiles, labels):
        for resfile_i, resfile in enumerate(resfiles):
            res = np.genfromtxt(resfile)
            resCombined = []
            for i in range(res.shape[0]):
                for j in range(res.shape[1]):
                    resCombined.append([(5 + i) + resfile_i*0.05 - 0.15, res[i,j]/1000000])
            resCombined = np.array(resCombined)
            plt.scatter(resCombined[:, 0], resCombined[:, 1], label = labels[resfile_i], alpha=0.5)
        plt.yticks( fontsize=14)
        plt.legend(loc='upper right', fontsize =18)
        plt.xticks([5,6,7,8,9,10,11], ["$2^5$","$2^6$","$2^7$","$2^8$","$2^9$","$2^{10}$","$2^{11}$"],  fontsize=14)
        plt.xlabel('parameter $k$ (space/3)',  fontsize=18)
        plt.ylabel('$\\varepsilon$ (error)', fontsize=18)
        plt.show()

        # data = np.load(path)
        # return data

if __name__ == '__main__':
    # Plotting.plotResults(["results/rand_q1.csv", "results/rand_q2.csv", "results/rand_q5.csv", "results/rand_q6.csv"])
    Plotting.plotResults(["results/rand_q1.csv","results/rand_q1s.csv",
                          "results/rand_q2.csv", "results/rand_q2s.csv",
                          "results/rand_q5.csv", "results/rand_q5s.csv",
                          "results/rand_cr.csv"], ["KLL", "KLL + sampling",  "KLL + greedy memory", "KLL + greedy memory + samlping", "KLL + limitied randomness", "KLL + limitied randomness + sampling", "RANDOM"])


