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

    @staticmethod
    def plotResults(resfile):
        dataset = {}
        dataset["s6"] = {}
        dataset["s6"]["CormodeRandom"] =[]
        dataset["s6"]["Quant2S"] = []
        dataset["s6"]["Quant5S"] =[]
        dataset["s7"] = {}
        dataset["s7"]["CormodeRandom"] = []
        dataset["s7"]["Quant2S"] = []
        dataset["s7"]["Quant5S"] = []
        dataset["r6"] = {}
        dataset["r6"]["CormodeRandom"] = []
        dataset["r6"]["Quant2S"] = []
        dataset["r6"]["Quant5S"] = []
        dataset["r7"] = {}
        dataset["r7"]["CormodeRandom"] = []
        dataset["r7"]["Quant2S"] = []
        dataset["r7"]["Quant5S"] = []
        dataset["zi6"] = {}
        dataset["zi6"]["CormodeRandom"] = []
        dataset["zi6"]["Quant2S"] = []
        dataset["zi6"]["Quant5S"] = []
        dataset["zi7"] = {}
        dataset["zi7"]["CormodeRandom"] = []
        dataset["zi7"]["Quant2S"] = []
        dataset["zi7"]["Quant5S"] = []

        for res in open(resfile,'r').readlines():
            resA = res.split(",")
            dataset[resA[0].strip()][resA[1].strip()].append([np.log2(int(resA[2].strip())),float(resA[3].strip()),int(resA[5].strip())])
        dataset["r6"]["Quant2S"] = np.array(dataset["r6"]["Quant2S"])

        dataset["r7"]["Quant2S"] = np.array(dataset["r7"]["Quant2S"])
        dataset["r7"]["Quant5S"] = np.array(dataset["r7"]["Quant5S"])
        dataset["r7"]["CormodeRandom"] = np.array(dataset["r7"]["CormodeRandom"])
        dataset["s7"]["Quant2S"] = np.array(dataset["r7"]["Quant2S"])
        dataset["s7"]["Quant5S"] = np.array(dataset["r7"]["Quant5S"])
        dataset["s7"]["CormodeRandom"] = np.array(dataset["r7"]["CormodeRandom"])

        plt.scatter(dataset["r6"]["Quant2S"][:, 0] -0.5 + dataset["r6"]["Quant2S"][:, 1], dataset["r6"]["Quant2S"][:, 2]/(10**6), alpha=0.5)
        plt.yticks(fontsize=14)
        plt.legend(loc='upper right', fontsize=18)
        plt.xticks([5, 6, 7, 8, 9, 10, 11], ["$2^5$", "$2^6$", "$2^7$", "$2^8$", "$2^9$", "$2^{10}$", "$2^{11}$"],
                   fontsize=14)
        plt.xlabel('space', fontsize=18)
        plt.ylabel('$\\varepsilon$ (error)', fontsize=18)
        plt.show()

        # resCombined = []
            #     for i in range(res.shape[0]):
        #         for j in range(res.shape[1]):
        #             resCombined.append([(5 + i) + resfile_i * 0.05 - 0.15, res[i, j] / 1000000])
        #     resCombined = np.array(resCombined)
        #     plt.scatter(resCombined[:, 0], resCombined[:, 1], label=labels[resfile_i], alpha=0.5)
        # plt.yticks(fontsize=14)
        # plt.legend(loc='upper right', fontsize=18)
        # plt.xticks([5, 6, 7, 8, 9, 10, 11], ["$2^5$", "$2^6$", "$2^7$", "$2^8$", "$2^9$", "$2^{10}$", "$2^{11}$"],
        #            fontsize=14)
        # plt.xlabel('parameter $k$ (space/3)', fontsize=18)
        # plt.ylabel('$\\varepsilon$ (error)', fontsize=18)
        # plt.show()

if __name__ == '__main__':
    # Plotting.plotResults(["results/rand_q1.csv", "results/rand_q2.csv", "results/rand_q5.csv", "results/rand_q6.csv"])
    # Plotting.plotResults(["results/rand_q1.csv","results/rand_q1s.csv",
    #                       "results/rand_q2.csv", "results/rand_q2s.csv",
    #                       "results/rand_q5.csv", "results/rand_q5s.csv",
    #                       "results/rand_cr.csv"], ["KLL", "KLL + sampling",  "KLL + greedy memory", "KLL + greedy memory + samlping", "KLL + limitied randomness", "KLL + limitied randomness + sampling", "RANDOM"])
    #

    Plotting.plotResults("resTemp1.csv")