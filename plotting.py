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
        dataset["s6"]["MRL00000"] = []
        dataset["s6"]["CormodeRandom00000"] =[]
        dataset["s6"]["KLL00000"] = []
        dataset["s6"]["KLL00001"] = []
        dataset["s6"]["KLL00010"] = []
        dataset["s6"]["KLL00100"] = []
        dataset["s6"]["KLL01000"] = []
        dataset["s6"]["KLL10000"] = []
        dataset["s6"]["KLL20000"] = []
        dataset["s6"]["KLL11111"] = []
        dataset["s6"]["KLL21111"] = []
        dataset["r6"] = {}
        dataset["r6"]["MRL00000"] = []
        dataset["r6"]["CormodeRandom00000"] =[]
        dataset["r6"]["KLL00000"] = []
        dataset["r6"]["KLL00001"] = []
        dataset["r6"]["KLL00010"] = []
        dataset["r6"]["KLL00100"] = []
        dataset["r6"]["KLL01000"] = []
        dataset["r6"]["KLL10000"] = []
        dataset["r6"]["KLL20000"] = []
        dataset["r6"]["KLL11111"] = []
        dataset["r6"]["KLL21111"] = []
        dataset["zi6"] = {}
        dataset["zi6"]["MRL00000"] = []
        dataset["zi6"]["CormodeRandom00000"] =[]
        dataset["zi6"]["KLL00000"] = []
        dataset["zi6"]["KLL00001"] = []
        dataset["zi6"]["KLL00010"] = []
        dataset["zi6"]["KLL00100"] = []
        dataset["zi6"]["KLL01000"] = []
        dataset["zi6"]["KLL10000"] = []
        dataset["zi6"]["KLL20000"] = []
        dataset["zi6"]["KLL11111"] = []
        dataset["zi6"]["KLL21111"] = []
        dataset["zo6"] = {}
        dataset["zo6"]["MRL00000"] = []
        dataset["zo6"]["CormodeRandom00000"] = []
        dataset["zo6"]["KLL00000"] = []
        dataset["zo6"]["KLL00001"] = []
        dataset["zo6"]["KLL00010"] = []
        dataset["zo6"]["KLL00100"] = []
        dataset["zo6"]["KLL01000"] = []
        dataset["zo6"]["KLL10000"] = []
        dataset["zo6"]["KLL20000"] = []
        dataset["zo6"]["KLL11111"] = []
        dataset["zo6"]["KLL21111"] = []

        for res in open(resfile,'r').readlines():
            resA = res.split(",")
            dataset[resA[0].strip()][resA[1].strip() + resA[3].strip()].append([np.log2(int(resA[2].strip())),float(resA[4].strip()),int(float(resA[6].strip()))])
        dataset["r6"]["MRL00000"] = np.array(dataset["r6"]["MRL00000"] )
        dataset["r6"]["CormodeRandom00000"] = np.array(dataset["r6"]["CormodeRandom00000"] )
        dataset["r6"]["KLL00001"] = np.array(dataset["r6"]["KLL00001"] )
        dataset["r6"]["KLL00010"] = np.array(dataset["r6"]["KLL00010"] )
        dataset["r6"]["KLL00100"] = np.array(dataset["r6"]["KLL00100"] )
        dataset["r6"]["KLL01000"] = np.array(dataset["r6"]["KLL01000"] )
        dataset["r6"]["KLL10000"] = np.array(dataset["r6"]["KLL10000"] )
        dataset["r6"]["KLL20000"] = np.array(dataset["r6"]["KLL20000"] )
        dataset["r6"]["KLL11111"] = np.array(dataset["r6"]["KLL11111"] )
        dataset["r6"]["KLL21111"] = np.array(dataset["r6"]["KLL21111"] ) 

        # dataset["r6"]["Quant2S"] = np.array(dataset["r6"]["Quant2S"])

        # dataset["r7"]["Quant2S"] = np.array(dataset["r7"]["Quant2S"])
        # dataset["r7"]["Quant5S"] = np.array(dataset["r7"]["Quant5S"])
        # dataset["r7"]["CormodeRandom"] = np.array(dataset["r7"]["CormodeRandom"])
        # dataset["s7"]["Quant2S"] = np.array(dataset["r7"]["Quant2S"])
        # dataset["s7"]["Quant5S"] = np.array(dataset["r7"]["Quant5S"])
        # dataset["s7"]["CormodeRandom"] = np.array(dataset["r7"]["CormodeRandom"])

        # plt.scatter(dataset["r6"]["Quant2S"][:, 0] -0.5 + dataset["r6"]["Quant2S"][:, 1], dataset["r6"]["Quant2S"][:, 2]/(10**6), alpha=0.5)
        plt.scatter(dataset["r6"]["MRL00000"][:, 0] -0.05 + 3*0.01, dataset["r6"]["MRL00000"][:, 2]/(10**6), alpha=0.5)
        plt.scatter(dataset["r6"]["CormodeRandom00000"][:, 0] -0.05 + 3*0.02, dataset["r6"]["CormodeRandom00000"][:, 2]/(10**6), alpha=0.5)
        plt.scatter(dataset["r6"]["KLL00001"][:, 0] -0.05 + 3*0.03, dataset["r6"]["KLL00001"][:, 2]/(10**6), alpha=0.5)
        plt.scatter(dataset["r6"]["KLL00010"][:, 0] -0.05 + 3*0.04, dataset["r6"]["KLL00010"][:, 2]/(10**6), alpha=0.5)
        plt.scatter(dataset["r6"]["KLL00100"][:, 0] -0.05 + 3*0.05, dataset["r6"]["KLL00100"][:, 2]/(10**6), alpha=0.5)
        plt.scatter(dataset["r6"]["KLL01000"][:, 0] -0.05 + 3*0.06, dataset["r6"]["KLL01000"][:, 2]/(10**6), alpha=0.5)
        plt.scatter(dataset["r6"]["KLL10000"][:, 0] -0.05 + 3*0.07, dataset["r6"]["KLL10000"][:, 2]/(10**6), alpha=0.5)
        plt.scatter(dataset["r6"]["KLL20000"][:, 0] -0.05 + 3*0.08, dataset["r6"]["KLL20000"][:, 2]/(10**6), alpha=0.5)
        plt.scatter(dataset["r6"]["KLL11111"][:, 0] -0.05 + 3*0.09, dataset["r6"]["KLL11111"][:, 2]/(10**6), alpha=0.5)
        plt.scatter(dataset["r6"]["KLL21111"][:, 0] -0.05 + 3*0.1, dataset["r6"]["KLL21111"][:, 2]/(10**6), alpha=0.5)


        plt.yticks(fontsize=14)
        plt.legend(loc='upper right', fontsize=18)
        plt.xticks([6, 7, 8, 9, 10, 11], ["$2^6$", "$2^7$", "$2^8$", "$2^9$", "$2^{10}$", "$2^{11}$"],
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

    Plotting.plotResults("resTemp2.csv")