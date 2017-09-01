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
        plt.xlim(8.5, 13.5)
        plt.show()

        # data = np.load(path)
        # return data

    @staticmethod
    def plotResults(resfile):
        dataset = {}
        dataset["s7"] = {}
        dataset["s7"]["MRL00000"] = []
        dataset["s7"]["CormodeRandom00000"] =[]
        dataset["s7"]["KLL00000"] = []
        dataset["s7"]["KLL21111"] = []
        dataset["s7"]["KLL21110"] = []
        dataset["s7"]["KLL21100"] = []
        dataset["s7"]["KLL21000"] = []
        dataset["s7"]["KLL10000"] = []
        dataset["s7"]["KLL20000"] = []
        dataset["s7"]["KLL00000"] = []
        dataset["s7"]["KLL21111"] = []
        dataset["r7"] = {}
        dataset["r7"]["MRL00000"] = []
        dataset["r7"]["CormodeRandom00000"] =[]
        dataset["r7"]["KLL00000"] = []
        dataset["r7"]["KLL21111"] = []
        dataset["r7"]["KLL21110"] = []
        dataset["r7"]["KLL21100"] = []
        dataset["r7"]["KLL21000"] = []
        dataset["r7"]["KLL10000"] = []
        dataset["r7"]["KLL20000"] = []
        dataset["r7"]["KLL00000"] = []
        dataset["r7"]["KLL21111"] = []
        dataset["zi7"] = {}
        dataset["zi7"]["MRL00000"] = []
        dataset["zi7"]["CormodeRandom00000"] =[]
        dataset["zi7"]["KLL00000"] = []
        dataset["zi7"]["KLL21111"] = []
        dataset["zi7"]["KLL21110"] = []
        dataset["zi7"]["KLL21100"] = []
        dataset["zi7"]["KLL21000"] = []
        dataset["zi7"]["KLL10000"] = []
        dataset["zi7"]["KLL20000"] = []
        dataset["zi7"]["KLL00000"] = []
        dataset["zi7"]["KLL21111"] = []
        dataset["zo7"] = {}
        dataset["zo7"]["MRL00000"] = []
        dataset["zo7"]["CormodeRandom00000"] = []
        dataset["zo7"]["KLL00000"] = []
        dataset["zo7"]["KLL21111"] = []
        dataset["zo7"]["KLL21110"] = []
        dataset["zo7"]["KLL21100"] = []
        dataset["zo7"]["KLL21000"] = []
        dataset["zo7"]["KLL10000"] = []
        dataset["zo7"]["KLL20000"] = []
        dataset["zo7"]["KLL00000"] = []
        dataset["zo7"]["KLL21111"] = []

        for res in open(resfile,'r').readlines():
            resA = res.split(",")
            dataset[resA[0].strip()][resA[1].strip() + resA[3].strip()].append([np.log2(int(resA[2].strip())),float(resA[4].strip()),int(float(resA[6].strip()))])
        dataset["r7"]["MRL00000"] = np.array(dataset["r7"]["MRL00000"] )
        dataset["r7"]["CormodeRandom00000"] = np.array(dataset["r7"]["CormodeRandom00000"] )
        dataset["r7"]["KLL00000"] = np.array(dataset["r7"]["KLL00000"])
        dataset["r7"]["KLL20000"] = np.array(dataset["r7"]["KLL20000"] )
        dataset["r7"]["KLL10000"] = np.array(dataset["r7"]["KLL10000"] )
        dataset["r7"]["KLL21000"] = np.array(dataset["r7"]["KLL21000"] )
        dataset["r7"]["KLL21100"] = np.array(dataset["r7"]["KLL21100"] )
        dataset["r7"]["KLL21110"] = np.array(dataset["r7"]["KLL21110"] )
        dataset["r7"]["KLL21111"] = np.array(dataset["r7"]["KLL21111"] )
        # dataset["r7"]["KLL21111"] = np.array(dataset["r7"]["KLL21111"] )

        # dataset["r7"]["Quant2S"] = np.array(dataset["r7"]["Quant2S"])

        # dataset["r7"]["Quant2S"] = np.array(dataset["r7"]["Quant2S"])
        # dataset["r7"]["Quant5S"] = np.array(dataset["r7"]["Quant5S"])
        # dataset["r7"]["CormodeRandom"] = np.array(dataset["r7"]["CormodeRandom"])
        # dataset["s7"]["Quant2S"] = np.array(dataset["r7"]["Quant2S"])
        # dataset["s7"]["Quant5S"] = np.array(dataset["r7"]["Quant5S"])
        # dataset["s7"]["CormodeRandom"] = np.array(dataset["r7"]["CormodeRandom"])

        # plt.scatter(dataset["r7"]["Quant2S"][:, 0] -0.5 + dataset["r7"]["Quant2S"][:, 1], dataset["r7"]["Quant2S"][:, 2]/(10**6), alpha=0.5)
        plt.scatter(dataset["r7"]["MRL00000"][:, 0] -0.05 + 3*0.01, dataset["r7"]["MRL00000"][:, 2]/(10**6), alpha=0.5, label="MRL")
        plt.scatter(dataset["r7"]["CormodeRandom00000"][:, 0] -0.05 + 3*0.02, dataset["r7"]["CormodeRandom00000"][:, 2]/(10**6), alpha=0.5, label="RANDOM")
        plt.scatter(dataset["r7"]["KLL00000"][:, 0] -0.05 + 3*0.03, dataset["r7"]["KLL00000"][:, 2]/(10**6), alpha=0.5, label="KLL0 = vanilla")
        # plt.scatter(dataset["r7"]["KLL10000"][:, 0] -0.05 + 3*0.04, dataset["r7"]["KLL10000"][:, 2]/(10**6), alpha=0.5, label="KLL greedy")
        plt.scatter(dataset["r7"]["KLL20000"][:, 0] -0.05 + 3*0.05, dataset["r7"]["KLL20000"][:, 2]/(10**6), alpha=0.5, label="KLL1 = greedy")
        plt.scatter(dataset["r7"]["KLL21000"][:, 0] -0.05 + 3*0.06, dataset["r7"]["KLL21000"][:, 2]/(10**6), alpha=0.5, label="KLL2 = KLL1 + lazy")
        plt.scatter(dataset["r7"]["KLL21100"][:, 0] -0.05 + 3*0.07, dataset["r7"]["KLL21100"][:, 2]/(10**6), alpha=0.5, label="KLL3 = KLL2 + sampling")
        plt.scatter(dataset["r7"]["KLL21110"][:, 0] -0.05 + 3*0.08, dataset["r7"]["KLL21110"][:, 2]/(10**6), alpha=0.5, label="KLL4 = KLL3 + reduced randomness")
        plt.scatter(dataset["r7"]["KLL21111"][:, 0] - 0.05 + 3 * 0.09, dataset["r7"]["KLL21111"][:, 2] / (10 ** 6),label="KLL5 = KLL4 + spreading error",
                    alpha=0.5)

        # plt.scatter(dataset["r6"]["KLL21111"][:, 0] -0.05 + 3*0.1, dataset["r6"]["KLL21111"][:, 2]/(10**6), alpha=0.5)


        plt.yticks(fontsize=14)
        plt.legend(loc='upper right', fontsize=18)
        plt.xticks([ 9, 10, 11, 12, 13], ["$2^9$", "$2^{10}$", "$2^{11}$" , "$2^{12}$" , "$2^{13}$"],
                   fontsize=14)
        plt.xlabel('space', fontsize=18)
        plt.ylabel('$\\varepsilon$ (error)', fontsize=18)
        plt.xlim(8.5, 13.5)

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

    Plotting.plotResults("results7.csv")