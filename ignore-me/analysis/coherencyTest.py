import pandas as pd
import numpy as np
import scipy.fft
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from hs import LCS, getError
from coherencyHelper import *
import os

def cosine_similarity(ChA, ChB, freqChA, freqChB):
    return cos_helper(ChA, ChB), cos_helper(freqChA, freqChB)

def RMS_similarity(ChA, ChB, freqChA, freqChB):
    return RMS_helper(ChA, ChB), RMS_helper(freqChA, freqChB)

def peak_similarity(ChA, ChB, freqChA, freqChB):
    return peak_helper(ChA, ChB), peak_helper(freqChA, freqChB)

def SSD_similarity(ChA, ChB, freqChA, freqChB):
    return SSD_helper(ChA, ChB), SSD_helper(freqChA, freqChB)


def DTW_similarity(ChA, ChB, freqChA, freqChB):
    distance, path = fastdtw(np.array([ChA.values]), np.array([ChB.values]), dist=euclidean)
    return distance, None

def LCS_similarity(ChA, ChB, freqChA, freqChB):
    error = getError(freqChA, freqChB)
    time_LCS = len(LCS(ChA, ChB, error))/len(ChA)
    freq_LCS = len(LCS(freqChA, freqChB, error))/len(freqChA)
    return time_LCS, freq_LCS


def coherencyMethods(ChA, ChB):

    functionIndex = ['cos', 'RMS', 'peak', 'SSD', 'DTW', 'LCS']
    domainIndex = ['time', 'frequency']
    coherencyTable = pd.DataFrame(index=functionIndex, columns=domainIndex)

    freqChA = scipy.fft.fft(ChA.values)
    freqChB = scipy.fft.fft(ChB.values)

    #Make a function table for all functions being called
    functionTable = [cosine_similarity, RMS_similarity, peak_similarity, SSD_similarity, DTW_similarity, LCS_similarity]

    #Loop through matching indices and functions
    for i in range (0, len(functionTable)):
        function = functionTable[i]
        index = coherencyTable.index[i]

        timeVal, freqVal = function(ChA, ChB, freqChA, freqChB)

        coherencyTable.loc[index, 'time'] = timeVal
        coherencyTable.loc[index, 'frequency'] = freqVal

    return coherencyTable


def doAnalysis(data, pairList):
    for pair in pairList:
        print("COHERENCY MEASURES FOR CHANNELS " + str(pair[0]) + " AND " + str(pair[1]))

        indexA, indexB = pair[0]-1, pair[1]-1
        seriesA, seriesB = data[indexA], data[indexB]

        coherencyTable = coherencyMethods(seriesA, seriesB)

        print(coherencyTable)
        print()


def main():
    directory = r"../Nature Raw Txt"
    for file in os.listdir(directory):
        EEGdata = pd.read_csv(directory + "/" + file, header=None)
        EEGdata = EEGdata.drop(columns=[16], axis=1)
        analysisChannels = [(1, 9), (2, 10), (6, 14), (7, 15), (8, 16), (3, 11), (4, 12), (5, 13)]

        print("COHERENCY FOR " + str(file))
        print()
        doAnalysis(EEGdata, analysisChannels)



if __name__ == "__main__":
    main()
