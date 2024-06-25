import pandas as pd
import numpy as np
import scipy.fft
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from hs import LCS_similarity, getError


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
def RMS_similarity(x, y):

    sumSquares = 0

    for i in range(0, len(x)):
        #numSim = 1
        #if (abs(x[i]) - abs(y[i])) != 0:
        numSim = 1 - (abs(x[i] - y[i]) / (abs(x[i]) + abs(y[i])))

        sumSquares += numSim ** 2



    return np.sqrt(sumSquares/len(x))

def peak_similarity(x, y):

    sumVal = 0
    for i in range (0, len(x)):
        sumVal += 1 - (abs(x[i] - y[i])/(2*max([abs(x[i]), abs(y[i])])))

    return sumVal/len(x)


def SSD_similarity(x, y):
    return sum((x-y)**2)


def DTW_similarity(ChA, ChB):
    distance, path = fastdtw(np.array([ChA.values]), np.array([ChB.values]), dist=euclidean)
    return distance



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

        timeVal, freqVal = None, None

        if (index!='LCS'):
            #Get time and frequency coherency values for given function and channels
            timeVal = function(ChA, ChB)
            #DTW can't be calculated in frequency domain
            if (index != 'DTW'):
                freqVal = function(freqChA, freqChB)
        else:
            #LCS takes special third parameter
            timeVal = function(ChA, ChB, True)
            freqVal = function(freqChA, freqChB, False)


        coherencyTable.loc[index, 'time'] = timeVal
        coherencyTable.loc[index, 'frequency'] = freqVal

    return coherencyTable





def main():
    directory = r"../Nature Raw Txt"
    EEGdata = pd.read_csv(directory + "/" + "Ball2_Nature_EEGData_fl10_N2.txt", header=None)
    EEGdata = EEGdata.drop(columns=[16], axis=1)


    analysisChannels = [(1, 9), (2, 10), (6, 14), (7, 15), (8, 16), (3, 11), (4, 12), (5, 13)]



    for pair in analysisChannels:
        print("COHERENCY MEASURES FOR CHANNELS " + str(pair[0]) + " AND " + str(pair[1]))

        indexA, indexB = pair[0]-1, pair[1]-1
        ChA, ChB = EEGdata[indexA], EEGdata[indexB]

        coherencyTable = coherencyMethods(ChA, ChB)

        print(coherencyTable)
        print()



if __name__ == "__main__":
    main()
