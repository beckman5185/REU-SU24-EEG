import pandas as pd
import numpy as np
import scipy.fft
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
def RMS_similarity(x, y):

    sumSquares = 0

    for i in range(0, len(x)):
        if abs(x[i]) - abs(y[i]) == 0:
            print("Division by 0: " + str(x[i]) + "," + str(y[i]))
        numSim = 1 - (abs(x[i] - y[i]) / (abs(x[i]) - abs(y[i])))

        sumSquares += numSim ** 2

    return np.sqrt(sumSquares/len(x))

def peak_similarity(x, y):

    sumVal = 0
    for i in range (0, len(x)):
        sumVal += 1 - (abs(x[i] - y[i])/(2*max([abs(x[i]), abs(y[i])])))

    return sumVal/len(x)


def SSD_similarity(x, y):
    return sum((x-y)**2)


def LCS_similarity(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]

    """Following steps build L[m + 1][n + 1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

                # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]


def TS_Unfiltered():
    directory = r"Nature Raw Txt"
    EEGdata = pd.read_csv(directory + "/" + "Ball2_Nature_EEGData_fl10_N2.txt", header=None)



    Ch1 = EEGdata[0]
    Ch9 = EEGdata[8]

    freqCh1 = scipy.fft.fft(Ch1.values)
    freqCh9 = scipy.fft.fft(Ch9.values)


    #Cosine similarity measure
    cosVal = cosine_similarity(Ch1, Ch9)
    freqCosVal = cosine_similarity(freqCh1, freqCh9)
    print("Cos measure of similarity: " + str(cosVal) + " in time domain, " + str(freqCosVal) + " in frequency domain")

    #Root mean square similarity measure
    #RMSVal = RMS_similarity(Ch1, Ch9)
    #freqRMSVal = RMS_similarity(freqCh1, freqCh9)
    # print("RMS measure of similarity: " + str(RMSVal) + " in time domain, " + str(freqRMSVal) + " in frequency domain")


    #Peak similarity measure
    peakVal = peak_similarity(Ch1, Ch9)
    freqPeakVal = peak_similarity(freqCh1, freqCh9)
    print("Peak measure of similarity: " + str(peakVal) + " in time domain, " + str(freqPeakVal) + " in frequency domain")

    #Sum of squared differences similarity measure
    SSDVal = SSD_similarity(Ch1, Ch9)
    freqSSDVal = SSD_similarity(freqCh1, freqCh9)
    print("SSD measure of similarity: " + str(SSDVal) + " in time domain, " + str(freqSSDVal) + " in frequency domain")

    #Need to figure out dynamic time warping - casting to real?
    distance, path = fastdtw(np.array([Ch1.values]), np.array([Ch9.values]), dist=euclidean)
    freqDistance, freqPath = fastdtw(np.array([freqCh1]), np.array([freqCh9]), dist=euclidean)
    print("DTW measure of similarity: " + str(distance) + " in time domain, " + str(freqDistance) + " in frequency domain")

    #See other file for Hunt Syzmanski algorithm
    #LCSVal = LCS_similarity(Ch1, Ch9)
    #print("LCS measure of similarity: " + str(LCSVal))


if __name__ == "__main__":
    TS_Unfiltered()
