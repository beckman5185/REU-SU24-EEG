import pandas as pd
import numpy as np
import scipy.fft
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from hs import LCS, getError
from coherencyHelper import *
import os
import matplotlib.pyplot as plt

def cosine_similarity(ChA, ChB, timeDomain):
    return cos_helper(ChA, ChB)

def RMS_similarity(ChA, ChB, timeDomain):
    return RMS_helper(ChA, ChB)

def peak_similarity(ChA, ChB, timeDomain):
    return peak_helper(ChA, ChB)

def SSD_similarity(ChA, ChB, timeDomain):
    return SSD_helper(ChA, ChB)


def DTW_similarity(ChA, ChB, timeDomain):
    #Can only do dynamic time warping on time domain data
    distance = None
    if timeDomain:
        distance, path = fastdtw(np.array([ChA]), np.array([ChB]), dist=euclidean)

    return distance

def LCS_similarity(ChA, ChB, timeDomain):
    if timeDomain:
        freqChA = scipy.fft.fft(ChA)
        freqChB = scipy.fft.fft(ChB)
        error = abs(getError(ChA, ChB))
    else:
        error = abs(getError(ChA, ChB))
    return len(LCS(ChA, ChB, error))/len(ChA)



def doAnalysis(data, function, timeDomain, filtered):

    analysisChannels = [(1, 9), (2, 10), (6, 14), (7, 15), (8, 16), (3, 11), (4, 12), (5, 13)]
    channelIndex = ['Fp1-Fp2', 'F3-F4', 'F7-F8', 'T3-T4', 'T5-T6', 'C3-C4', 'P3-P4', 'O1-O2']

    coherencySeries = pd.Series(index=channelIndex)

    for i in range(0, len(analysisChannels)):

        channelName = channelIndex[i]
        indexA, indexB = analysisChannels[i][0] - 1, analysisChannels[i][1] - 1
        seriesA, seriesB = data[indexA].values, data[indexB].values

        #need to apply filter before converting to frequency domain
        #savgol implementation casts all values to real, losing imaginary parts of frequency
        if(filtered):
            seriesA = scipy.signal.savgol_filter(seriesA, 15, 5)
            seriesB = scipy.signal.savgol_filter(seriesB, 15, 5)
            '''
            numtaps1 = 200
            cutoff1 = [1, 40]
            samplingFreq = 500
            hammingCoeffs = scipy.signal.firwin(numtaps1, cutoff1, window='hamming', pass_zero='bandpass',
                                                fs=samplingFreq)

            denoms = [1] * len(hammingCoeffs)


            seriesA = scipy.signal.lfilter(hammingCoeffs, denoms, seriesA)
            seriesB = scipy.signal.lfilter(hammingCoeffs, denoms, seriesB)

            seriesA = np.abs(scipy.fft.fft(seriesA)) ** 2
            seriesB = np.abs(scipy.fft.fft(seriesB)) ** 2
            '''



        #for frequency domain, get the power spectrum of the time series
        if (not timeDomain):
            seriesA = np.abs(scipy.fft.fft(seriesA)) ** 2
            seriesB = np.abs(scipy.fft.fft(seriesB)) ** 2

            #getting just alpha band of power spectrum
            low, high = 8, 13
            scale = 30000 / 500
            low_idx, high_idx = int(low * scale), int(high * scale)
            seriesA = seriesA[low_idx:high_idx]
            seriesB = seriesB[low_idx:high_idx]


        coherencySeries[channelName] = function(seriesA, seriesB, timeDomain)


    return coherencySeries


def main(sound, function, timeDomain, filtered):
    channelIndex = ['Fp1-Fp2', 'F3-F4', 'F7-F8', 'T3-T4', 'T5-T6', 'C3-C4', 'P3-P4', 'O1-O2']

    timeAnalysisTable = pd.DataFrame(columns=channelIndex)

    directory = r"../Nature Raw Txt"
    for file in os.listdir(directory):
        soundName = file.split('_')[-1].strip('.txt')
        lastName = file.split('_')[0]

        if (soundName == sound):
            EEGdata = pd.read_csv(directory + "/" + file, header=None)
            EEGdata = EEGdata.drop(columns=[16], axis=1)

            coherencySeries = doAnalysis(EEGdata, function, timeDomain, filtered)
            timeAnalysisTable.loc[lastName] = coherencySeries



    print("Sound: " + sound)
    print("Method of analysis: " + function.__name__)

    directory = r"excel-output//" + function.__name__
    filename = directory + "//" + sound + "_" + function.__name__ + "_"
    if timeDomain:
        filename += "time_"
    else: filename += "frequency_"
    if filtered:
        filename += "filtered.xlsx"
    else:
        filename += "unfiltered.xlsx"
    timeAnalysisTable.to_excel(filename)

def comp(sound, function, timeDomain):

    channelIndex = ['Fp1-Fp2', 'F3-F4', 'F7-F8', 'T3-T4', 'T5-T6', 'C3-C4', 'P3-P4', 'O1-O2']

    timeAnalysisTableFiltered = pd.DataFrame(columns=channelIndex)
    timeAnalysisTableUnfiltered = pd.DataFrame(columns=channelIndex)

    directory = r"../Nature Raw Txt"
    for file in os.listdir(directory):
        soundName = file.split('_')[-1].strip('.txt')
        lastName = file.split('_')[0]

        if (soundName == sound):
            EEGdata = pd.read_csv(directory + "/" + file, header=None)
            EEGdata = EEGdata.drop(columns=[16], axis=1)

            coherencySeriesFiltered = doAnalysis(EEGdata, function, timeDomain, True)
            coherencySeriesUnfiltered = doAnalysis(EEGdata, function, timeDomain, False)
            timeAnalysisTableFiltered.loc[lastName] = coherencySeriesFiltered
            timeAnalysisTableUnfiltered.loc[lastName] = coherencySeriesUnfiltered

    print("Difference in averages: ")
    for channelPair in channelIndex:
        filtered = timeAnalysisTableFiltered[channelPair]
        unfiltered = timeAnalysisTableUnfiltered[channelPair]
        difference = np.mean(filtered) - np.mean(unfiltered)
        print(channelPair + ": " + str(difference))


if __name__ == "__main__":
    #main('N2', LCS_similarity, True)
    #print()
    #main('N2', cos_helper, False)

    #main('N2', cosine_similarity, True, False)
    comp('N4', RMS_similarity, True)
