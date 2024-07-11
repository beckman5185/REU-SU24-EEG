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
        #get fourier transform
        #freqChA = scipy.fft.fft(ChA)
        #freqChB = scipy.fft.fft(ChB)

        #get power spectrum
        #freqChA = np.abs(scipy.fft.fft(ChA)) ** 2
        #freqChB = np.abs(scipy.fft.fft(ChB)) ** 2

        # getting just alpha band
        #low, high = 8, 13
        #scale = 30000 / 500
        #low_idx, high_idx = int(low * scale), int(high * scale)
        #freqChA = freqChA[low_idx:high_idx]
        #freqChB = freqChB[low_idx:high_idx]

        #error = abs(getError(freqChA, freqChB))

        #error is 1/10th mean of the two series
        error = abs(getError(ChA, ChB))
    else:
        #error is 1/10th mean of power spectrum of alpha band
        error = abs(getError(ChA, ChB))

    #get length of longest common subsequence relative to vector length
    length = len(LCS(ChA, ChB, error))
    relativeLength = length/len(ChA)

    return relativeLength

def filter(seriesA, seriesB, filtered):
    if (filtered):
        # apply Savitzky-Golay filter to data
        seriesA = scipy.signal.savgol_filter(seriesA, 15, 5)
        seriesB = scipy.signal.savgol_filter(seriesB, 15, 5)

        # apply bandpass filter to data
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

    return seriesA, seriesB


def frequency(seriesA, seriesB, timeDomain, alpha):
    if (not timeDomain):
        # getting power spectrum
        seriesA = np.abs(scipy.fft.fft(seriesA)) ** 2
        seriesB = np.abs(scipy.fft.fft(seriesB)) ** 2

        # getting fourier transform
        # seriesA = scipy.fft.fft(seriesA)
        # seriesB = scipy.fft.fft(seriesB)

        # getting just alpha band
        if alpha:
            low, high = 8, 13
        else:
        # getting just gamma band
            low, high = 35, 44
        scale = 30000 / 500
        low_idx, high_idx = int(low * scale), int(high * scale)
        seriesA = seriesA[low_idx:high_idx]
        seriesB = seriesB[low_idx:high_idx]

    return seriesA, seriesB




def doAnalysis(data, function, timeDomain, alpha, filtered):

    #channels under analysis
    analysisChannels = [(1, 9), (2, 10), (6, 14), (7, 15), (8, 16), (3, 11), (4, 12), (5, 13)]
    channelIndex = ['Fp1-Fp2', 'F3-F4', 'F7-F8', 'T3-T4', 'T5-T6', 'C3-C4', 'P3-P4', 'O1-O2']

    #storing one coherency value for each channel pair
    coherencySeries = pd.Series(index=channelIndex)


    #loop over all channel pairs
    for i in range(0, len(analysisChannels)):

        #get channels to analyze from current pair
        channelName = channelIndex[i]
        indexA, indexB = analysisChannels[i][0] - 1, analysisChannels[i][1] - 1
        seriesA, seriesB = data[indexA].values, data[indexB].values

        #need to apply filter before converting to frequency domain
        #savgol implementation casts all values to real, losing imaginary parts of frequency
        seriesA, seriesB = filter(seriesA, seriesB, filtered)

        #for frequency domain, get the power spectrum of the time series
        seriesA, seriesB = frequency(seriesA, seriesB, timeDomain, alpha)

        #get coherency value for current channel pair
        coherencySeries[channelName] = function(seriesA, seriesB, timeDomain)


    return coherencySeries

def getGender(lastName):
    gender = None
    maleList = ['Barb', 'Ford', 'Helmick', 'Stewart', 'White', 'Carr', 'Washington', 'Harris', 'Marin', 'Stevens',
                'Carey', 'MoneyPenny', 'DeVaul', 'Lough']
    femaleList = ['Farley', 'Paushel', 'Forbes', 'Harper', 'Barr', 'Ball', 'Robin', 'Ball2', 'Prickett', 'Pryor',
                  'Collins', 'Shingleton', 'Jackson', 'Hartman', 'Beach', 'Uphold', 'Bucklew', 'Chenoweth',
                  'Winsten', 'Queen', 'Loss', 'Nestor', 'Bradley', 'Koveski', 'Coen', 'Massangale']
    if lastName in maleList:
        gender = 'M'
    elif lastName in femaleList:
        gender = 'F'

    return gender



def generateTable(timeDomain, alpha, filtered):
    #list of channels under analysis and methods of analysis
    channelIndex = ['Fp1-Fp2', 'F3-F4', 'F7-F8', 'T3-T4', 'T5-T6', 'C3-C4', 'P3-P4', 'O1-O2']
    methodList = [cosine_similarity, RMS_similarity, peak_similarity, SSD_similarity, DTW_similarity, LCS_similarity]

    #variables stored in output table
    tableIndex = ['Subject', 'Gender', 'Sound', 'Coherency']

    #pair table list has dataframes for output for each channel pair for each method of analysis
    pairTableList = [None] * len(methodList)
    for i in range(len(pairTableList)):
        pairTableList[i] = [None] * len(channelIndex)
        for j in range (0, len(channelIndex)):
            pairTableList[i][j] = pd.DataFrame(columns=tableIndex)
    #pairTableList = pd.DataFrame(index=methodList, columns=channelIndex)
    #for i in range(0, len(methodList)):
    #    for j in range (0, len(channelIndex)):
    #        pairTableList.iloc[i, j] = pd.DataFrame(columns=tableIndex)


    #looking into folder with data
    directory = r"../Nature Raw Txt"
    for file in os.listdir(directory):
        #get sound and last name and gender of participant
        soundName = file.split('_')[-1].strip('.txt')
        lastName = file.split('_')[0]
        gender = getGender(lastName)


        #read in EEG data
        EEGdata = pd.read_csv(directory + "/" + file, header=None)
        EEGdata = EEGdata.drop(columns=[16], axis=1)

        #do analysis for each method
        for i in range(len(methodList)):
            coherencySeries = doAnalysis(EEGdata, methodList[i], timeDomain, alpha, filtered)

            #for each channel pair, insert coherency results into pair list table
            for j in range (0, len(channelIndex)):
                row = pd.Series([lastName, gender, soundName, coherencySeries.iloc[j]], index=tableIndex)
                pairTableList[i][j] = pairTableList[i][j]._append(row, ignore_index=True)


                #row = pd.DataFrame([[lastName, gender, soundName, coherencySeries.iloc[j]]], columns=tableIndex)
                #row = {tableIndex[0]:lastName, tableIndex[1]:gender, tableIndex[2]:soundName, tableIndex[3]:coherencySeries.iloc[j]}
                #print(pairTableList.to_string())
                #pairTableList.loc[len(pairTableList.index)] = [lastName, gender, soundName, coherencySeries.iloc[j]]
                #pairTableList.iloc[i,j] = pd.concat([pairTableList.iloc[i, j], row])



    #for each table in pair table list, print output to file in relevant directory
    for i in range (len(methodList)):
        for j in range (len(channelIndex)):
            filename = "//" + channelIndex[j] + ".csv"
            if timeDomain:
                directory2 = r"time-"
            elif alpha:
                directory2 = r"alpha-"
            else:
                directory2 = r"gamma-"

            if filtered:
                directory2 += r"filtered-output//" + methodList[i].__name__
            else:
                directory2 += r"unfiltered-output//" + methodList[i].__name__

            if not os.path.exists(directory2):
                os.makedirs(directory2)

            pairTableList[i][j].to_csv(directory2 + filename)


def generateAll():
    timeList = [True, False]
    filterList = [True, False]

    #timeDomain: true is time, false is frequency
    #alpha: true is alpha band, false is gamma band (only applicable if frequency)
    #filtered: true is filtered, false is unfiltered
    generateTable(False, False, False)

    #for time in timeList:
        #for filter in filterList:
            #generateTable(time, filter)


if __name__ == "__main__":
    generateAll()
