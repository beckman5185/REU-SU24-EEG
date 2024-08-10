import pandas as pd
import numpy as np
import scipy.fft
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt

def cos_helper(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def RMS_helper(x, y):
    series = [None] * len(x)

    for i in range(0, len(x)):
        num1 = x[i]
        num2 = y[i]
        if (num1 != 0 or num2 != 0):
            series[i] = (1 - (abs(num1 - num2) / (abs(num1) + abs(num2)))) ** 2
        else:
            # if x[i] and y[i] are 0, special case (assume 0 to avoid division by 0)
            series[i] = 1

    return np.sqrt(np.mean(series))


def peak_helper(x, y):
    series = []

    for i in range (0, len(x)):
        num1 = x[i]
        num2 = y[i]
        if (num1 != 0 or num2 != 0):
            series[i] = 1 - (abs(num1 - num2)/(2*max([abs(num1), abs(num2)])))
        else:
            # if x[i] and y[i] are 0, special case (assume 0 to avoid division by 0)
            series[i] = 1

    return np.mean(series)


def peak_at_peak_helper(x, y):

    series = []

    peak_indices_x, properties_x = scipy.signal.find_peaks(x)
    peak_indices_y, properties_y = scipy.signal.find_peaks(y)


    all_peak_indices = np.concatenate((peak_indices_x, peak_indices_y))
    all_peak_indices = np.unique(all_peak_indices)


    for i in range(0, len(all_peak_indices)):
        index = all_peak_indices[i]
        num1 = x[index]
        num2 = y[index]
        if (num1 != 0 or num2 != 0):
            series.append( 1 - (abs(num1 - num2)/(2*max([abs(num1), abs(num2)]))))
        else:
         #if x[i] and y[i] are 0, special case (assume 0 to avoid division by 0)
            series.append(1)
            print('exception!')


    return np.mean(series)


def SSD_helper(x, y):
    return sum((x-y)**2)


def getAlpha(frequencies, spectrum):
    alpha_series = np.empty(0)

    low, high = 8, 13
    index = 0
    while (frequencies[index] < high):
        if frequencies[index] > low:
            alpha_series = np.append(alpha_series, spectrum[index])

        index += 1

    return alpha_series

def coherence_helper(x, y, imaginary):
    # get coherency values for frequencies 0-500 Hz
    fs = 500
    numSegs = 4096
    f1, Pxx = scipy.signal.welch(x, fs=fs, nperseg=numSegs)
    f2, Pyy = scipy.signal.welch(y, fs=fs, nperseg=numSegs)
    f3, Pxy = scipy.signal.csd(x, y, fs=fs, nperseg=numSegs)


    #sample_frequencies, coherence_series = scipy.signal.coherence(x, y, fs=fs, nperseg=numSegs)

    # get coherency values from alpha frequency band (8Hz - 13Hz)
    #scale_factor = 500/(numSegs/2)
    #low_idx, high_idx = int(low * scale_factor), int(high * scale_factor)

    #getting just alpha
    #coherence_band = getAlpha(sample_frequencies, coherence_series)
    Pxx = getAlpha(f1, Pxx)
    Pyy = getAlpha(f2, Pyy)
    Pxy = getAlpha(f3, Pxy)


    if imaginary:
        #ICoh = np.abs(np.imag(Pxy)) ** 2 / (Pxx * Pyy)
        #Coh = np.imag(coherence_band)
        Coh = np.abs(np.imag(Pxy))/np.sqrt(Pxx * Pyy)
        #band_coherence_series = ICoh[low_idx:high_idx]
    else:
        #Coh = np.abs(Pxy)**2/(Pxx * Pyy)
        #Coh = coherence_band
        Coh = np.abs(Pxy)/np.sqrt(Pxx * Pyy)
        #band_coherence_series = Coh[low_idx:high_idx]


    # print(len(band_coherence_series))

    peak_indices, properties = scipy.signal.find_peaks(Coh)

    peak_vals = []
    for i in range(0, len(peak_indices)):
        index = peak_indices[i]
        peak_vals.append(Coh[index])

    # average coherence in frequency band is similarity measure
    #coherence = np.mean(Coh)
    coherence = np.mean(peak_vals)

    return coherence