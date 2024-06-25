import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft


def readIn():
    directory = r"../Nature Raw Txt"
    EEGdata = pd.read_csv(directory + "/" + "Ball2_Nature_EEGData_fl10_N2.txt", header=None)
    EEGdata = EEGdata.drop(columns=[16], axis=1)

    return EEGdata

def rawTimePlots(channelData, title):

    # Data for plotting
    t = np.arange(0.0, 60.002, 0.002)
    s = channelData / 500

    fig, ax = plt.subplots()

    ax.plot(t, s)

    plt.xlim(10, 15)

    ax.set(xlabel='Time (s)', ylabel='Wave Amplitude (microV)', title=title)
    ax.grid()

    plt.show()

def savGolApp(channelData, length, order):
    filteredCh = scipy.signal.savgol_filter(channelData, length, order)
    return filteredCh


def plotBoth(channelData, filteredData):

    # Data for plotting
    t = np.arange(0.0, 60.002, 0.002)
    s = channelData / 500
    p = filteredData / 500
    fig, ax = plt.subplots()
    ax.plot(t, s, label="Unfiltered")
    ax.plot(t, p, label="Filtered")


    ax.set(xlabel='Time (s)', ylabel='Wave Amplitude (microV)', title="EEG Data, Filtered vs. Unfiltered")
    ax.grid()
    ax.legend(shadow=True, framealpha=1)
    plt.xlim(10, 15)

    plt.show()

def rawFreqPlots(channelData, title):


    plt.figure(2, figsize=(12, 9))

    #x value is frequency, 60 sec at 500Hz
    x = [None] * 30001
    for i in range (0, 30001):
        x[i] = i / 60.0


    #Plotting Ch1 and Ch9 data on same graph
    h = channelData
    plt.plot(x, abs(h))


    #show alpha band only
    plt.xlim(8, 13)

    #graph labels
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Wave Amplitude (microV)')
    plt.title(title)
    plt.legend(shadow=True, framealpha=1)
    plt.ylim(-10000, 10000)
    plt.show()

def plotBothFreq(channelData, filteredData):

    plt.figure(2, figsize=(12, 9))

    #x value is frequency, 60 sec at 500Hz
    x = [None] * 30001
    for i in range (0, 30001):
        x[i] = i / 60.0


    #Plotting Ch1 and Ch9 data on same graph
    h = channelData
    plt.plot(x, abs(h), label="Unfiltered")
    h = filteredData
    plt.plot(x, abs(h), label="Filtered")

    #show alpha band only
    plt.xlim(8, 13)

    #graph labels
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Wave Amplitude (microV)')
    plt.title('EEG Data in Frequency Domain: Filtered vs. Unfiltered')
    plt.legend(shadow=True, framealpha=1)
    plt.ylim(-10000, 10000)
    plt.show()

def main(length, order):

    EEGdata = readIn()
    Ch1 = pd.Series(EEGdata[0]).values
    #rawTimePlots(Ch1, "EEG Data in Time Domain (unfiltered)")
    filteredCh1 = savGolApp(Ch1, length, order)
    rawTimePlots(filteredCh1, "EEG Data in Time Domain (filtered)")
    plotBoth(Ch1, filteredCh1)

    freqCh1 = scipy.fft.fft(Ch1)
    filteredFreq1 = scipy.fft.fft(filteredCh1)
    #rawFreqPlots(freqCh1, "EEG Data in Frequency Domain (unfiltered)")
    rawFreqPlots(filteredFreq1, "EEG Data in Frequency Domain (filtered)")
    plotBothFreq(freqCh1, filteredFreq1)



if __name__ == "__main__":
    main(100, 2)

