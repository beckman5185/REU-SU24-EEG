import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np



def readIn():
    directory = r"Nature Raw Txt"
    EEGdata = pd.read_csv(directory + "/" + "Ball2_Nature_EEGData_fl10_N2.txt", header=None)
    EEGdata = EEGdata.drop(columns=[16], axis=1)

    return EEGdata

def rawTimePlots(channelData, title):

    # Data for plotting
    t = np.arange(0.0, 60.002, 0.002)
    s = channelData / 2

    fig, ax = plt.subplots()

    ax.plot(t, s)

    ax.set(xlabel='Time (s)', ylabel='Wave Amplitude (microV)', title=title)
    ax.grid()

    plt.show()

def savGolApp(channelData):
    filteredCh = scipy.signal.savgol_filter(channelData, 100, 2)
    return filteredCh


def plotBoth(channelData, filteredData):

    # Data for plotting
    t = np.arange(0.0, 60.002, 0.002)
    s = channelData / 2
    p = filteredData / 2
    fig, ax = plt.subplots()
    ax.plot(t, s, label="Unfiltered")
    ax.plot(t, p, label="Filtered")


    ax.set(xlabel='Time (s)', ylabel='Wave Amplitude (microV)', title="EEG Data, Filtered vs. Unfiltered")
    ax.grid()
    ax.legend(shadow=True, framealpha=1)
    plt.xlim(10, 15)

    plt.show()

def main():
    EEGdata = readIn()
    Ch1 = pd.Series(EEGdata[0]).values
    rawTimePlots(Ch1, "EEG Data in Time Domain (unfiltered)")
    filteredCh1 = savGolApp(Ch1)
    rawTimePlots(filteredCh1, "EEG Data in Time Domain (filtered)")
    plotBoth(Ch1, filteredCh1)

if __name__ == "__main__":
    main()

