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


    ax.set(xlabel='Time (s)', ylabel='Wave Amplitude (V)', title="EEG Data, Filtered vs. Unfiltered")
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
    plt.ylabel('Wave Amplitude (V)')
    plt.title('EEG Data in Frequency Domain: Filtered vs. Unfiltered')
    plt.legend(shadow=True, framealpha=1)
    plt.ylim(0, 50000)
    plt.show()



    #code below is largely from nirpyresearch article
    #https://nirpyresearch.com/choosing-optimal-parameters-savitzky-golay-smoothing-filter/
def findPowerSpectrum(ChA):
    # Calculate the power spectrum
    regionStart, regionEnd = 13300, 13500
    ps = np.abs(np.fft.fftshift(np.fft.fft(ChA[regionStart:regionEnd]))) ** 2

    # Define pixel in original signal and Fourier Transform
    #pix = np.arange(ChA[regionStart:regionEnd].shape[0])
    pix = np.arange(regionStart/500, regionEnd/500, (1/500))
    #fpix = np.arange(ps.shape[0]) - ps.shape[0] // 2
    fpix = np.arange(0, 500, (500/200)) - 500 // 2

    with plt.style.context(('ggplot')):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        axes[0].plot(pix, ChA[regionStart:regionEnd])
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('EEG spectrum')

        axes[1].semilogy(fpix, ps, 'b')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Power Spectrum')

def powerSpectrum(ChA):
    # Set some reasonable parameters to start with
    w = 5
    p = 2
    # Calculate three different smoothed spectra

    numtaps1 = 200
    cutoff1 = 40
    samplingFreq = 500
    hammingCoeffs = scipy.signal.firwin(numtaps1, cutoff1, window='hamming', pass_zero='lowpass', fs=samplingFreq)

    denoms = [1] * len(hammingCoeffs)

    # Do we give it frequency or time series? What are the units? Does it return frequency or time series?
    #filteredData = scipy.signal.lfilter(hammingCoeffs, denoms, ChA)

    #X_smooth_1 = scipy.signal.savgol_filter(filteredData, 20, 15)
    X_smooth_1 = scipy.signal.savgol_filter(ChA, w, polyorder=p, deriv=0)
    X_smooth_2 = scipy.signal.savgol_filter(ChA, 2 * w + 1, polyorder=p, deriv=0)
    X_smooth_3 = scipy.signal.savgol_filter(ChA, 20, 4)
    #X_smooth_3 = scipy.signal.lfilter(hammingCoeffs, denoms, ChA)
    #X_smooth_3 = scipy.signal.savgol_filter(ChA, 4 * w + 1, polyorder=3 * p, deriv=0)

    # Calculate the power spectra in a featureless region
    regionStart, regionEnd = 13300, 13500
    ps = np.abs(np.fft.fftshift(np.fft.fft(ChA[regionStart:regionEnd]))) ** 2
    ps_1 = np.abs(np.fft.fftshift(np.fft.fft(X_smooth_1[regionStart:regionEnd]))) ** 2
    ps_2 = np.abs(np.fft.fftshift(np.fft.fft(X_smooth_2[regionStart:regionEnd]))) ** 2
    ps_3 = np.abs(np.fft.fftshift(np.fft.fft(X_smooth_3[regionStart:regionEnd]))) ** 2

    # Define pixel in Fourier space
    #fpix = np.arange(ps.shape[0]) - ps.shape[0] // 2
    fpix = np.arange(0, 500, (500/200)) - 500 // 2

    plt.figure(figsize=(10, 8))
    with plt.style.context(('ggplot')):
        plt.semilogy(fpix, ps, 'b', label='No smoothing')
        #plt.semilogy(fpix, ps_1, 'r', label='Smoothing: w/p = 2.5')
        #plt.semilogy(fpix, ps_2, 'g', label='Smoothing: w/p = 5.5')
        plt.semilogy(fpix, ps_3, 'm', label='Smoothing: w=15, p=5')
        plt.legend()
        plt.xlabel('Frequency (Hz)')

        plt.show()




def main(length, order):

    EEGdata = readIn()
    Ch1 = pd.Series(EEGdata[0]).values

    findPowerSpectrum(Ch1)
    powerSpectrum(Ch1)

    #rawTimePlots(Ch1, "EEG Data in Time Domain (unfiltered)")
    filteredCh1 = savGolApp(Ch1, length, order)
    #rawTimePlots(filteredCh1, "EEG Data in Time Domain (filtered)")
    plotBoth(Ch1, filteredCh1)

    freqCh1 = scipy.fft.fft(Ch1)
    filteredFreq1 = scipy.fft.fft(filteredCh1)
    #rawFreqPlots(freqCh1, "EEG Data in Frequency Domain (unfiltered)")
    #rawFreqPlots(filteredFreq1, "EEG Data in Frequency Domain (filtered)")
    plotBothFreq(freqCh1, filteredFreq1)



if __name__ == "__main__":
    main(15, 5)

