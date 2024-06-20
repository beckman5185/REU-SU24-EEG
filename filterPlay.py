import pandas as pd
import scipy.signal
import scipy.fft
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 20})


def main():
    directory = r"Nature Raw Txt"
    EEGdata = pd.read_csv(directory + "/" + "Ball2_Nature_EEGData_fl10_N2.txt", header=None)
    EEGdata = EEGdata.drop(columns=[16], axis=1)

    Ch1 = pd.Series(EEGdata[0])

    freqCh1 = scipy.fft.fft(Ch1.values)

    plt.figure(2, figsize=(12, 9))

    # Data for plotting
    #t = np.arange(0.0, 60.002, 0.002)
    #s = freqCh1 /500.0

    #fig, ax = plt.subplots()


    #ax.plot(t, s)

    #ax.set(xlabel='time (s)', ylabel='Microvolts (muV)', title='EEG Data Sample')
    #ax.grid()
    #plt.ylim(0,45)

    #plt.show()



    #x value is frequency, 60 sec at 500Hz
    x = [None] * 30001
    for i in range (0, 30001):
        x[i] = i / 60.0

    #not sure what units original EEG data are in?
    h = freqCh1
    plt.plot(x, abs(h))


    plt.xlim(8, 13)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Wave Amplitude (microV)')
    plt.title('EEG Data in Frequency Domain (unfiltered)')
    plt.ylim(0, 60000)
    plt.show()
    #plt.grid(True)


def filterCompare():

    numtaps1 = 100
    numtaps2 = 200
    cutoff1 = 40
    cutoff2 = 45
    samplingFreq = 500
    hammingCoeffs = scipy.signal.firwin(numtaps1, cutoff1, window='hamming', pass_zero='lowpass', fs=samplingFreq)
    hammingCoeffs2 = scipy.signal.firwin(numtaps2, cutoff1, window='hamming', pass_zero='lowpass', fs=samplingFreq)
    hammingCoeffs3 = scipy.signal.firwin(numtaps1, cutoff2, window='hamming', pass_zero='lowpass', fs=samplingFreq)
    hammingCoeffs4 = scipy.signal.firwin(numtaps2, cutoff2, window='hamming', pass_zero='lowpass', fs=samplingFreq)
    rectCoeffs = scipy.signal.firwin(numtaps1, cutoff1, window='boxcar', pass_zero='lowpass', fs=samplingFreq)
    rectCoeffs2 = scipy.signal.firwin(numtaps2, cutoff1, window='boxcar', pass_zero='lowpass', fs=samplingFreq)
    rectCoeffs3 = scipy.signal.firwin(numtaps1, cutoff2, window='boxcar', pass_zero='lowpass', fs=samplingFreq)
    rectCoeffs4 = scipy.signal.firwin(numtaps2, cutoff2, window='boxcar', pass_zero='lowpass', fs=samplingFreq)

    #print(hammingCoeffs)
    #print(rectCoeffs)

    # Plot the frequency responses of the filters.
    plt.figure(1, figsize=(12, 9))
    plt.clf()

    # First plot the desired ideal response as a green(ish) rectangle.
    rect = plt.Rectangle((0, 0), 30, 1.0,
                         facecolor="#60ff60", alpha=0.2)
    plt.gca().add_patch(rect)

    # Plot the frequency response of each filter.
    w, h = scipy.signal.freqz(hammingCoeffs, 1, worN=2000, fs=samplingFreq)
    plt.plot(w, abs(h), label="100 taps 40Hz")

    #w, h = scipy.signal.freqz(hammingCoeffs2, 1, worN=2000, fs=samplingFreq)
    #plt.plot(w, abs(h), label="200 taps 40Hz")

    w, h = scipy.signal.freqz(hammingCoeffs3, 1, worN=2000, fs=samplingFreq)
    plt.plot(w, abs(h), label="100 taps 45Hz")

    #w, h = scipy.signal.freqz(hammingCoeffs4, 1, worN=2000, fs=samplingFreq)
    #plt.plot(w, abs(h), label="200 taps 45Hz")

    #w, h = scipy.signal.freqz(rectCoeffs, 1, worN=2000, fs=samplingFreq)
    #plt.plot(w, abs(h), label="Rectangular window 100 taps 40Hz")

    plt.xlim(0, 50)
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend(shadow=True, framealpha=1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency response of Hamming window FIR filters')

    plt.show()


    plt.figure(1, figsize=(12, 9))
    plt.clf()

    # First plot the desired ideal response as a green(ish) rectangle.
    rect = plt.Rectangle((0, 0), 30, 1.0,
                         facecolor="#60ff60", alpha=0.2)
    plt.gca().add_patch(rect)

    # Plot the frequency response of each filter.
    w, h = scipy.signal.freqz(hammingCoeffs, 1, worN=2000, fs=samplingFreq)
    plt.plot(w, abs(h), label="Hamming window")

    w, h = scipy.signal.freqz(rectCoeffs, 1, worN=2000, fs=samplingFreq)
    plt.plot(w, abs(h), label="Boxcar window")

    plt.xlim(0, 50)
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend(shadow=True, framealpha=1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency response of FIR filters, 100 taps 40 Hz')
    plt.show()


    hammingCoeffsFinal = scipy.signal.firwin(numtaps2, [1, cutoff1], window='hamming', pass_zero='bandpass', fs=samplingFreq)
    plt.figure(1, figsize=(12, 9))
    plt.clf()

    # First plot the desired ideal response as a green(ish) rectangle.
    rect = plt.Rectangle((4, 0), 26, 1.0,
                         facecolor="#60ff60", alpha=0.2)
    plt.gca().add_patch(rect)

    # Plot the frequency response of each filter.
    w, h = scipy.signal.freqz(hammingCoeffsFinal, 1, worN=2000, fs=samplingFreq)
    plt.plot(w, abs(h))

    plt.xlim(0, 50)
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('FIR Filter: Hamming, bandpass 1-40Hz, 200 taps')
    plt.show()


def tryFilter():
    directory = r"Nature Raw Txt"
    EEGdata = pd.read_csv(directory + "/" + "Ball2_Nature_EEGData_fl10_N2.txt", header=None)
    EEGdata = EEGdata.drop(columns=[16], axis=1)

    #Data for plotting
    t = np.arange(0.0, 60.002, 0.002)
    s = pd.Series(EEGdata[0]).values / 2

    fig, ax = plt.subplots()

    ax.plot(t, s)

    ax.set(xlabel='Time (s)', ylabel='Wave Amplitude (microV)', title='EEG Data in Time Domain (unfiltered)')
    ax.grid()

    plt.show()

    Ch1 = scipy.fft.fft(pd.Series(EEGdata[0]).values)

    numtaps1 = 100
    cutoff1 = 40
    samplingFreq = 500
    hammingCoeffs = scipy.signal.firwin(numtaps1, cutoff1, window='hamming', pass_zero='lowpass', fs=samplingFreq)

    denoms = [1] * len(hammingCoeffs)

    filteredData = scipy.signal.lfilter(hammingCoeffs, denoms, Ch1)

    hammingCoeffs2 = scipy.signal.firwin(numtaps1, [1, cutoff1], window='hamming', pass_zero='bandpass', fs=samplingFreq)

    filterData2 = scipy.signal.lfilter(hammingCoeffs2, denoms, pd.Series(EEGdata[0]).values)

    x = [None] * 30001
    for i in range(0, 30001):
        x[i] = i / 60.0

    h = filterData2

    plt.xlim(8, 13)
    plt.ylim(0, 20)

    plt.title('Test Graph: FIR on time domain, plotted against frequency')
    plt.plot(x, abs(h))
    plt.show()







    x = [None] * 30001
    for i in range(0, 30001):
        x[i] = i / 500.0

    h = scipy.fft.ifft(filterData2)

    plt.xlim(10, 30)
    plt.ylim(-0.05, 0.05)

    plt.title('Reversed IFFT for Filtered Data')

    plt.plot(x, h)
    plt.show()







    # x value is time, 60 sec at 500Hz
    x = [None] * 30001
    for i in range(0, 30001):
        x[i] = i / 500.0

    # not sure what units original EEG data are in?
    h = filteredData / 500 # to convert it to volts?
    plt.plot(x, abs(h))

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spectral density (V)?')
    plt.title('Frequency Domain EEG Data (filtered)')

    #plt.ylim(0, 60)
    plt.show()






    x = [None] * 30001
    for i in range(0, 30001):
        x[i] = i / 500.0

    timeFiltered = scipy.fft.ifft(filteredData)



    # Data for plotting
    t = np.arange(0.0, 60.002, 0.002)
    s = timeFiltered / 1000


    fig, ax = plt.subplots()

    ax.plot(t, s)

    ax.set(xlabel='time (s)', ylabel='Microvolts (muV)', title='EEG Data Time Domain (Filtered)')
    ax.grid()
    #plt.ylim(0,45)

    plt.show()




    #plt.plot(x, abs(h))

    #plt.ylim(0, 60)
    #plt.show()


def tryFilterNoGraphs():
    directory = r"Nature Raw Txt"
    EEGdata = pd.read_csv(directory + "/" + "Ball2_Nature_EEGData_fl10_N2.txt", header=None)
    EEGdata = EEGdata.drop(columns=[16], axis=1)


    #Units: microvolts
    Ch1 = pd.Series(EEGdata[0])
    print(Ch1)


    #Units: Hz
    freqCh1 = scipy.fft.fft(pd.Series(EEGdata[0]).values)
    print(freqCh1)

    numtaps1 = 100
    cutoff1 = 40
    samplingFreq = 500
    hammingCoeffs = scipy.signal.firwin(numtaps1, cutoff1, window='hamming', pass_zero='lowpass', fs=samplingFreq)

    denoms = [1] * len(hammingCoeffs)

    #Do we give it frequency or time series? What are the units? Does it return frequency or time series?
    filteredData = scipy.signal.lfilter(hammingCoeffs, denoms, freqCh1)

    print(filteredData)


    #Units: ?
    timeFiltered = scipy.fft.ifft(filteredData)

    print(timeFiltered)


if __name__ == "__main__":
    #main()
    filterCompare()
    #tryFilter()
    #tryFilterNoGraphs()