import pandas as pd
import scipy.signal
import scipy.fft
import matplotlib.pyplot as plt
import numpy as np

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



    #x value is time, 60 sec at 500Hz
    x = [None] * 30001
    for i in range (0, 30001):
        x[i] = i / 500.0

    #not sure what units original EEG data are in?
    h = freqCh1 / 2
    plt.plot(x, abs(h))


    plt.ylim(0, 60)
    plt.show()
    #plt.grid(True)
    #plt.xlabel('Time (s)')
    #plt.ylabel('Frequency (Hz)')
    #plt.title('EEG Data in Frequency Domain for Ball2 Sound N2')


    numtaps1 = 100
    numtaps2 = 200
    cutoff1 = 40
    cutoff2 = 45
    samplingFreq = 500
    hammingCoeffs = scipy.signal.firwin(numtaps1, cutoff1, window='hamming', pass_zero='lowpass', fs=samplingFreq)
    hammingCoeffs2 = scipy.signal.firwin(numtaps2, cutoff1, window='hamming', pass_zero='lowpass', fs=samplingFreq)
    hammingCoeffs3 = scipy.signal.firwin(numtaps1, cutoff2, window='hamming', pass_zero='lowpass', fs=samplingFreq)
    hammingCoeffs4 = scipy.signal.firwin(numtaps2, cutoff2, window='hamming', pass_zero='lowpass', fs=samplingFreq)
    #rectCoeffs = scipy.signal.firwin(numtaps1, cutoff1, window='boxcar', pass_zero='lowpass', fs=samplingFreq)

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
    plt.plot(w, abs(h), label="Hamming window 100 taps 40Hz")

    w, h = scipy.signal.freqz(hammingCoeffs2, 1, worN=2000, fs=samplingFreq)
    plt.plot(w, abs(h), label="Hamming window 200 taps 40Hz")

    w, h = scipy.signal.freqz(hammingCoeffs3, 1, worN=2000, fs=samplingFreq)
    plt.plot(w, abs(h), label="Hamming window 100 taps 45Hz")

    w, h = scipy.signal.freqz(hammingCoeffs4, 1, worN=2000, fs=samplingFreq)
    plt.plot(w, abs(h), label="Hamming window 200 taps 45Hz")

    #w, h = scipy.signal.freqz(rectCoeffs, 1, worN=2000, fs=samplingFreq)
    #plt.plot(w, abs(h), label="Rectangular window 100 taps 40Hz")

    plt.xlim(0, 50)
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend(shadow=True, framealpha=1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency response of several FIR filters')

    plt.show()
    # plt.savefig('plot.png')

if __name__ == "__main__":
    main()