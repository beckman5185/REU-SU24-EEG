import pandas as pd
import scipy.signal
import scipy.fft
import matplotlib.pyplot as plt
import numpy as np

def main():

    #reads csv, taken from filterPlay.py
    directory = r"../Nature Raw Txt"
    EEGdata = pd.read_csv(directory + "/" + "Ball_Nature_EEGData_fl10_N2.txt", header=None)
    EEGdata = EEGdata.drop(columns=[16], axis=1)

    #Selects specific channels for comparison
    Ch1 = pd.Series(EEGdata[0])
    Ch9 = pd.Series(EEGdata[8])

    numSegs = 1024
    #Finds imaginary component. 
    #Taken from: https://dsp.stackexchange.com/questions/61661/how-to-compare-imaginary-coherence-values
    f1, Pxx = scipy.signal.welch(Ch1, fs=500, nperseg = numSegs )
    f2, Pyy = scipy.signal.welch(Ch9, fs=500, nperseg = numSegs )
    f3, Pxy = scipy.signal.csd(Ch1, Ch9, fs=500, nperseg = numSegs)
    Icoh = np.abs(np.imag(Pxy))/np.sqrt(Pxx*Pyy)
    Coh = np.abs((Pxy))/np.sqrt(Pxx*Pyy)

    xAxis = np.arange(0, 500 + (500/(numSegs/2)), (500/(numSegs/2)))

    #Audrey added to play around with it
    #sample_frequencies1, coherence_series1 = scipy.signal.coherence(Ch1, Ch9, fs=500, nperseg=2048)
    #sample_frequencies3, coherence_series3 = scipy.signal.coherence(Ch1, Ch9, fs=500, nperseg=512)
    #sample_frequencies4, coherence_series4 = scipy.signal.coherence(Ch1, Ch9, fs=500)


    sample_frequencies2, coherence_series2 = scipy.signal.coherence(Ch1, Ch9, fs=500, nperseg=1024)
    coherence_series5, sample_frequencies5 = plt.cohere(Ch1, Ch9, Fs=500, NFFT=1024)

    f1, Pxx = scipy.signal.welch(Ch1, fs=500, nperseg=numSegs)
    f2, Pyy = scipy.signal.welch(Ch9, fs=500, nperseg=numSegs)
    f3, Pxy = scipy.signal.csd(Ch1, Ch9, fs=500, nperseg=numSegs)
    #Coh = np.abs(Pxy) ** 2 / (Pxx * Pyy)
    #ICoh = np.abs(np.imag(Pxy)) ** 2 / (Pxx * Pyy)



    #print(Icoh)

    #plt.xlim(0,60)
    #Plots resulting array
    #plt.plot(xAxis, Icoh, label="ICoh")
    #plt.plot(xAxis, Coh, label="Coh")
    #plt.legend(shadow=True, framealpha=1)
    #plt.show()

    #plt.xlim(0, 20)
    #plt.plot(sample_frequencies1, coherence_series1)
    #plt.plot(xAxis / 2, Icoh, label="ICoh")
    #plt.plot(sample_frequencies3, coherence_series3)
    plt.plot(xAxis / 2, Coh, label="Coh")
    #plt.plot(sample_frequencies2, coherence_series2, label="Calculation by Scipy Method")
    #plt.plot(sample_frequencies5, coherence_series5, label="Calculation by MatPlotLib Method")
    plt.legend(shadow=True, framealpha=1)
    plt.show()

   


if __name__ == "__main__":
    main()
