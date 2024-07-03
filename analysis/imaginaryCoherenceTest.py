import pandas as pd
import scipy.signal
import scipy.fft
import matplotlib.pyplot as plt
import numpy as np

def main():

    #reads csv, taken from filterPlay.py
    directory = r"Nature Raw Txt"
    EEGdata = pd.read_csv(directory + "/" + "Ball2_Nature_EEGData_fl10_N2.txt", header=None)
    EEGdata = EEGdata.drop(columns=[16], axis=1)

    #Selects specific channels for comparison
    Ch1 = pd.Series(EEGdata[0])
    Ch9 = pd.Series(EEGdata[8])

    numSegs = 1024
    #Finds imaginary component. 
    #Taken from: https://dsp.stackexchange.com/questions/61661/how-to-compare-imaginary-coherence-values
    f1, Pxx = scipy.signal.welch(Ch1, fs=500, nperseg = numSegs )
    f2, Pyy = scipy.signal.welch(Ch1, fs=500, nperseg = numSegs )
    f3, Pxy = scipy.signal.csd(Ch1, Ch9, fs=500, nperseg = numSegs)
    Icoh = np.abs(np.imag(Pxy))/np.sqrt(Pxx*Pyy)
    Coh = np.abs((Pxy))/np.sqrt(Pxx*Pyy)

    xAxis = np.arange(0, 500 + (500/(numSegs/2)), (500/(numSegs/2)))

    #print(Icoh)

    plt.xlim(0,60)
    #Plots resulting array
    plt.plot(xAxis, Icoh, Coh)
    plt.show()

    plt.xlim(0,60)
    plt.plot(xAxis, Coh)
    plt.show()

   


if __name__ == "__main__":
    main()
