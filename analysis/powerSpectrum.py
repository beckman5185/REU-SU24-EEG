import scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#code is largely from https://raphaelvallat.com/bandpower.html
def powerSpectrum(seriesA):
    # Define window length (4 seconds)
    sf = 500
    win = 4 * sf
    freqs, psd = scipy.signal.welch(seriesA, sf, nperseg=win)

    freqs = np.arange(0, 500 + (500/30000), (500/30000))
    psd = np.abs(scipy.fft.fft(seriesA.values)) ** 2

    low, high = 8, 13
    scale = 30000/500
    low_idx, high_idx = int(low*scale), int(high*scale)
    psd = psd[low_idx:high_idx]
    freqs = freqs[low_idx:high_idx]

    # Define delta lower and upper limits
    low, high = 8, 13

    # Find intersecting values in frequency vector
    #idx_alpha = np.logical_and(freqs >= low, freqs <= high)


    # Plot the power spectrum
    fig, ax = plt.subplots()
    plt.plot(freqs, psd, color='k', lw=2)
    #plt.fill_between(freqs, psd, where=idx_alpha, color='skyblue')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (V^2 / Hz)')
    plt.title("Welch's periodogram")
    #plt.xlim(8, 13)
    plt.ylim(0, 10**9)
    plt.show()

def main():
    directory = r"../Nature Raw Txt"
    EEGdata = pd.read_csv(directory + "/" + "Ball2_Nature_EEGData_fl10_N2.txt", header=None)
    EEGdata = EEGdata.drop(columns=[16], axis=1)

    Ch1 = pd.Series(EEGdata[0])
    powerSpectrum(Ch1)

if __name__ == "__main__":
    main()