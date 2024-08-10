import numpy as np
import statistics
import statsmodels.api as sm
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


def tcc(delta, ei, ej):
    m = len(ei)

    if delta >= 0:
        tcc_val = 0
        for i in range (0, m - delta):
            tcc_val += ei[i+delta] * ej[i]
    else:
        tcc_val = tcc(-1*delta, ej, ei)

    return tcc_val


def disp(ei, ej):
    #note: don't use on frequency domain, only time series
    m = len(ei)

    #subtract mean from vectors
    normalized_ei = ei - np.mean(ei)
    normalized_ej = ej - np.mean(ej)

    #find correlation between normalized means
    correlation = scipy.signal.correlate(normalized_ei, normalized_ej)

    #divide by vector length and standard deviations to normalize
    normalizing_factor = m * np.std(ei, ddof=1) * np.std(ej, ddof=1)
    norm_correlation = correlation / normalizing_factor


    #displacement is mean of normalized cross-correlations for every lag
    disp =  np.mean(norm_correlation)
    return disp


def plotCorrelation():
    signalFrame = pd.read_csv(r"../Nature Raw Txt/Ball2_Nature_EEGData_fl10_N2.txt", header=None)
    signalFrame2 = pd.read_csv(r"../Nature Raw Txt/Ball2_Nature_EEGData_fl10_N8.txt", header=None)

    signal1 = pd.Series(signalFrame[0])
    signal2 = pd.Series(signalFrame[8])

    signal3 = pd.Series(signalFrame2[0])
    signal4 = pd.Series(signalFrame2[8])

    m = len(signal1)


    delta_values = []
    for i in range(-1*m + 1, m):
        time = i / 500
        delta_values.append(time)


    corr_values = scipy.signal.correlate(signal1 - np.mean(signal1), signal2 - np.mean(signal2)) / (m * np.std(signal1, ddof = 1) * np.std(signal2, ddof = 1))
    corr_values2 = scipy.signal.correlate(signal3 - np.mean(signal3), signal4 - np.mean(signal4)) / (m * np.std(signal3, ddof = 1) * np.std(signal4, ddof = 1))



    plt.figure(2, figsize=(12, 9))

    plt.plot(delta_values, corr_values2, label="N8")
    plt.plot(delta_values, corr_values, label="N2")
    plt.xlabel("Lag (s)")
    plt.ylabel("Normalized Cross Correlation")
    plt.legend(shadow=True, framealpha=1)
    plt.xlim(-5, 5)
    plt.show()
    print(corr_values[m])
    print(corr_values2[m])



def improvedCosine(alpha, ei, ej):

    total_cosine_similarity = np.dot(ei, ej) / (np.linalg.norm(ei) * np.linalg.norm(ej))

    cross_correlation_displacement = disp(ei, ej)

    dist_ei_ej = (alpha/2) * (1 - total_cosine_similarity) + (1 - alpha) * cross_correlation_displacement

    return dist_ei_ej


def main():
    weight_coefficient = 0.5

    signalFrame = pd.read_csv(r"../Nature Raw Txt/Ball2_Nature_EEGData_fl10_N2.txt", header=None)

    signal1 = pd.Series(signalFrame[0])

    signal2 = pd.Series(signalFrame[8])
    #signal2 = pd.Series(signalFrame[8]) #[0, -1, -3, -4, -7, -2]
    #signal2 = signal1 #[6, 5, 4, 3, 2, 1]
    #signal2 = [6, 8, 2, 4, 7, 8]

    print("LAG 0")
    print(tcc(0, signal1 - np.mean(signal1), signal2 - np.mean(signal2)) / len(signal1) / (np.std(signal1) * np.std(signal2)))
    print(scipy.stats.pearsonr(signal1, signal2)[0])

    oldcos = np.dot(signal1, signal2) / (np.linalg.norm(signal1) * np.linalg.norm(signal2))
    print("Traditional Cos Similarity")
    print(oldcos)

    cos = improvedCosine (weight_coefficient, signal1, signal2)
    print("Improved Cos Similarity")
    print(cos)


    corr = disp(signal1, signal2)
    print("Normalized Cross Correlation")
    print(corr)



    #normalize correlation with St. Andrews equations
    #try to see cross-correlation with different lags
    #cross-correlation as its own thing and also in modified cosine



if __name__ == "__main__":
    main()