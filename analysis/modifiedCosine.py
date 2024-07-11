import numpy as np
import statistics
import statsmodels.api as sm
import scipy.signal
import pandas as pd

'''
def tcc(delta, ei, ej):
    m = len(ei)

    if delta >= 0:
        tcc_val = 0
        for i in range (0, m - delta):
            tcc_val += ei[i+delta] * ej[i]
    else:
        tcc_val = tcc(-1*delta, ej, ei)

    return tcc_val
'''

def disp(ei, ej):
    m = len(ei)

    normalized_ei = ei / np.linalg.norm(ei)
    normalized_ej = ej / np.linalg.norm(ej)

    correlation = scipy.signal.correlate(normalized_ei, normalized_ej)
    #sum = np.sum(correlation)

    #factor1 = np.sum(normalized_ei ** 2)
    #factor2 = np.sum(normalized_ej ** 2)
    #normalizing_factor = np.sqrt(factor1 * factor2)
    #normalizing_factor = np.linalg.norm(normalized_ei) * np.linalg.norm(normalized_ej)

    #note: don't use on frequency domain, only time series


    disp =  np.mean(correlation) #normalizing_factor / (2*m - 1)

    return disp

def improvedCosine(alpha, ei, ej):

    total_cosine_similarity = np.dot(ei, ej) / (np.linalg.norm(ei) * np.linalg.norm(ej))

    cross_correlation_displacement = disp(ei, ej)

    dist_ei_ej = (alpha/2) * (1 - total_cosine_similarity) + (1 - alpha) * cross_correlation_displacement

    return dist_ei_ej


def main():
    weight_coefficient = 0.5

    signalFrame = pd.read_csv(r"../Nature Raw Txt/Ball2_Nature_EEGData_fl10_N2.txt", header=None)

    signal1 = pd.Series(signalFrame[0]) #[0, 1, 3, 4, 7, 2]
    signal2 = pd.Series(signalFrame[8]) #[0, -1, -3, -4, -7, -2]
    #signal2 = [6, 8, 2, 4, 7, 8]

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