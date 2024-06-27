import pandas as pd
import numpy as np
import scipy.fft
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def cos_helper(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def RMS_helper(x, y):
    series = [None] * len(x)

    for i in range(0, len(x)):
        num1 = x[i]
        num2 = y[i]
        if (num1 != 0 or num2 != 0):
            series[i] = (1 - (abs(num1 - num2) / (abs(num1) + abs(num2)))) ** 2
        else:
            # if x[i] and y[i] are 0, special case (assume 0 to avoid division by 0)
            series[i] = 1

    return np.sqrt(np.mean(series))


def peak_helper(x, y):
    series = [None] * len(x)

    for i in range (0, len(x)):
        num1 = x[i]
        num2 = y[i]
        if (num1 != 0 or num2 != 0):
            series[i] = 1 - (abs(num1 - num2)/(2*max([abs(num1), abs(num2)])))
        else:
            # if x[i] and y[i] are 0, special case (assume 0 to avoid division by 0)
            series[i] = 1


    return np.mean(series)


def SSD_helper(x, y):
    return sum((x-y)**2)