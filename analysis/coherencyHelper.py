import pandas as pd
import numpy as np
import scipy.fft
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def cos_helper(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def RMS_helper(x, y):
    sumSquares = 0

    for i in range(0, len(x)):
        numSim = 1 - (abs(x[i] - y[i]) / (abs(x[i]) + abs(y[i])))
        sumSquares += numSim ** 2

    return np.sqrt(sumSquares/len(x))


def peak_helper(x, y):
    sumVal = 0
    for i in range (0, len(x)):
        sumVal += 1 - (abs(x[i] - y[i])/(2*max([abs(x[i]), abs(y[i])])))

    return sumVal/len(x)


def SSD_helper(x, y):
    return sum((x-y)**2)