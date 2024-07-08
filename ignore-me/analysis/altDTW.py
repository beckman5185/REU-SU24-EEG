import numpy as np
import pandas as pd
from dtw import *
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def main():
    directory = r"Nature Raw Txt"
    EEGdata = pd.read_csv(directory + "/" + "Ball2_Nature_EEGData_fl10_N2.txt", header=None)

    Ch1 = np.array([EEGdata[0].values])
    Ch9 = np.array([EEGdata[8].values])

    distance, path = fastdtw(Ch1, Ch9, dist=euclidean)
    print(distance)
    alignment = dtw(Ch1, Ch9, keep_internals=True)
    print(alignment.distance)

    alignment.plot(type="threeway")


if __name__ == "__main__":
    main()