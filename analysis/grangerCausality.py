from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd

def readIn():
    directory = r"../Nature Raw Txt"
    EEGdata = pd.read_csv(directory + "/" + "Ball2_Nature_EEGData_fl10_N2.txt", header=None)
    EEGdata = EEGdata.drop(columns=[16], axis=1)

    return EEGdata
def main():
    EEGdata = readIn()

    grangercausalitytests(EEGdata[[0, 8]], maxlag=[1])
    grangercausalitytests(EEGdata[[8, 0]], maxlag=[1])

if __name__ == "__main__":
    main()