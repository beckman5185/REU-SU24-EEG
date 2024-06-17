import pandas as pd

def euclideanDistance(vectorX, vectorY):
    distSquared = 0.0

    for i in range(0, len(vectorX)):
        distSquared += (vectorX[i] - vectorY[i]) ** 2

    return distSquared ** (1 / 2)




def main():
    directory = r"Nature Raw Txt"
    EEGdata = pd.read_csv(directory + "/" + "Ball2_Nature_EEGData_fl10_N2.txt", header=None)

    #Pearson correlation
    print(EEGdata.corr())

    Ch1 = EEGdata[0]
    Ch9 = EEGdata[8]

    distance = euclideanDistance(Ch1, Ch9)

    print(distance)

if __name__ == "__main__":
    main()
