import pandas as pd
import os
import re

def main():

    #dictionary mapping participants to their data
    nameDict = {}

    directory = r"Nature Raw Txt"
    for name in os.listdir(directory):

        splitName = name.split('_')

       # type = splitName[2]
        #if type == "PPG":
         #   os.remove(os.path.join(directory, name))
          #  print(f"Removed {name}")

        lastName = splitName[0]




        if not lastName in nameDict:
            nameDict[lastName] = pd.Series()


        soundName = splitName[4].strip(".txt")

        if not soundName in nameDict[lastName].index:
            nameDict[lastName][soundName] = pd.Series()

        fileName = directory + "/" + name
        nameDict[lastName][soundName] = pd.read_csv(fileName, header=None)


    print(nameDict["Ball"]["N2"])





    #exampleDF = pd.read_csv('EEG_Files/Ball_Nature_EEG_N1_Seg1_Ch2_fl10.txt', header=None)[0]

    #print(exampleDF)

if __name__ == "__main__":
    main()