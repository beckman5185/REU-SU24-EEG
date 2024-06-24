import pandas as pd
import os

def main():

    #dictionary mapping participants to their data
    nameDict = {}

    #read each file in from folder
    directory = r"Nature Raw Txt"

    for fileName in os.listdir(directory):

        #get last name from file naming convention
        splitName = fileName.split('_')
        lastName = splitName[0]
        soundName = splitName[4].strip(".txt")


        #add last name to dictionary of participants
        if not lastName in nameDict:
            nameDict[lastName] = pd.Series()


        #add sound name to series of sound names in dictionary for certain participant
        if not soundName in nameDict[lastName].index:
            nameDict[lastName][soundName] = pd.Series()


        #add EEG data dataframe to series for certain sound
        EEGdata = pd.read_csv(directory + "/" + fileName, header=None)
        nameDict[lastName][soundName] = EEGdata.drop(columns=[16], axis=1)




    print(nameDict["Ball"]["N2"])





    #exampleDF = pd.read_csv('EEG_Files/Ball_Nature_EEG_N1_Seg1_Ch2_fl10.txt', header=None)[0]

    #print(exampleDF)

if __name__ == "__main__":
    main()