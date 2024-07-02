import pingouin as pg
import pandas as pd
import os

def main():

    #list of methods and styles: use first method list for time domain, second method list for frequency domain
    #no DTW for frequency domain ^
    #methodList = ['cosine_similarity', 'RMS_similarity', 'peak_similarity', 'SSD_similarity', 'DTW_similarity', 'LCS_similarity']
    methodList = ['cosine_similarity', 'RMS_similarity', 'peak_similarity', 'SSD_similarity', 'LCS_similarity']
    paramsList = ['time-unfiltered-output', 'time-filtered-output', 'frequency-unfiltered-output', 'frequency-filtered-output']

    #choosing which parameters to do analysis for
    style = paramsList[1]

    #for each method
    for i in range(len(methodList)):
        directory = style + "/" + methodList[i]
        #for each channel pair
        for file in os.listdir(directory):

            #read in coherency data table and perform mixed ANOVA
            #between = gender, within = sounds
            coherenceData = pd.read_csv(directory + "/" + file, header=0)
            rmanova = pg.mixed_anova(coherenceData, dv='Coherency', within='Sound', subject='Subject', between='Gender')

            #print ANOVA results to file
            with open(style + ".txt", "a") as f:
                print("Channel pair: " + file, file=f)
                print("Method of analysis: " + methodList[i], file=f)
                print(rmanova.to_string(), file=f)
                print(file=f)

def test():
    methodList = ['cosine_similarity', 'RMS_similarity', 'peak_similarity', 'SSD_similarity', 'DTW_similarity', 'LCS_similarity']
    #methodList = ['cosine_similarity', 'RMS_similarity', 'peak_similarity', 'SSD_similarity', 'LCS_similarity']
    paramsList = ['time-unfiltered-output', 'time-filtered-output', 'frequency-unfiltered-output', 'frequency-filtered-output']

    style = paramsList[1]


    with open("test.txt", "r") as file:

        coherenceData = pd.read_csv(file, header=0)
        rmanova = pg.mixed_anova(coherenceData, dv='Coherency', within='Sound', subject='Subject', between='Gender')
        with open("test-" + style + ".txt", "a") as f:
            print("Channel pair: C3-C4", file=f)
            print("Method of analysis: LCS_similarity", file=f)
            print(rmanova.to_string(), file=f)
            print(file=f)


if __name__ == "__main__":
    main()