import pingouin as pg
import pandas as pd
import os

def main():

    #methodList = ['cosine_similarity', 'RMS_similarity', 'peak_similarity', 'SSD_similarity', 'DTW_similarity', 'LCS_similarity']
    methodList = ['cosine_similarity', 'RMS_similarity', 'peak_similarity', 'SSD_similarity', 'LCS_similarity']
    paramsList = ['time-unfiltered-output', 'time-filtered-output', 'frequency-unfiltered-output', 'frequency-filtered-output']

    style = paramsList[1]

    for i in range(len(methodList)):
        directory = style + "/" + methodList[i]
        for file in os.listdir(directory):

            coherenceData = pd.read_csv(directory + "/" + file, header=0)
            rmanova = pg.mixed_anova(coherenceData, dv='Coherency', within='Sound', subject='Subject', between='Gender')
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