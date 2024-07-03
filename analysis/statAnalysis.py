import pingouin as pg
import pandas as pd
import os
import scipy.stats

def leveneTest(dataset):
    #get index for gender and coherency
    gender_index = dataset.columns.get_loc('Gender')
    coherency_index = dataset.columns.get_loc('Coherency')

    #start lists
    f_sample = []
    m_sample = []

    #separate coherency values based on gender of subject
    for row in dataset:
        if row[gender_index] == "F":
            f_sample.append(row[coherency_index])
        elif row[gender_index] == "M":
            m_sample.append(row[coherency_index])

    #perform levene test
    testStat, pVal = scipy.stats.levene(f_sample, m_sample)

    #see if levene test p value is significant
    if pVal <= 0.05:
        leveneResult = False
    else:
        leveneResult = True

    return leveneResult



def main():

    #list of methods and styles: use first method list for time domain, second method list for frequency domain
    #no DTW for frequency domain ^
    #methodList = ['cosine_similarity', 'RMS_similarity', 'peak_similarity', 'SSD_similarity', 'DTW_similarity', 'LCS_similarity']
    methodList = ['cosine_similarity', 'RMS_similarity', 'peak_similarity', 'SSD_similarity', 'LCS_similarity']
    paramsList = ['time-unfiltered-output', 'time-filtered-output', 'alpha-unfiltered-output',
                  'alpha-filtered-output', 'gamma-unfiltered-output', 'gamma-filtered-output']

    #choosing which parameters to do analysis for
    style = paramsList[5]

    #for each method
    for i in range(len(methodList)):
        directory = style + "/" + methodList[i]
        #for each channel pair
        for file in os.listdir(directory):

            #read in coherency data table and perform mixed ANOVA
            #between = gender, within = sounds
            coherenceData = pd.read_csv(directory + "/" + file, header=0)

            leveneResult = leveneTest(coherenceData)



            rmanova = pg.mixed_anova(coherenceData, dv='Coherency', within='Sound', subject='Subject', between='Gender')

            #name = "significance-results//" + style + "//" + file.strip(".csv") + "-" + methodList[i] + ".xlsx"
            #rmanova.to_excel(name)

            #p_unc_index = rmanova.columns.get_loc('p-unc')
            #p_corr_index = rmanova.columns.get_loc('p-GG-corr')

            #for row in rmanova.iterrows():

            #    if row[p_unc_index] <= 0.05 or row[p_corr_index] <= 0.05:
            #        with open("signficance-results//" + style + ".txt", "a") as f:
            #            print("Channel pair: " + file, file=f)
            #            print("Method of analysis: " + methodList[i], file=f)
            #            print(row.to_string(), file=f)
            #            print(file=f)

            #print ANOVA results to file
            with open(style + ".txt", "a") as f:
                print("Channel pair: " + file, file=f)
                print("Method of analysis: " + methodList[i], file=f)
                print("Levene test: " + str(leveneResult), file=f)
                print(rmanova.to_string(), file=f)
                print(file=f)

def test():
    methodList = ['cosine_similarity', 'RMS_similarity', 'peak_similarity', 'SSD_similarity', 'DTW_similarity', 'LCS_similarity']
    #methodList = ['cosine_similarity', 'RMS_similarity', 'peak_similarity', 'SSD_similarity', 'LCS_similarity']
    paramsList = ['time-unfiltered-output', 'time-filtered-output', 'alpha-unfiltered-output', 'alpha-filtered-output']

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