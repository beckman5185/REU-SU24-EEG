import pingouin as pg
import pandas as pd
import os
import scipy.stats
import itertools
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statistics

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



def tukey(coherenceData, style, file, method):
    # get mixed ANOVA for sounds and gender
    rmanova = pg.mixed_anova(coherenceData, dv='Coherency', within='Sound', subject='Subject', between='Gender', correction=True)

    soundIndex = 1
    interactionIndex = 2
    sphericity = rmanova['sphericity'][soundIndex] == True
    significance = rmanova['p-unc'][interactionIndex] <= 0.05
    # significance2 = rmanova['p-unc'][interactionIndex] <= 0.05
    tukey = None

    if sphericity and significance:
        tukey = pairwise_tukeyhsd(endog=coherenceData['Coherency'], groups=coherenceData['Sound'], alpha=0.05)

    return rmanova, tukey


def varRuleOfThumb(data1, data2):
    variance1 = statistics.variance(data1)
    variance2 = statistics.variance(data2)

    #if ratio of larger variance to smaller variance < 4, then approx. equal variances
    if max([variance1, variance2]) / min([variance1, variance2]) < 4:
        equalVar = True
    else:
        equalVar = False

    return equalVar



def t_test_bonferroni(coherenceData):
    soundIndex = 3
    coherenceIndex = 4

    #initialize dict to store lists of coherency values sorted by sound
    soundList = ['N2', 'N3', 'N4', 'N5', 'N6', 'N8']
    soundDict = {}
    for sound in soundList:
        soundDict[sound] = []


    #sort coherency data by sound
    for i in range (0, len(coherenceData.index)):
        soundName = coherenceData.iloc[i, soundIndex]
        coherence = coherenceData.iloc[i, coherenceIndex]

        soundDict[soundName].append(coherence)


    #find every combination of two sounds
    combinations = list(itertools.combinations(soundList, 2))

    #Bonferroni correction for alpha value
    alpha = 0.05
    corrected_alpha = alpha / len(combinations)

    #for each pair of sounds, perform a two-tailed two-sample t test with corrected alpha
    for pair in combinations:
        #get list of coherence values for given sound
        sound1, sound2 = soundDict[pair[0]], soundDict[pair[1]]

        #equal variances determined by variance rule of thumb
        equalVar = varRuleOfThumb(sound1, sound2)

        #calculate t statistic and p value
        t_statistic, p_value = scipy.stats.ttest_ind(sound1, sound2, equal_var=equalVar)

        #evaluate significance with corrected alpha
        significance = False
        if (p_value < corrected_alpha):
            significance = True
            print("SIGNIFICANT RESULT:")
            print("Pair: " + str(pair))
            print("P value: " + str(p_value))









def printResults(filename, style, channels, method, results):
    with open(filename, "a") as f:
        print("Parameters: " + style, file=f)
        print("Channel pair: " + channels, file=f)
        print("Method of analysis: " + method, file=f)
        print(results, file=f)
        print(file=f)



def main():
    paramsList = ['time-unfiltered-output', 'time-filtered-output', 'alpha-unfiltered-output',
                  'alpha-filtered-output', 'gamma-unfiltered-output', 'gamma-filtered-output']

    #for each parameter combination
    for style in paramsList:

        #no DTW in frequency domain
        if style == 'time-unfiltered-output' or style == 'time-filtered-output':
            methodList = ['cosine_similarity', 'RMS_similarity', 'peak_similarity', 'SSD_similarity', 'DTW_similarity',
                          'LCS_similarity']
        else:
            methodList = ['cosine_similarity', 'RMS_similarity', 'peak_similarity', 'SSD_similarity', 'LCS_similarity']


        #for each method
        for i in range(len(methodList)):
            directory = style + "/" + methodList[i]

            #for each channel pair
            for file in os.listdir(directory):

                #read in coherency data table and perform mixed ANOVA
                #between = gender, within = sounds
                coherenceData = pd.read_csv(directory + "/" + file, header=0)

                #check that variance of between groups is similar
                #leveneResult = leveneTest(coherenceData)

                #perform Tukey test
                #rmanova, tukey = tukey(coherenceData)

                t_test_bonferroni(coherenceData)

                #filename = "sample4.txt"
                #results = rmanova.to_string() + "\n" + "TUKEY:" + "\n" + str(tukey)
                #printResults(filename, style, file, methodList[i], results):






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