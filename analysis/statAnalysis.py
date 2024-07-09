import pingouin as pg
import pandas as pd
import os
import scipy.stats
import itertools
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statistics
from scikit_posthocs import posthoc_scheffe
import numpy as np

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



def tukey(rmanova, coherenceData):
    tukey = pairwise_tukeyhsd(endog=coherenceData['Coherency'], groups=coherenceData['Sound'], alpha=0.05)

    return tukey


def scheffe(rmanova, coherenceData):

    scheffe = posthoc_scheffe(coherenceData, val_col='Coherency', group_col='Sound')

    return scheffe


def varRuleOfThumb(data1, data2):
    variance1 = statistics.variance(data1)
    variance2 = statistics.variance(data2)

    #if ratio of larger variance to smaller variance < 4, then approx. equal variances
    if max([variance1, variance2]) / min([variance1, variance2]) < 4:
        equalVar = True
    else:
        equalVar = False

    return equalVar


def sorted_sounds(soundList, coherenceData):
    #establish indexes important to us
    soundIndex = 3
    coherenceIndex = 4

    #set up sound dict with empty list for each sound
    soundDict = {}
    for sound in soundList:
        soundDict[sound] = []


    #sort coherency data by sound
    for i in range (0, len(coherenceData.index)):
        soundName = coherenceData.iloc[i, soundIndex]
        coherence = coherenceData.iloc[i, coherenceIndex]

        soundDict[soundName].append(coherence)

    return soundDict


def t_test_sound(coherenceData):

    #initialize dict to store lists of coherency values sorted by sound
    soundList = ['N2', 'N3', 'N4', 'N5', 'N6', 'N8']
    soundDict = sorted_sounds(coherenceData, soundList)


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
            print("SIGNIFICANT RESULT (SOUND):")
            print("Pair: " + str(pair))
            print("P value: " + str(p_value))




def t_test_gender(coherenceData):
    genderIndex = 2
    soundIndex = 3
    coherenceIndex = 4

    #initialize dict to store coherency values sorted by sound and gender
    soundList = ['N2', 'N3', 'N4', 'N5', 'N6', 'N8']
    soundDict = {}
    for sound in soundList:
        soundDict[sound] = {'F':[], 'M':[]}


    #sort coherency data by sound and gender
    for i in range (0, len(coherenceData.index)):
        soundName = coherenceData.iloc[i, soundIndex]
        coherence = coherenceData.iloc[i, coherenceIndex]
        gender = coherenceData.iloc[i, genderIndex]

        if gender in soundDict[soundName].keys():
            soundDict[soundName][gender].append(coherence)


    #Bonferroni correction for alpha value
    alpha = 0.05
    corrected_alpha = alpha / len(soundList)

    #for each sound, perform a two-tailed two-sample t test with corrected alpha
    for sound in soundList:
        #get list of coherence values for given sound
        Fsound, Msound = soundDict[sound]['F'], soundDict[sound]['M']

        #equal variances determined by variance rule of thumb
        equalVar = varRuleOfThumb(Fsound, Msound)

        #calculate t statistic and p value
        t_statistic, p_value = scipy.stats.ttest_ind(Fsound, Msound, equal_var=equalVar)

        #evaluate significance with corrected alpha
        significance = False
        if (p_value < corrected_alpha):
            significance = True
            print("SIGNIFICANT RESULT (GENDER):")
            print("Sound: " + sound)
            print("P value: " + str(p_value))



def significance(rmanova):
    soundIndex = 1
    interactionIndex = 2
    sphericity = rmanova['sphericity'][soundIndex] == True
    significance = rmanova['p-unc'][soundIndex] <= 0.05
    significance2 = rmanova['p-unc'][interactionIndex] <= 0.05

    return sphericity and (significance or significance2)



def fisher_lsd_test(rmanova, coherenceData):
    #get t critical value for significance level and degrees of freedom within
    alpha = 0.05
    soundIndex = 1
    df_within = rmanova.loc[soundIndex, 'DF2']
    t_val = scipy.stats.t.ppf(1 - alpha/2, df_within)

    #get mean square within
    msw = rmanova.loc[soundIndex, 'MS']


    #get number of samples in each group
    numSounds = 6
    n1 = len(coherenceData.index) / numSounds

    #get fisher least significant difference
    fisher_lsd = t_val * np.sqrt(msw * (n1 + n1))


    #initialize dict to store lists of coherency values sorted by sound
    soundList = ['N2', 'N3', 'N4', 'N5', 'N6', 'N8']
    soundDict = sorted_sounds(soundList, coherenceData)

    #get mean of coherency values for each sound
    for sound in soundDict.keys():
        group_mean = np.mean(soundDict[sound])
        soundDict[sound] = group_mean

    #find every combination of two sounds
    combinations = list(itertools.combinations(soundList, 2))

    #for each combination, find the mean difference and print if significant
    for pair in combinations:
        sound1, sound2 = pair[0], pair[1]
        difference = np.abs(soundDict[sound1] - soundDict[sound2])

        if difference > fisher_lsd:
            with open('sample5.txt', "a") as f:
                print ("LSD: " + str(fisher_lsd) + "\nDIFFERENCE: " + str(difference) + "\n", file=f)













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
                leveneResult = leveneTest(coherenceData)

                #perform Tukey test
                tukeyResult, scheffeResult = None, None
                significant = False
                if leveneResult:
                    # get mixed ANOVA for sounds and gender
                    rmanova = pg.mixed_anova(coherenceData, dv='Coherency', within='Sound', subject='Subject',
                                             between='Gender', correction=True)

                    significant = significance(rmanova)

                    if (significant):
                        #tukeyResult = tukey(rmanova, coherenceData)
                        #scheffeResult = scheffe(rmanova, coherenceData)
                        fisher_lsd_test(rmanova, coherenceData)

                        filename = "sample4.txt"
                        #results = rmanova.to_string() + "\n" + "TUKEY:" + "\n" + str(tukeyResult) + "\n" + "SCHEFFE: " + str(scheffeResult)
                        #printResults(filename, style, file, methodList[i], results)





                #significant results for gender
                #no signficant results for sound
                #t_test_sound(coherenceData)
                #t_test_gender(coherenceData)



                #Fisher LSD test










if __name__ == "__main__":
    main()