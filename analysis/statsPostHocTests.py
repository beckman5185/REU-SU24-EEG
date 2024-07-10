from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_scheffe
import numpy as np
import itertools
import scipy.stats
from statVarianceTests import varRuleOfThumb

def tukey(rmanova, coherenceData):

    #perform Tukey test on data
    #dependent variable is coherency, within factor is sound
    tukeyResult = pairwise_tukeyhsd(endog=coherenceData['Coherency'], groups=coherenceData['Sound'], alpha=0.05)

    return tukeyResult


def scheffe(rmanova, coherenceData):

    #perform Scheffe test on data
    #dependent variable is coherency, within factor is sound
    scheffeResult = posthoc_scheffe(coherenceData, val_col='Coherency', group_col='Sound')

    return scheffeResult



def sorted_sounds(soundList, coherenceData):
    #establish indexes important to us
    soundIndex = 3
    coherenceIndex = 4

    #set up sound dict with empty list for each sound
    soundDict = {}
    for sound in soundList:
        soundDict[sound] = []


    #sort coherency data into lists by sound
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
    #doing a test for every combination of sounds
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
    #establish important indexes
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
    #doing a test for each sound
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



def fisher_lsd_test(rmanova, coherenceData):
    #get t critical value for given significance level and degrees of freedom within
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

    #for each combination, find the mean difference
    for pair in combinations:
        sound1, sound2 = pair[0], pair[1]
        difference = np.abs(soundDict[sound1] - soundDict[sound2])

        #print if significant
        if difference > fisher_lsd:
            with open('sample5.txt', "a") as f:
                print ("LSD: " + str(fisher_lsd) + "\nDIFFERENCE: " + str(difference) + "\n", file=f)