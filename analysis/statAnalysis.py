import pingouin as pg
import pandas as pd
import numpy as np
import os
from statVarianceTests import leveneTest
from statsPostHocTests import tukey, scheffe, t_test_sound, t_test_gender, fisher_lsd_test


def mixedSignificance(leveneResult, rmanova):
    #establish important indexes
    soundIndex = 1
    interactionIndex = 2

    #get validity (sphericity and Levene test assumptions of equal varainces valid)
    sphericity = rmanova['sphericity'][soundIndex] == True
    validity = leveneResult and sphericity

    #get significance (p value less than 0.05)
    significance = rmanova['p-unc'][soundIndex] <= 0.05
    significance2 = rmanova['p-unc'][interactionIndex] <= 0.05

    #significant if valid and sound/interaction has significant p value
    return validity and (significance or significance2)



def oneWaySignificance(rmanova):
    #establish important index
    index = 0

    #get sphericity of variances
    sphericity = rmanova['sphericity'][index] == True

    #get significance
    significance = rmanova['p-unc'][index] <= 0.05

    #significant if sphericity preserved and sound has significant p value
    return sphericity and significance


def runTests(significant, rmanova, coherenceData, oneWay):
    resultsString = ""

    #if (significant):
        #tukeyResult = tukey(coherenceData)
        #scheffeResult = scheffe(coherenceData)
        #fisherResult = fisher_lsd_test(rmanova, coherenceData, oneWay)

        #resultsString += "TUKEY: \n" + str(tukeyResult) + "\nSCHEFFE: \n" + str(scheffeResult) + "\nFISHER: \n" + fisherResult.to_string()

    if (not oneWay):
        t_test_sound_results = t_test_sound(coherenceData)
        resultsString += "\nT TEST (SOUND): \n" + t_test_sound_results.to_string()

    #if not oneWay:
    #    t_test_gender_results = t_test_gender(coherenceData)
    #    resultsString += "\nT TEST (GENDER): \n" + t_test_gender_results.to_string()

    return resultsString





def doOneWayANOVA(coherenceData):
    #establish important indexes
    genderIndex = 2
    coherenceIndex = 4

    #initialize dict to store coherency values sorted by gender
    FCoherenceData = pd.DataFrame()
    MCoherenceData = pd.DataFrame()


    #sort coherency data by gender
    for i in range (0, len(coherenceData.index)):

        gender = coherenceData.iloc[i, genderIndex]

        if gender == 'F':
            FCoherenceData = FCoherenceData._append(coherenceData.iloc[i], ignore_index=True)
        elif gender == 'M':
            MCoherenceData = MCoherenceData._append(coherenceData.iloc[i], ignore_index=True)

    #perform one way repeated measures anova on both sets of data
    #note: ANOVA generation automatically performs sphericity test
    F_rmanova = pg.rm_anova(FCoherenceData, dv='Coherency', within='Sound', subject='Subject', correction=True)
    M_rmanova = pg.rm_anova(MCoherenceData, dv='Coherency', within='Sound', subject='Subject', correction=True)

    return (FCoherenceData, F_rmanova, MCoherenceData, M_rmanova)

def printOneWayANOVA(F_rmanova, M_rmanova, style, channels, method):


    F_results = "ONE-WAY RM ANOVA: \n" + F_rmanova.to_string()
    M_results = "ONE-WAY RM ANOVA: \n" + M_rmanova.to_string()

    directory3 = r"significance-results" + "\\" + style + "\\"
    filename = directory3 + style + "-" + method + "-F-" + "one-way-anova.txt"
    printResults(filename, style, channels, method, F_results)
    filename = directory3 + style + "-" + method + "-M-" + "one-way-anova.txt"
    printResults(filename, style, channels, method, M_results)


def doMixedANOVA(coherenceData):
    #do Levene test to ensure variances are equal between genders
    leveneResult = leveneTest(coherenceData)

    #do mixed repeated measures ANOVA
    #note: ANOVA generation automatically performs sphericity test
    rmanova = pg.mixed_anova(coherenceData, dv='Coherency', within='Sound', subject='Subject',
                             between='Gender', correction=True)

    return (leveneResult, rmanova)



def printMixedANOVA(leveneResult, rmanova, style, channels, method):

    results = "LEVENE TEST: " + str(leveneResult) + "\nMIXED ANOVA: \n" + rmanova.to_string()
    directory3 = r"significance-results" + "\\" +  style + "\\"
    filename = directory3 + style + "-" + method + "-" + "rmanova.txt"
    printResults(filename, style, channels, method, results)





def printResults(filename, style, channels, method, results):
    with open(filename, "a") as f:
        print("Parameters: " + style, file=f)
        print("Channel pair: " + channels, file=f)
        print("Method of analysis: " + method, file=f)
        print(results, file=f)
        print(file=f)






def main():
    #paramsList = ['time-unfiltered-output', 'time-filtered-output', 'alpha-unfiltered-output',
    #              'alpha-filtered-output', 'gamma-unfiltered-output', 'gamma-filtered-output']

    paramsList = ['time-filtered-output', 'alpha-filtered-output', 'gamma-filtered-output', 'full-filtered-output']

    #paramsList = ['time-filtered-output']

    #for each parameter combination
    for style in paramsList:

        #no DTW in frequency domain
        if style == 'time-filtered-output': #or style == 'time-unfiltered-output':
            methodList = ['cosine_similarity', 'RMS_similarity', 'peak_similarity', 'SSD_similarity', 'DTW_similarity',
                          'LCS_similarity']
        else:
            methodList = ['cosine_similarity', 'RMS_similarity', 'peak_similarity', 'SSD_similarity', 'LCS_similarity']

        #methodList = ['LCS_similarity']

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






                leveneResult, rmanova = doMixedANOVA(coherenceData)
                #printMixedANOVA(leveneResult, rmanova, style, file, methodList[i])
                FCoherenceData, F_rmanova, MCoherenceData, M_rmanova = doOneWayANOVA(coherenceData)
                #printOneWayANOVA(F_rmanova, M_rmanova, style, file, methodList[i])


                #mixed ANOVA and one way ANOVA significant if assumptions about variance valid and p value significant
                mixedSignificant = mixedSignificance(leveneResult, rmanova)
                F_significant = oneWaySignificance(F_rmanova)
                M_significant = oneWaySignificance(M_rmanova)



                resultsString1 = runTests(mixedSignificant, rmanova, coherenceData, False)
                resultsString2 = runTests(F_significant, F_rmanova, FCoherenceData, True)
                resultsString3 = runTests(M_significant, M_rmanova, MCoherenceData, True)

                print(resultsString1)
                print(resultsString2)
                print(resultsString3)

                directory4 = r"significance-results" + "\\" + style + "\\"

                if not os.path.exists(directory4):
                    os.makedirs(directory4)



                #filename = style + "-" + methodList[i]
                #printResults(directory4 + filename + "-mixed-posthoc.txt", style, file, methodList[i], resultsString1)
                #printResults(directory4 + filename + "-F-posthoc.txt", style, file, methodList[i], resultsString2)
                #printResults(directory4 + filename + "-M-posthoc.txt", style, file, methodList[i], resultsString3)







'''
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
                    '''





                #significant results for gender
                #no signficant results for sound
                #t_test_sound(coherenceData)
                #t_test_gender(coherenceData)



                #Fisher LSD test










if __name__ == "__main__":
    main()