import pandas as pd
import numpy as np
import seaborn as sb
import os
import matplotlib.pyplot as plt
from coherencyTest5 import cross_correlation_similarity, getGender
from statsPostHocTests import t_test_sound

def generateCrossCorrelationData():
    # looking into folder with data
    directory = r"../Nature Raw Txt"

    channels = list(range(0, 16))
    channelNames = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']
    listMap = pd.DataFrame(columns=channelNames, index=channelNames)
    valMap = pd.DataFrame(columns=['First Channel', 'Second Channel', 'Cross-Correlation'])

    for i in channels:
        for j in channels:
            listMap.iloc[i, j] = []

    for file in os.listdir(directory):
        # get sound and last name and gender of participant
        soundName = file.split('_')[-1].strip('.txt')
        lastName = file.split('_')[0]
        gender = getGender(lastName)

        # read in EEG data
        EEGdata = pd.read_csv(directory + "/" + file, header=None)
        EEGdata = EEGdata.drop(columns=[16], axis=1)

        for i in channels:
            for j in channels:
                sim_val = cross_correlation_similarity(EEGdata[i], EEGdata[j], True)
                listMap.iloc[i, j].append(sim_val)

    for i in channels:
        for j in channels:
            list_vals = listMap.iloc[i, j]
            new_row = pd.DataFrame([[channelNames[i], channelNames[j], np.mean(list_vals)]], columns=valMap.columns)
            valMap = pd.concat([valMap, new_row], ignore_index=True)

    valMap.to_csv("heatMap-cross-correlation-all.csv")


def generateData(domain, method):
    directory = r"" + domain + "-filtered-output//" + method + "//"

    megaDataFrame = pd.DataFrame(columns=['Channel Pair', 'Sound', 'Coherency'])

    for file in os.listdir(directory):
        channelPair = str(file).strip(".csv")

        dataFrame = pd.read_csv(directory + file)

        soundSeries = ['N2', 'N3', 'N4', 'N5', 'N6', 'N8']
        for sound in soundSeries:
            soundFrame = dataFrame[dataFrame['Sound'] == sound]
            coherency = np.mean(soundFrame['Coherency'])

            new_row = pd.DataFrame([[channelPair, sound, coherency]], columns=megaDataFrame.columns)
            megaDataFrame = pd.concat([megaDataFrame, new_row], ignore_index=True)



    heatMapFrame = megaDataFrame.pivot(index='Channel Pair', columns='Sound', values='Coherency')

    sb.set(font_scale=1.25)
    final_heatmap = sb.heatmap(heatMapFrame, cmap='YlGn')
    plt.title("Coherency: " + " " + method)
    plt.tight_layout()
    plt.show()


def generatePostHoc(domain, method):

    # for each method

    directory = domain + "-filtered-output/" + method

    megaDataFrame = pd.DataFrame()


    # for each channel pair
    for file in os.listdir(directory):

        # read in coherency data table and perform mixed ANOVA
        # between = gender, within = sounds
        coherenceData = pd.read_csv(directory + "/" + file, header=0)

        # check that variance of between groups is similar
        # leveneResult = leveneTest(coherenceData)

        dataFrame = t_test_sound(coherenceData)

        print(dataFrame)

        channelPair = str(file).strip(".csv")

        dataFrame['Channel Pair'] = [channelPair] * len(dataFrame.index)
        dataFrame['Sound Pair'] = dataFrame.index

        megaDataFrame = pd.concat([megaDataFrame, dataFrame], ignore_index=True)



    heatMapFrame = megaDataFrame.pivot(index='Sound Pair', columns='Channel Pair', values='P Value')

    #https://stackoverflow.com/questions/66099438/how-to-annot-only-values-greater-than-x-on-a-seaborn-heatmap

    # Generate annotation labels array (of the same size as the heatmap data)- filling cells you don't want to annotate with an empty string ''
    annot_labels = np.empty_like(heatMapFrame, dtype=str)
    annot_mask = heatMapFrame < (0.05/15)
    annot_labels[annot_mask] = "S"

    sb.set(font_scale=1.25)
    final_heatmap = sb.heatmap(heatMapFrame, annot=annot_labels, cmap='GnBu_r', fmt="", vmax=0.05)
    plt.title("T Test: " + method)
    plt.tight_layout()
    plt.show()





def heatMap():
    valMap = pd.read_csv('heatMap-cross-correlation-all.csv')


    valMap = valMap.pivot(index='First Channel', columns='Second Channel', values='Cross-Correlation')


    #from https://stackoverflow.com/questions/2318529/plotting-only-upper-lower-triangle-of-a-heatmap#:~:text=The%20key%20for%20solution%20is%20cm.set_bad%20function.%20You,set_bad%20to%20white%2C%20instead%20of%20the%20default%20black.

    #set up mask for upper triangle and diagonal
    mask = np.zeros_like(valMap, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    mask[np.diag_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the mask
    sb.set(font_scale=2)
    cross_corr_heatmap = sb.heatmap(valMap, mask=mask, cmap='YlGn')
    cross_corr_heatmap.set_xticklabels(cross_corr_heatmap.get_xmajorticklabels(), fontsize=20)
    cross_corr_heatmap.set_yticklabels(cross_corr_heatmap.get_ymajorticklabels(), fontsize=20)
    ax.set_xlabel(ax.get_xlabel(), fontsize=20)
    ax.set_ylabel(ax.get_ylabel(), fontsize=20)
    plt.title("Cross-Correlation for All Channel Pairs")
    plt.tight_layout()
    plt.show()

def main():
    #generateCrossCorrelationData()

    #domains and methods we are generating heat maps for
    paramPairs = [('time', 'cross_correlation_similarity'), ('alpha', 'cosine_similarity'), ('time', 'coherence_similarity'), ('time', 'i_coherence_similarity')] #, ('alpha', 'LCS_similarity')]

    for i in range (0, len(paramPairs)):
        current = paramPairs[i]

        #generateData(current[0], current[1])
        #generatePostHoc(current[0], current[1])

    heatMap()



if __name__ == "__main__":
    main()