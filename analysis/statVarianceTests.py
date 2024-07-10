import scipy.stats
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



def varRuleOfThumb(data1, data2):
    #get variance of both data series
    variance1 = statistics.variance(data1)
    variance2 = statistics.variance(data2)

    #use rule of thumb to determine equality of variances
    #rule of thumb: ratio of large varaince to small variance no greater than 4
    if max([variance1, variance2]) / min([variance1, variance2]) < 4:
        equalVar = True
    else:
        equalVar = False

    return equalVar
