# Copyright 2022 Ray Gardner (rdg, raygard)
# License: 0BSD

# Compute longest common subsequence (LCS) using Hunt and Szymanski's method
#
# See "A Fast Algorithm for Computing Longest Common Subsequences",
# J.W. Hunt and T.G. Szymanski, CACM vol 20 no. 5 (May 1977), p350-353

# Use bisect module to get fast C binary search implementation.
import bisect
import pandas as pd
import scipy.fft
import numpy as np


#Audrey change: added error as a parameter
def LCS(A, B, error):
    m, n = (len(A), len(B))


    # Step 1: build linked lists
    matchlist = [[] for k in range(m + 1)]
    # Note line numbers in reverse order
    aa = sorted(zip(A, range(1, m+1)), key=lambda t: (t[0], -t[1]))
    bb = sorted(zip(B, range(1, n+1)), key=lambda t: (t[0], -t[1]))
    ai = bi = 0
    while ai < m and bi < n:
        av, bv = aa[ai][0], bb[bi][0]
        #Audrey change: accounting for error bound instead of exact match
        if (av + error) < bv:
            ai += 1
        elif (av - error) > bv:
            bi += 1
        else:
            k = aa[ai][1]
            while bi < n and bb[bi][0] == bv:
                matchlist[k] += [bb[bi][1]]
                bi += 1
            ai += 1
            while ai < m and aa[ai][0] == av:
                matchlist[aa[ai][1]] = matchlist[k]
                ai += 1

    # Step 2: initialize the THRESH array
    thresh = [n+1] * (m + 1)
    thresh[0] = 0

    # Step 3: compute successive THRESH values
    link = [None] * (m+1)
    for i in range(1, m+1):
        for j in matchlist[i]:
            #find k such that thresh[k-1] < j <= thresh[k]
            k = bisect.bisect_left(thresh, j)
            #assert thresh[k-1] < j <= thresh[k]
            if j < thresh[k]:
                thresh[k] = j
                link[k] = (i, j, link[k-1])
                #print(f'dmatch({i}, {j})')
                #assert A[i-1] == B[j-1]

    # Step 4: recover longest common subsequence pairs in reverse order
    k = 0
    while k < m and thresh[k+1] != n + 1:
        k += 1
    p = link[k]
    # v will hold (i,j) pairs
    v = []
    while p != None:
        v.append(p[:2])
        p = p[2]
    v.reverse()

    #print(f'lcslen: {len(v)=}')
    #return v

    #print(v)
    # Return actual LCS
    # Do this if actually recovering the LCS itself.
    for k in range(len(v)):
        v[k] = A[v[k][0]-1]
    return v



def main():
    directory = r"Nature Raw Txt"
    EEGdata = pd.read_csv(directory + "/" + "Ball2_Nature_EEGData_fl10_N2.txt", header=None)

    Ch1 = EEGdata[0]
    Ch9 = EEGdata[8]

    freqCh1 = scipy.fft.fft(Ch1.values)
    freqCh9 = scipy.fft.fft(Ch9.values)

    #getting average of alpha frequency band
    low = 8*60
    high = 13*60
    sum = 0
    for i in range (low, high):
        sum += freqCh1[i]

    #getting error from mean of alpha frequency band
    mean = sum/len(freqCh1)
    error = mean/10

    lcs = LCS(Ch1, Ch9, error)
    freqLCS = LCS(freqCh1, freqCh9, error)

    print("Time domain LCS: " + str(lcs))
    print("Time domain LCS length: " + str(len(lcs)))
    print("Time domain LCS relative length: " + str(len(lcs)/len(Ch1)))
    print("Frequency domain LCS: " + str(freqLCS))
    print("Frequency domain LCS length: " + str(len(freqLCS)))
    print("Frequency domain LCS relative length: " + str(len(freqLCS)/len(freqCh1)))



if __name__ == "__main__":
    main()