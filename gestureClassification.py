# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 10:51:35 2017

@author: amwytsma
"""
from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

dataPath = "/u5/amwytsma/Documents/ThalmicChallenge/GestureData/"

fileNms = [dataFile for dataFile in listdir(dataPath) if isfile(join(dataPath, dataFile))]

numFiles = len(fileNms)

openFile = open(dataPath + fileNms[0], 'r')
fileReader = csv.reader(openFile, delimiter = ',')

numTimeSamps = 0
for fileLine in fileReader:
    numChnls = len(fileLine)
    numTimeSamps += 1

openFile.close()

classData = np.zeros(numFiles)
featureData = np.zeros((numFiles, numTimeSamps, numChnls))

for fileNum in range(numFiles):
    classData[fileNum] = int(fileNms[fileNum][7:fileNms[fileNum].index('_')])
    
    openFile = open(dataPath + fileNms[fileNum], 'r')
    fileReader = csv.reader(openFile, delimiter = ',')
    
    timeNum = 0
    for fileLine in fileReader:
        featureData[fileNum][timeNum] = fileLine
        
        timeNum += 1
    
    openFile.close()

featureData = featureData.astype(int)

featureMeans = np.mean(featureData, axis = 1)
featureEarlyMeans = np.mean(featureData[:, 0:int(numTimeSamps/3), :], axis = 1)
featureMidMeans = np.mean(featureData[:, int(numTimeSamps/3):int(2*numTimeSamps/3), :], axis = 1)
featureLateMeans = np.mean(featureData[:, int(2*numTimeSamps/3):numTimeSamps, :], axis = 1)

featureVars = np.var(featureData, axis = 1)
featureEarlyVars = np.var(featureData[:, 0:int(numTimeSamps/3), :], axis = 1)
featureMidVars = np.var(featureData[:, int(numTimeSamps/3):int(2*numTimeSamps/3), :], axis = 1)
featureLateVars = np.var(featureData[:, int(2*numTimeSamps/3):numTimeSamps, :], axis = 1)

#Also covariance???
#Normalize???

generatedFeatures = np.hstack((featureMeans, featureVars))

classif = OneVsRestClassifier(SVC(kernel='linear'))
#classif.fit(X, Y)

"""
for gestureNum in range(1, 7):
    gestureObs = featureData[classData == gestureNum]
    
    print np.mean(gestureObs, axis = (0, 1))
"""