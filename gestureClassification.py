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
#from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GMM

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

gestures = set()
classData = np.zeros(numFiles)
featureData = np.zeros((numFiles, numTimeSamps, numChnls))

for fileNum in range(numFiles):
    classData[fileNum] = int(fileNms[fileNum][7:fileNms[fileNum].index('_')])
    gestures.add(classData[fileNum])
    
    openFile = open(dataPath + fileNms[fileNum], 'r')
    fileReader = csv.reader(openFile, delimiter = ',')
    
    timeNum = 0
    for fileLine in fileReader:
        featureData[fileNum][timeNum] = fileLine
        
        timeNum += 1
    
    openFile.close()

numGestures = len(gestures)

#Normalize???
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

finalFeatures = np.hstack((featureEarlyMeans, featureMidMeans, featureLateMeans, featureEarlyVars, featureMidVars, featureLateVars))

featuresTrain = np.zeros((0, np.shape(finalFeatures)[1]))
featuresTest = np.zeros((0, np.shape(finalFeatures)[1]))
classTrain = np.zeros(0)
classTest = np.zeros(0)

for gestureNum in gestures:
    featuresTrainGest, featuresTestGest, classTrainGest, classTestGest = train_test_split(finalFeatures[classData == gestureNum, :], classData[classData == gestureNum], test_size=0.2)
    featuresTrain = np.vstack((featuresTrain, featuresTrainGest))
    featuresTest = np.vstack((featuresTest, featuresTestGest))
    classTrain = np.hstack((classTrain, classTrainGest))
    classTest = np.hstack((classTest, classTestGest))

#Randomize order of observations!!
#More preprocessing??
min_max_scaler = preprocessing.MinMaxScaler()
featuresTrain = min_max_scaler.fit_transform(featuresTrain)
featuresTest = min_max_scaler.transform(featuresTest)

#classSVM = OneVsRestClassifier(SVC(kernel='linear'))
#classSVM.fit(featuresTrain, classTrain)

#classDT = RandomForestClassifier(min_samples_leaf=5)
#classDT.fit(featuresTrain, classTrain)
#classDT.score(featuresTest, classTest)

#classKNN = KNeighborsClassifier(n_neighbors=3)
#classKNN.fit(featuresTrain, classTrain)
#classKNN.score(featuresTest, classTest)

#classLogReg = LogisticRegression()#penalty='newton-cg', dual=False)#, solver='liblinear')
#classLogReg.fit(featuresTrain, classTrain)
#classLogReg.score(featuresTest, classTest)

classGMM = GMM(n_components=numGestures, covariance_type='tied', init_params='wc', n_iter=100)
classGMM.means_ = np.array([featuresTrain[classTrain == i].mean(axis=0) for i in gestures])
classGMM.fit(featuresTrain)
#classGMM.fit(featuresTrain, classTrain)
#print classGMM.score(featuresTest, classTest)