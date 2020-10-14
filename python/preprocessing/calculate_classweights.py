#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
calculate_classweights.py: returns weights for each class
by calculating proportion of pixels for each class in the mask
'''

import os
import glob
import argparse

import cv2
import numpy as np
from sklearn.utils import class_weight


def calculateWeights(maskPath, outPath, fileName, numClasses):

    weights = []
    i=0
    masks = glob.glob(os.path.join(maskPath,'*'))
    for f in masks:
        mask = cv2.imread(f)
        labels = mask.reshape(-1)
        classes = np.unique(labels)
        classWeights = class_weight.compute_class_weight('balanced', classes, labels)
    
        weightDict = {c:0 for c in range(numClasses)} 
        

        weightKey = list(zip(classes, classWeights))   
        for k, v in weightKey:
           weightDict[k]=v
    
        values=list(weightDict.values())
        weights.append(list(values))

        i=i+1
        #print(i, classWeights, flush=True)

    finalWeights = list(zip(*weights))
    averageWeights = [np.mean(np.array(w)) for w in finalWeights]
    print(averageWeights, flush=True)

    np.savetxt(fileName+'.csv', averageWeights, delimiter=',')


if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-mp', '--maskpath', required=True, help='path to mask files')
    ap.add_argument('-op', '--outpath', required=True, help='path to save weights')
    ap.add_argument('-sn', '--savename', required=True, help='filename for saving')
    ap.add_argument('-nc', '--number', required=True, help='number of classes')
    args = vars(ap.parse_args())

    maskPath = args['maskpath']
    outPath = args['outpath']
    fileName = args['savename']
    numClasses = int(args['number'])
        
    calculateWeights(maskPath, outPath, fileName, numClasses)
