#!/usr/bin/env python3
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

    total = {c:0 for c in range(numClasses)}
    i=0

    masks = glob.glob(os.path.join(maskPath,'*'))
    for f in masks:
        mask = cv2.imread(f)
        labels = mask.reshape(-1)
        classes = np.unique(labels, return_counts=True)

        pixelDict = dict(list(zip(*classes))) 
    
        for k, v in pixelDict.items():
            total[k] = total[k] + v
        print(i)
        i+=1

    print(total)
    if numClasses==2:
        weight = total[0]/total[1]
    else:
        weight = [1/v for v in list(total.values())]
        

    #np.savetxt(fileName+'.csv', averageWeights, delimiter=',')
    print(weight)

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
