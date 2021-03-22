#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'''
predict.py: segmentation prediction on wsi images
'''

import os
import glob
import json

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

from utilities.evaluation import diceCoef, iouScore
from utilities.utils import oneHotToMask
from utilities.augmentation import Augment, Normalize


__author__ = 'Gregory Verghese'
__email__ = 'gregory.verghese@kcl.ac.uk'


class WSIPrediction():
    '''
    class that applies model to predict 
    on large region/entire whole slide image
    '''
    def __init__(self,model,image,mask,step):

        self.model=model
        self.mask=mask
        self.prediction=prediction
        self.step=step

    @property
    def diceScore(self):
        return diceCoef(self.prediction, self.mask)

    @property
    def iouScore(self):
        return iouScore(self.prediction,self.mask)


    def split(self,xStep,yStep):

        _,x,y_=K.int_shape(self.image)
        for y in range(0,y,yStep):
            for x in range(0,x,xStep):
                yield self.image[:,i:i+xStep,j:j+yStep,:],x,y
        

    def predict(self, threshold):
        '''
        wsi prediction with trained model image is split 
        into patches if bigger than self.step threshold.
        '''
        _,x,y,_ = K.int_shape(self.image)
        xStep = self.step if x>self.step else x
        yStep = self.step if y>self.step else y
        
        temp=np.zeros((x,y))

        for p,x,y in self.split(xStep,yStep):
            with tf.device('/cpu:0'):
                probabilities=self.model.predict(p)
                probabilities[y:y+y,x:x+xsize]=p[:,:]
    
        self.prediction=tf.cast((probabilities>self.threshold), tf.float32)
        
        return prediction


    def thresholdTuning(self,thresholds):

        for t in thresholds:
            _ =self.predict(t)
            dice=self.dice
            iou=self.iou

        return dice, iou
        
'''    
def applyNormalization(self, image, mask):
        
    norm=Normalize(self.channelMeans, self.channelStd)
    data=[(image,mask)]
        
    for method in self.normalize:
        f=lambda x: getattr(norm,'get'+method)(x[0],x[1])
        data=list(map(f, data))

    image,mask=data[0][0],data[0][1]

    return image,mask


def getImagePrediction(predict,image,mask,outPath): 

    if tasktype=='multi':
        mask = tf.cast(mask[:,:,:,0], tf.int32)
        mask = tf.one_hot(mask, depth=3, dtype=tf.float32)
    elif tasktype=='binary':
        mask = mask[:,:,:,2:3]
        mask = tf.cast(mask, tf.float32)

    image,mask = predict.applyNormalization(image,mask)            
    prediction = predict.predict(image)
 
    prediction=prediction.numpy().astype(np.uint8)[0,:,:,:]
    mask=mask.numpy().astype(np.uint8)[0,:,:,:]
    
    #TODO:work out dice score
    if self.tasktype=='multi':
        prediction = oneHotToMask(prediction)
            
    outpath = os.path.join(outPath, label +'_pred.png')
    cv2.imwrite(outpath, prediction*int(255))

    return predict


def generatePredictions(model, threshold, tfrecordPath, outPath, step,
                       normalize, channelMeans, channelStd):

    '''
    set up paths and iterate over dataset calling predict.
    save down results for each image in csv
    Args:
        path: string path to
        tfrecordDir: path to testing tfrecord files
        outPath: path to save prediction images down
    Returns:
         avgDice: average of test dice scores across test images
        avgIOU: average of test iou across test images
    '''
         
    predict = WSIPredictions(model,step,threshold,normlize,
                             channelMeans,channelStd)
    
    #TODO: make the outpath directories in main function of ln_segmentation
    records = glob.glob(os.path.join(tfrecordPath, '*'))
        
    dataset = tf.data.TFRecordDataset(record)
    dataset = dataset.map(readTF, num_parallel_calls=1)
    dataset = dataset.batch(1)

    names=[]
    for data in dataset:
        image = tf.cast(data[0], tf.float32)
        mask = data[1]
        label = (data[2].numpy()[0]).decode('utf-8')
        names.append(label)
        predict=self.getTestPredictions(predict)
    
    dices = predict._diceScores
    ious = predict._iouScores
   e imgscores = pd.DataFrame({'image': names, 'dice':dices,'iou':ious})
    imgscores.to_csv(outPath + '_imgscores.csv')
        
    avgDice=predict.modelDice
    avgIou=predict.modelIou
    summary = pd.DataFrame({'dice':[avgDice], 'iou': [avgIou]})
    summary.to_csv(outPath + '_summary.csv')
            
    return avgDice, avgIou
'''
