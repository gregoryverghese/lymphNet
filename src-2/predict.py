#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'''
predict.py: segmentation prediction on 10 lymph nodes
selected for testing
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


def readTF(serialized):
    '''
    read tfrecord files (.tfrecords) and converts
    serialized image and mask data into 4d-tensors
    Args:
        serialized: tfrecord file
    Returns:
        image: tensorflow 4d-tensor 
        mask : tensorfow 4d-tensor groundtruth
        label: string name
    '''

    dataMap = {'image': tf.io.FixedLenFeature((), tf.string),
               'mask': tf.io.FixedLenFeature((), tf.string),
               'xDim': tf.io.FixedLenFeature((), tf.int64),
               'yDim': tf.io.FixedLenFeature((), tf.int64), 
               'label': tf.io.FixedLenFeature((), tf.string)}
    
    example = tf.io.parse_single_example(serialized, dataMap)
    image = tf.image.decode_png(example['image'])
    mask = tf.image.decode_png(example['mask'])
    xDim = example['xDim']
    yDim = example['yDim']
    label = example['label']

    print('xDim: {}, yDim:{}'.format(xDim, yDim))

    image = tf.reshape(image, (xDim, yDim, 3))
    mask = tf.reshape(mask, (xDim, yDim, 3))
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)

    return image, mask, label


#ToDo: merge this class into WSIPredict class
class PatchPredictions(object):
    '''
    class that applies trained model to predict
    the class that each pixel belongs to. Calculates
    dice/iou score for each patch and saves down 
    average across test dataset
    '''

    def __init__(self, model, modelName, batchSize, currentTime, 
                 currentDate, threshold):

        self.model = model
        self.modelName = modelName
        self.batchSize = batchSize
        self.currentTime = currentTime
        self.currentDate = currentDate
        self.threshold = threshold


    @tf.function
    def testStep(self, x):
        logits = self.model(x, training=False)
        return logits


    def __call__(self, testdataset, outPath):
 
        path=os.path.join(outPath, 'patches', self.currentDate)

        try:
            os.mkdir(path)
        except Exception as e:
            print(e)
        
        path=os.path.join(path, self.modelName + '_' +
                          self.currentTime)
        try:
            os.mkdir(path)    
        except Exception as e:
            print(e)

        diceLst=[]
        iouLst=[]

        for i, data in enumerate(testdataset):
            image = tf.cast(data[0], tf.float32)
            mask = tf.cast(data[1], tf.float32)
            
            #predicts on 4d image and calculates class based on 
            #activation threshold
            logits = self.testStep(image)
            prediction = tf.cast(logits > self.threshold, tf.float32)

            dice = diceCoef(mask, prediction)
            diceLst.append(dice.numpy())

            iou = iouScore(mask, prediction)
            iouLst.append(iou.numpy())
            
            image = tf.cast(image, tf.uint8)
            fig, axs = plt.subplots(1, 3, figsize=(5, 5))
            axs[0].imshow(image[0,:,:,:], cmap='gray')
            axs[1].imshow(mask[0,:,:,0]*255, cmap='gray')
            axs[2].imshow(prediction[0,:,:,0]*255, cmap='gray')
            fig.savefig(os.path.join(path, str(i) + '.png'))
            plt.close()

        avgDice = np.mean(np.array(diceLst))
        avgIOU = np.mean(np.array(iouLst))
        print('Avg dice: {} \n Avg iou {}'.format(avgDice, avgIOU))
        
        #saves down dice/iou for each patch and avg across testadataset
        imgscores = pd.DataFrame({'dice':diceLst, 'iou':iouLst})
        imgscores.to_csv(os.path.join(path, self.modelName+'_imgscores.csv'))
        summary = pd.DataFrame({'dice':[avgDice], 'iou': [avgIOU]})
        summary.to_csv(os.path.join(path, self.modelName+'_summary.csv'))



class WSIPredictions(object):
    '''
    class that applies model to predict on large region/entire whople slide
    image
    '''
    resolutionDict = {'2.5x':16,'5x':8,'10x':4}

    def __init__(self, model, modelName, feature, magnification, imgDims, 
                 step, threshold, currentTime, currentDate, tasktype, 
                 normalize, normalizeParams, figureSize=500):

        self.model = model 
        self.modelName = modelName
        self.feature = feature
        self.imgDims = imgDims
        self.figureSize = figureSize
        self.magnification = magnification
        self.currentDate = currentDate
        self.currentTime = currentTime
        self.threshold = threshold,
        self.step = step
        self.tasktype = tasktype
        self.normalize = normalize
        self.normalizeParams = normalizeParams

    
    def predict(self, image, mask):
        '''
        takes a image and predicts a class for each pixel using 
        trained self.model. If the image is large than certain 
        threshold self.step we split the image into smaller regions 
        and predict on each one and then stitch back together.
        Args:
            image: 4d-tensor image
            mask: 4d-tensor groundtruth mask
            label: string name
            outPath: path to save down predictions
        Returns:
            dice: float dice score
            iou: float iou score
        '''
        _,x,y,_ = K.int_shape(image)
        #xStep = self.step if x>self.step else x
        #yStep = self.step if y>self.step else y
        xStep=256
        yStep=256
        #split image into patches if x,y 
        #greater than step size
        patches=[]
        for i in range(0, y, yStep):
            row=[image[:,j:j+xStep,i:i+yStep,:] for j in range(0, x, xStep)]
            patches.append(row)
        probs=[]
        for i in range(len(patches)):
            row=[self.model.predict(img) for img in patches[i]]
            probs.append(row)
            
        probs=np.dstack([np.hstack(p) for p in probs])
        prediction=tf.cast((probs>self.threshold), tf.float32)
        dice = [diceCoef(mask[:,:,:,i] ,prediction[:,:,:,i]) 
               for i in range(mask.shape[-1])]
        iou = [iouScore(mask[:,:,:,i], prediction[:,:,:,i]) 
               for i in range(mask.shape[-1])]
               
        return np.mean(dice), np.mean(iou), prediction


    def __applyNormalization(self, image, mask):
   
        channelMeans = self.normalizeParams['channelMeans']
        channelStd = self.normalizeParams['channelStd']
        norm=Normalize(channelMeans, channelStd)
        data=[(image,mask)]
        
        for method in self.normalize:
            f=lambda x: getattr(norm,'get'+method)(x[0],x[1])
            data=list(map(f, data))

        image,mask=data[0][0],data[0][1]

        return image,mask


    def getTestPredictions(self,dataset, outPath):
        diceLst = []
        iouLst = []
        names = []
        #no=['U_100188_15_B_NA_15_L1',
            #'U_100233_17_X_LOW_9_L2',
            #'U_100233_17_X_LOW_9_L2',
            #'U_90444_4_X_LOW_4_L1',
            #'U_90183_5_X_LOW_4_L1',
            #'100188_01_R']
        for data in dataset:
            image = tf.cast(data[0], tf.float32)
            mask = tf.cast(data[1], tf.float32)
            label = (data[2].numpy()[0]).decode('utf-8')
            #if label in no:
                #continue
            names.append(label)
            print('name',label)
            if len(self.normalize)>0:
                image,mask = self.__applyNormalization(image,mask)
            
            if self.tasktype=='multi':
                mask = tf.cast(mask[:,:,:,0], tf.int32)
                mask = tf.one_hot(mask, depth=3, dtype=tf.float32)
            elif self.tasktype=='binary':
                mask = mask[:,:,:,2:3]
            
            dice,iou,prediction = self.predict(image, mask)
            print('shape:{},Image:{},dice:{}'.format(K.int_shape(image),label,dice)) 
            prediction = prediction.numpy().astype(np.uint8)[0,:,:,:]
            mask  = mask.numpy().astype(np.uint8)[0,:,:,:]

            if self.tasktype=='multi':
                prediction = oneHotToMask(prediction)
            
            outpath = os.path.join(outPath, label +'_pred.png')
            cv2.imwrite(outpath, prediction*int(255))
            diceLst.append(dice)
            iouLst.append(iou)

        return diceLst, iouLst, names


    def __call__(self, path, outPath):
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

        outPath = os.path.join(outPath, self.currentDate)
        try:
            os.mkdir(outPath)
        except Exception as e:
            print(e)
        
        outPath = os.path.join(outPath, self.modelName + '_' + self.currentTime)
        try:
            os.mkdir(outPath)
        except Exception as e:
            print(e)
 
        record = glob.glob(os.path.join(path, self.feature, '*'))
        print('recordpath: {}'.format(os.path.join(path,self.feature)))
        
        dataset = tf.data.TFRecordDataset(record[0])
        dataset = dataset.map(readTF, num_parallel_calls=1)
        dataset = dataset.batch(1)
        dices,ious,names=self.getTestPredictions(dataset,outPath)
    
        imgscores = pd.DataFrame({'image': names, 'dice':dices, 'iou':ious})
        imgscores.to_csv(outPath + '_imgscores.csv')
        
        avgDice=np.mean(dices)
        avgIou=np.mean(ious)
        summary = pd.DataFrame({'dice':[avgDice], 'iou': [avgIou]})
        summary.to_csv(outPath + '_summary.csv')
            
        return avgDice, avgIou

